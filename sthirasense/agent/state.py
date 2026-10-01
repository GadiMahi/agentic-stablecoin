"""Agent state: memory, attention, and the alert lifecycle.

Three jobs.

**Memory.** The agent is not a reflex — it keeps a rolling view of recent
history so it can tell "0.4% below peg, as it has been all morning" apart from
"0.4% below peg, having been flat an hour ago".

**Attention.** The agent decides where to look. Each minute it computes how
abnormal each watched metric is relative to its own recent baseline (a robust
z-score using median/MAD, so a single spike does not poison the baseline), and
publishes the result as a focus map. The dashboard uses this to emphasise the
numbers the agent is actually reacting to — attention is a real output of the
reasoning layer, not a UI decoration.

**Alert lifecycle.** An action at WARN or above opens an alert. The alert stays
open for the model's own prediction horizon (24h) and then resolves as a true or
false positive depending on whether the peg actually broke. Resolved alerts are
what the learning element consumes.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from .utility import Action

# Metrics the agent watches for abnormality. These are the inputs a human risk
# analyst would actually stare at, and all are real engineered features.
WATCHED = (
    "peg_deviation",
    "volume_vs_24h",
    "volume_imbalance",
    "taker_buy_ratio",
    "trade_size_vs_24h",
    "peg_deviation_volatility",
    "close_std_1440m",
    "circulating_supply_percent_change_1d",
)

HUMAN_LABEL = {
    "peg_deviation": "Peg deviation",
    "volume_vs_24h": "Volume vs 24h",
    "volume_imbalance": "Order-flow imbalance",
    "taker_buy_ratio": "Taker buy ratio",
    "trade_size_vs_24h": "Trade size vs 24h",
    "peg_deviation_volatility": "Peg volatility",
    "close_std_1440m": "Price dispersion 24h",
    "circulating_supply_percent_change_1d": "Supply change 1d",
}

BASELINE_BARS = 240          # 4h robust baseline for attention z-scores
PREDICTION_HORIZON = 1440    # the models predict over the next 24h


@dataclass
class Alert:
    id: int
    opened_ts: int
    level: str
    probability: float
    stress: float
    rationale: str
    expires_ts: int
    resolved: bool = False
    outcome: Optional[str] = None       # "TRUE_POSITIVE" | "FALSE_POSITIVE"
    resolved_ts: Optional[int] = None
    peak_deviation: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id, "opened_ts": self.opened_ts, "level": self.level,
            "probability": self.probability, "stress": round(self.stress, 2),
            "rationale": self.rationale, "resolved": self.resolved,
            "outcome": self.outcome, "resolved_ts": self.resolved_ts,
            "peak_deviation": self.peak_deviation,
        }


@dataclass
class AgentState:
    """Everything the agent remembers between ticks."""
    history: Dict[str, Deque[float]] = field(default_factory=dict)
    prob_history: Deque[float] = field(default_factory=lambda: deque(maxlen=720))
    dev_history: Deque[float] = field(default_factory=lambda: deque(maxlen=PREDICTION_HORIZON + 8))
    alerts: List[Alert] = field(default_factory=list)
    _next_alert_id: int = 1
    current_level: Action = Action.HOLD
    level_since_ts: int = 0
    consecutive_elevated: int = 0

    def __post_init__(self) -> None:
        for m in WATCHED:
            self.history.setdefault(m, deque(maxlen=BASELINE_BARS))

    # ------------------------------------------------------------- observation
    def observe(self, feats: Dict[str, float], probability: float) -> None:
        for m in WATCHED:
            if m in feats:
                self.history[m].append(float(feats[m]))
        self.prob_history.append(float(probability))
        self.dev_history.append(float(feats.get("peg_deviation", 0.0)))

    # --------------------------------------------------------------- attention
    def attention(self, feats: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """How abnormal is each watched metric right now?

        Robust z-score: (x − median) / (1.4826 · MAD). The constant makes MAD a
        consistent estimator of sigma for normal data. Falls back to zero when
        the baseline is degenerate (all values identical), which is the common
        case for binary flags during calm markets.
        """
        out: Dict[str, Dict[str, float]] = {}
        for m in WATCHED:
            buf = self.history[m]
            cur = float(feats.get(m, 0.0))
            if len(buf) < 30:
                out[m] = {"value": cur, "z": 0.0, "weight": 0.0, "label": HUMAN_LABEL[m]}
                continue
            vals = list(buf)
            med = statistics.median(vals)
            mad = statistics.median([abs(v - med) for v in vals])
            sigma = 1.4826 * mad
            z = 0.0 if sigma <= 1e-12 else (cur - med) / sigma
            if not math.isfinite(z):
                z = 0.0
            z = max(-12.0, min(12.0, z))
            out[m] = {"value": cur, "z": z, "weight": 0.0, "label": HUMAN_LABEL[m]}

        # Normalise |z| into attention weights summing to 1 (0 if nothing is odd).
        mass = sum(abs(v["z"]) for v in out.values())
        if mass > 1e-9:
            for v in out.values():
                v["weight"] = abs(v["z"]) / mass
        return out

    def focus_intensity(self, attention: Dict[str, Dict[str, float]]) -> float:
        """Overall 0-1 'how hard is the agent concentrating' signal.

        Driven by the single most abnormal metric: one metric at 6 sigma is more
        alarming than eight metrics at 1 sigma.
        """
        if not attention:
            return 0.0
        peak = max(abs(v["z"]) for v in attention.values())
        return float(min(1.0, peak / 6.0))

    # ------------------------------------------------------------ alert cycle
    def open_alert(self, ts: int, action: Action, probability: float,
                   stress: float, rationale: str) -> Alert:
        a = Alert(
            id=self._next_alert_id, opened_ts=ts, level=action.label,
            probability=probability, stress=stress, rationale=rationale,
            expires_ts=ts + PREDICTION_HORIZON * 60,
        )
        self._next_alert_id += 1
        self.alerts.append(a)
        return a

    @property
    def open_alerts(self) -> List[Alert]:
        return [a for a in self.alerts if not a.resolved]

    def resolve_due(self, ts: int, breached_recently: bool) -> List[Alert]:
        """Resolve alerts whose 24h horizon has elapsed.

        An alert is a true positive if the peg actually breached the band at any
        point inside its horizon. ``breached_recently`` is supplied by the agent
        from ground truth it has observed since the alert opened.
        """
        done: List[Alert] = []
        for a in self.open_alerts:
            if ts >= a.expires_ts:
                a.resolved = True
                a.resolved_ts = ts
                a.outcome = "TRUE_POSITIVE" if breached_recently else "FALSE_POSITIVE"
                done.append(a)
        return done

    def note_deviation(self, dev: float) -> None:
        for a in self.open_alerts:
            if dev > a.peak_deviation:
                a.peak_deviation = dev
