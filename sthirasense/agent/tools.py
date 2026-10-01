"""Tool interface.

The agent does not reach into objects directly — it calls named tools through a
registry, and every invocation is recorded with its arguments, latency and
result summary. That makes tool use an auditable component rather than an
implementation detail, and it means the agent's whole interaction with the world
can be replayed from the log.

Five tools, matching the architecture diagram:

    fetch_market_data()      pull the next raw bar from the feed
    compute_features()       raw bar -> 106-feature vector
    score_models()           feature vector -> ensemble probability + stress
    query_incident_memory()  find the closest historical analogue
    dispatch_alert()         emit an alert to the outside world
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    summary: str
    ms: float
    ts: int

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "args": self.args, "summary": self.summary,
                "ms": round(self.ms, 3), "ts": self.ts}


class ToolRegistry:
    """Named callables with a bounded audit trail."""

    def __init__(self, maxlog: int = 400):
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._summarisers: Dict[str, Callable[[Any], str]] = {}
        self.log: Deque[ToolCall] = deque(maxlen=maxlog)
        self.counts: Dict[str, int] = {}

    def register(self, name: str, fn: Callable[..., Any],
                 summarise: Optional[Callable[[Any], str]] = None) -> None:
        self._tools[name] = fn
        self._summarisers[name] = summarise or (lambda r: type(r).__name__)
        self.counts.setdefault(name, 0)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def call(self, name: str, ts: int = 0, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name!r} (have {self.names()})")
        t0 = time.perf_counter()
        result = self._tools[name](**kwargs)
        dt = (time.perf_counter() - t0) * 1000.0
        self.counts[name] = self.counts.get(name, 0) + 1
        try:
            summary = self._summarisers[name](result)
        except Exception:
            summary = "?"
        self.log.append(ToolCall(name, {k: _brief(v) for k, v in kwargs.items()},
                                 summary, dt, ts))
        return result

    def recent(self, n: int = 12) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in list(self.log)[-n:]]


def _brief(v: Any) -> Any:
    """Keep the audit log readable — never dump a 106-float vector into it."""
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (list, tuple)):
        return f"<{type(v).__name__} len={len(v)}>"
    if isinstance(v, dict):
        return f"<dict keys={len(v)}>"
    return f"<{type(v).__name__}>"


# --------------------------------------------------------------------------
# Incident memory: real historical depeg episodes, used to attach a plain
# language analogue to an escalation. Static reference data, not a prediction.
# --------------------------------------------------------------------------
INCIDENTS: List[Dict[str, Any]] = [
    {
        "name": "USDC / Silicon Valley Bank",
        "date": "2023-03-11",
        "trough": 0.87,
        "signature": {"peg_deviation": 0.13, "volume_vs_24h": 6.0, "volume_imbalance": -0.45},
        "note": "Reserve-backing scare. $3.3B of reserves (~8%) stranded at a failed bank. "
                "Heavy one-sided selling and a 6x volume spike; peg recovered within days.",
    },
    {
        "name": "Terra UST",
        "date": "2022-05-09",
        "trough": 0.10,
        "signature": {"peg_deviation": 0.90, "volume_vs_24h": 12.0, "volume_imbalance": -0.80},
        "note": "Algorithmic death spiral. Supply expanded violently as the peg broke; "
                "never recovered.",
    },
    {
        "name": "USDT brief discount",
        "date": "2022-05-12",
        "trough": 0.95,
        "signature": {"peg_deviation": 0.05, "volume_vs_24h": 4.0, "volume_imbalance": -0.30},
        "note": "Contagion from the UST collapse. Redemptions honoured, peg restored "
                "within 24h.",
    },
    {
        "name": "DAI / USDC collateral contagion",
        "date": "2023-03-11",
        "trough": 0.90,
        "signature": {"peg_deviation": 0.10, "volume_vs_24h": 5.0, "volume_imbalance": -0.35},
        "note": "Second-order break: DAI was heavily USDC-collateralised, so it "
                "tracked USDC down.",
    },
]


def query_incident_memory(peg_deviation: float, volume_vs_24h: float,
                          volume_imbalance: float) -> Dict[str, Any]:
    """Nearest historical analogue by normalised distance over three signals."""
    cur = {"peg_deviation": peg_deviation,
           "volume_vs_24h": volume_vs_24h,
           "volume_imbalance": volume_imbalance}
    best, best_d = None, float("inf")
    for inc in INCIDENTS:
        d = 0.0
        for k, v in inc["signature"].items():
            scale = max(abs(v), 1e-6)
            d += ((cur[k] - v) / scale) ** 2
        d **= 0.5
        if d < best_d:
            best, best_d = inc, d
    # Similarity in [0,1]; distance of 1.0 in normalised units -> 0.5 similarity.
    similarity = 1.0 / (1.0 + best_d)
    return {"incident": best["name"], "date": best["date"], "trough": best["trough"],
            "note": best["note"], "similarity": round(similarity, 4),
            "distance": round(best_d, 4)}
