"""The utility-based agent.

One tick of the loop:

    perceive   fetch a raw bar, fold it into the feature engine
    believe    score the real ensemble -> depeg probability + stress index
    attend     work out which inputs are abnormal; set its own sampling rate
    reason     score every action by expected utility, pick the best
    act        open/escalate/stand down alerts, dispatch
    learn      resolve alerts whose 24h horizon elapsed, re-tune cost weights

The agent never sees the simulator's intent — only the same raw bars a live
exchange feed would deliver. Everything it reports is computed from those bars
by the trained models and the cost model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..config import PEG
from ..features import FeatureEngine
from ..models import DepegEnsemble, Prediction
from .learning import LearningElement
from .state import AgentState, PREDICTION_HORIZON, WATCHED
from .tools import ToolRegistry, query_incident_memory
from .utility import Action, CostModel, Decision, choose

# Sampling cadence in simulated seconds. The agent controls this itself — this
# is the "adaptive sensing" feedback path in the architecture diagram.
POLL_BY_ACTION = {Action.HOLD: 300, Action.WATCH: 60, Action.WARN: 15, Action.CRITICAL: 5}

# Episode control. An alert covers one escalation episode; the agent must go
# quiet for this many consecutive scored ticks before a new episode can begin.
EPISODE_CALM_TICKS = 20
ALERT_COOLDOWN_MIN = 90

# Ground truth for scoring the agent against itself. A breach is a move beyond
# this band; the models' own training target used a volume-adaptive band
# (Carey 2023), but for live outcome labelling we use a fixed, auditable 0.5%.
BREACH_BAND = 0.005


@dataclass
class Tick:
    """Everything produced by one agent step — this is what the UI renders."""
    ts: int
    bar: Dict[str, float]
    warm: bool
    warmup_progress: float
    prediction: Optional[Prediction]
    decision: Optional[Decision]
    attention: Dict[str, Dict[str, float]]
    focus_intensity: float
    poll_interval_s: int
    open_alerts: List[Dict[str, Any]]
    resolved: List[Dict[str, Any]]
    analogue: Optional[Dict[str, Any]]
    learning: Dict[str, Any]
    tools: List[Dict[str, Any]]
    breached: bool
    scored: bool = True          # False when the adaptive-sensing gate skipped inference

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "bar": {k: (round(v, 8) if isinstance(v, float) else v) for k, v in self.bar.items()},
            "warm": self.warm,
            "warmup_progress": round(self.warmup_progress, 4),
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "decision": self.decision.to_dict() if self.decision else None,
            "attention": self.attention,
            "focus_intensity": round(self.focus_intensity, 4),
            "poll_interval_s": self.poll_interval_s,
            "open_alerts": self.open_alerts,
            "resolved": self.resolved,
            "analogue": self.analogue,
            "learning": self.learning,
            "tools": self.tools,
            "breached": self.breached,
            "scored": self.scored,
        }


class DepegAgent:
    """Utility-based agent with a learning element."""

    def __init__(self, ensemble: DepegEnsemble, feed: Callable[[], Optional[Dict[str, float]]],
                 cost: Optional[CostModel] = None, on_dispatch: Optional[Callable] = None):
        self.ensemble = ensemble
        self.engine = FeatureEngine(ensemble.feature_cols)
        self.state = AgentState()
        self.cost = cost or CostModel()
        self.learner = LearningElement(self.cost)
        self.on_dispatch = on_dispatch

        self.tools = ToolRegistry()
        self.tools.register("fetch_market_data", feed,
                            lambda r: "no data" if r is None else f"bar @ close={r['close']:.6f}")
        self.tools.register("compute_features", self._tool_features,
                            lambda r: "warming up" if r is None else f"{len(r)} features")
        self.tools.register("score_models", self._tool_score,
                            lambda r: f"p={r.depeg_proba:.3e} stress={r.stress:.1f}")
        self.tools.register("query_incident_memory", query_incident_memory,
                            lambda r: f"{r['incident']} (sim {r['similarity']:.2f})")
        self.tools.register("dispatch_alert", self._tool_dispatch,
                            lambda r: f"dispatched {r['level']} #{r['id']}")

        self._last_feats: Dict[str, float] = {}
        self._breach_log: List[int] = []     # timestamps of observed breaches
        self._ticks = 0
        # Adaptive sensing: perception is cheap and runs every bar, but the
        # models are only invoked when the agent's own poll schedule says to.
        # While it is calm the agent deliberately looks less often; as it grows
        # concerned it tightens the interval all the way to every bar.
        self._next_score_ts = 0
        self._last_tick: Optional[Tick] = None
        self._scored = 0
        # An alert tracks a whole *episode*, not a minute. Without this the
        # agent reopens an alert every tick for as long as the peg is broken,
        # which both floods the operator and swamps the learning signal.
        self._episode_active = False
        self._calm_run = 0
        self._cooldown_until = 0

    # --------------------------------------------------------------- tool impls
    def _tool_features(self, bar: Dict[str, float]) -> Optional[Dict[str, float]]:
        return self.engine.push(bar)

    def _tool_score(self, feats: Dict[str, float]) -> Prediction:
        vec = self.engine.vector(feats)
        return self.ensemble.predict(vec, feats)

    def _tool_dispatch(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        if self.on_dispatch:
            self.on_dispatch(alert)
        return alert

    # -------------------------------------------------------------------- tick
    def tick(self) -> Optional[Tick]:
        bar = self.tools.call("fetch_market_data")
        if bar is None:
            return None
        ts = int(bar["ts"])
        self._ticks += 1

        feats = self.tools.call("compute_features", ts=ts, bar=bar)
        if feats is None:
            return Tick(ts=ts, bar=bar, warm=False,
                        warmup_progress=self.engine.warmup_progress(),
                        prediction=None, decision=None, attention={}, focus_intensity=0.0,
                        poll_interval_s=POLL_BY_ACTION[Action.HOLD], open_alerts=[],
                        resolved=[], analogue=None, learning=self.learner.snapshot(),
                        tools=self.tools.recent(4), breached=False)

        self._last_feats = feats

        # --- adaptive sensing gate -------------------------------------------
        # Perception above is cheap and always runs, so the rolling windows stay
        # complete. Inference is the expensive step, and how often the agent
        # spends it is itself one of the agent's decisions.
        if ts < self._next_score_ts and self._last_tick is not None:
            prev_tick = self._last_tick
            dev_now = feats["peg_deviation"]
            self.state.note_deviation(dev_now)
            if dev_now >= BREACH_BAND:
                self._breach_log.append(ts)
            skipped = Tick(
                ts=ts, bar=bar, warm=True, warmup_progress=1.0,
                prediction=prev_tick.prediction, decision=prev_tick.decision,
                attention=prev_tick.attention, focus_intensity=prev_tick.focus_intensity,
                poll_interval_s=prev_tick.poll_interval_s,
                open_alerts=[a.to_dict() for a in self.state.open_alerts],
                resolved=[], analogue=prev_tick.analogue,
                learning=self.learner.snapshot(), tools=[],
                breached=dev_now >= BREACH_BAND,
            )
            skipped.scored = False
            return skipped

        pred = self.tools.call("score_models", ts=ts, feats=feats)
        self._scored += 1

        # ---- ground truth bookkeeping -------------------------------------
        dev = feats["peg_deviation"]
        breached = dev >= BREACH_BAND
        if breached:
            self._breach_log.append(ts)
        self.state.note_deviation(dev)
        self.state.observe(feats, pred.depeg_proba)

        # ---- attention -----------------------------------------------------
        attention = self.state.attention(feats)
        intensity = self.state.focus_intensity(attention)

        # ---- reason --------------------------------------------------------
        decision: Decision = choose(pred.depeg_proba, pred.stress, self.cost)

        # ---- act -----------------------------------------------------------
        analogue = None
        if decision.action >= Action.WARN:
            analogue = self.tools.call(
                "query_incident_memory", ts=ts,
                peg_deviation=dev,
                volume_vs_24h=feats["volume_vs_24h"],
                volume_imbalance=feats["volume_imbalance"],
            )

        prev = self.state.current_level
        if decision.action != prev:
            self.state.current_level = decision.action
            self.state.level_since_ts = ts
        self.state.consecutive_elevated = (
            self.state.consecutive_elevated + 1 if decision.action >= Action.WATCH else 0
        )

        # Episode bookkeeping: an episode ends once the agent has been calm
        # (below WARN) for a sustained stretch, which is what makes the next
        # escalation a genuinely new event rather than a continuation.
        if decision.action < Action.WARN:
            self._calm_run += 1
            if self._calm_run >= EPISODE_CALM_TICKS:
                self._episode_active = False
        else:
            self._calm_run = 0

        # Open an alert only on the rising edge of a *new* episode.
        if (decision.action >= Action.WARN and not self.state.open_alerts
                and not self._episode_active and ts >= self._cooldown_until):
            self._episode_active = True
            self._cooldown_until = ts + ALERT_COOLDOWN_MIN * 60
            alert = self.state.open_alert(ts, decision.action, pred.depeg_proba,
                                          pred.stress, decision.rationale)
            payload = alert.to_dict()
            payload["analogue"] = analogue
            self.tools.call("dispatch_alert", ts=ts, alert=payload)

        # ---- learn ---------------------------------------------------------
        resolved_out: List[Dict[str, Any]] = []
        for a in list(self.state.open_alerts):
            hit = any(a.opened_ts <= b <= ts for b in self._breach_log)
            # A true positive is confirmable the moment the peg actually breaks
            # inside the alert's horizon — there is nothing to gain by waiting
            # out the remaining window. Only a *false* positive needs the full
            # 24h to elapse before it can be declared.
            if hit:
                a.resolved, a.resolved_ts, a.outcome = True, ts, "TRUE_POSITIVE"
                self.learner.on_true_positive(ts)
                resolved_out.append(a.to_dict())
            elif ts >= a.expires_ts:
                a.resolved, a.resolved_ts, a.outcome = True, ts, "FALSE_POSITIVE"
                self.learner.on_false_positive(ts)
                resolved_out.append(a.to_dict())

        # A breach while the agent was asleep is the expensive failure.
        if breached and decision.action == Action.HOLD and not self.state.open_alerts:
            if self._ticks % 60 == 0:      # count once per simulated hour, not per minute
                self.learner.on_missed_breach(ts)

        self._prune_breach_log(ts)
        poll = POLL_BY_ACTION[decision.action]
        self._next_score_ts = ts + poll

        tick = Tick(
            ts=ts, bar=bar, warm=True, warmup_progress=1.0,
            prediction=pred, decision=decision,
            attention=attention, focus_intensity=intensity,
            poll_interval_s=poll,
            open_alerts=[a.to_dict() for a in self.state.open_alerts],
            resolved=resolved_out, analogue=analogue,
            learning=self.learner.snapshot(),
            tools=self.tools.recent(6),
            breached=breached,
        )
        self._last_tick = tick
        return tick

    def _prune_breach_log(self, ts: int) -> None:
        cutoff = ts - (PREDICTION_HORIZON + 60) * 60
        if len(self._breach_log) > 4000:
            self._breach_log = [b for b in self._breach_log if b >= cutoff]

    # ------------------------------------------------------------------ status
    def status(self) -> Dict[str, Any]:
        return {
            "ticks": self._ticks,
            "model_invocations": self._scored,
            "sensing_efficiency": round(1.0 - (self._scored / self._ticks), 4) if self._ticks else 0.0,
            "warm": self.engine.warm,
            "warmup_progress": round(self.engine.warmup_progress(), 4),
            "level": self.state.current_level.label,
            "open_alerts": len(self.state.open_alerts),
            "total_alerts": len(self.state.alerts),
            "threshold": self.ensemble.threshold,
            "cost_model": self.cost.to_dict(),
            "learning": self.learner.stats.to_dict(),
            "tools": self.tools.names(),
            "tool_counts": dict(self.tools.counts),
            "watched_metrics": list(WATCHED),
        }
