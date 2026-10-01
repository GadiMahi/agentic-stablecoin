"""The agent's utility model.

This is the core of Requirement 5: the agent does not act on a threshold, it
acts on *expected utility*. Every minute it holds a depeg probability ``p`` and
a liquidity-stress score, and must choose one action from a graded set. Each
action is scored by what it would cost in both worlds — the one where a depeg
is coming and the one where it is not — and the best-scoring action wins.

    EU(a) = − p·C_miss(a) − (1−p)·C_false(a) − C_op(a)

Why this rather than a goal-based rule: "never miss a depeg" is satisfied by an
agent that alerts constantly, which destroys its own usefulness. The two ways of
being wrong are asymmetric and both real, so the decision is a trade-off, not a
goal test. That asymmetry is what the cost model below encodes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import IntEnum
from typing import Dict, List


class Action(IntEnum):
    """Graded response levels. Ordering is meaningful — higher means louder."""
    HOLD = 0        # stay quiet, sample lazily
    WATCH = 1       # internal elevated monitoring, no human notified
    WARN = 2        # notify an operator
    CRITICAL = 3    # escalate hard

    @property
    def label(self) -> str:
        return self.name


# How much of a real depeg each action is assumed to "cover" — i.e. how much of
# the loss it averts by being taken in time. HOLD averts nothing.
COVERAGE = {Action.HOLD: 0.0, Action.WATCH: 0.35, Action.WARN: 0.80, Action.CRITICAL: 1.0}

# Operator-fatigue multiplier: how costly this action is if nothing happens.
# WATCH is nearly free because no human is interrupted; CRITICAL is expensive
# because crying wolf at that level is what gets an alert system switched off.
FATIGUE = {Action.HOLD: 0.0, Action.WATCH: 0.15, Action.WARN: 1.0, Action.CRITICAL: 3.2}

# Resource cost of performing the action (compute, polling, dispatch).
OP_COST = {Action.HOLD: 0.0, Action.WATCH: 0.40, Action.WARN: 1.10, Action.CRITICAL: 2.40}


@dataclass
class CostModel:
    """Tunable cost weights. The learning element adjusts these in place."""
    # Calibrated against the scale the ensemble actually operates on. The
    # meta-learner is precision-tuned and emits small probabilities (its tuned
    # threshold is 1.8e-3), so the cost of a miss has to be large relative to
    # the cost of a false alarm for escalation ever to be rational. These
    # defaults make the full HOLD -> WATCH -> WARN -> CRITICAL ladder reachable
    # across the observed probability range without firing on a calm market.
    miss_base: float = 4000.0     # cost of a fully-missed depeg
    false_base: float = 30.0      # cost of a false alarm at WARN level
    stress_gain: float = 1.0      # how much stress amplifies the cost of a miss
    min_miss: float = 1000.0
    max_miss: float = 20000.0
    min_false: float = 5.0
    max_false: float = 400.0

    def severity(self, stress: float) -> float:
        """Stress amplifies what a miss costs: a break during an already
        stressed market does more damage than one from a calm book."""
        return 1.0 + self.stress_gain * (stress / 50.0)

    def clamp(self) -> None:
        self.miss_base = float(min(max(self.miss_base, self.min_miss), self.max_miss))
        self.false_base = float(min(max(self.false_base, self.min_false), self.max_false))

    def to_dict(self) -> Dict[str, float]:
        return {"miss_base": round(self.miss_base, 2),
                "false_base": round(self.false_base, 2),
                "stress_gain": round(self.stress_gain, 3)}


@dataclass
class ActionScore:
    action: Action
    expected_utility: float
    cost_if_miss: float
    cost_if_false_alarm: float
    op_cost: float

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["action"] = self.action.label
        d["expected_utility"] = round(self.expected_utility, 4)
        d["cost_if_miss"] = round(self.cost_if_miss, 3)
        d["cost_if_false_alarm"] = round(self.cost_if_false_alarm, 3)
        d["op_cost"] = round(self.op_cost, 3)
        return d


@dataclass
class Decision:
    """The full reasoning trace for one minute — what the agent chose and why."""
    action: Action
    probability: float
    stress: float
    scores: List[ActionScore] = field(default_factory=list)
    margin: float = 0.0            # utility gap to the runner-up
    rationale: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": self.action.label,
            "probability": self.probability,
            "stress": round(self.stress, 2),
            "scores": [s.to_dict() for s in self.scores],
            "margin": round(self.margin, 4),
            "rationale": self.rationale,
        }


def score_actions(p: float, stress: float, cost: CostModel) -> List[ActionScore]:
    """Score every available action under the current belief."""
    sev = cost.severity(stress)
    out: List[ActionScore] = []
    for a in Action:
        c_miss = cost.miss_base * sev * (1.0 - COVERAGE[a])
        c_false = cost.false_base * FATIGUE[a]
        c_op = OP_COST[a]
        eu = -(p * c_miss) - ((1.0 - p) * c_false) - c_op
        out.append(ActionScore(a, eu, c_miss, c_false, c_op))
    return out


def choose(p: float, stress: float, cost: CostModel) -> Decision:
    """argmax over expected utility."""
    scores = score_actions(p, stress, cost)
    ranked = sorted(scores, key=lambda s: s.expected_utility, reverse=True)
    best, runner_up = ranked[0], ranked[1]
    margin = best.expected_utility - runner_up.expected_utility

    if best.action is Action.HOLD:
        rationale = (f"p={p:.2e} too low to justify acting; staying quiet beats "
                     f"{runner_up.action.label} by {margin:.2f} utility")
    else:
        rationale = (f"at p={p:.2e} and stress {stress:.1f}, the expected cost of "
                     f"missing ({best.cost_if_miss:,.0f}) outweighs the cost of a "
                     f"false alarm ({best.cost_if_false_alarm:,.1f})")

    return Decision(action=best.action, probability=p, stress=stress,
                    scores=scores, margin=margin, rationale=rationale)
