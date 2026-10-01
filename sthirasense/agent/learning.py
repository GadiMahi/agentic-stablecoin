"""The learning element.

Requirement 5 claims a *learning* utility-based agent, so the cost weights must
actually move in response to outcomes rather than being constants with a nice
name. This module is what closes that loop.

The rule is deliberately simple and auditable — gradient-free, bounded, and
explainable in one sentence to a reviewer:

* a **false positive** means the agent paid the fatigue cost for nothing, so the
  cost of a false alarm is revised *up* and it becomes more reluctant to escalate;
* a **true positive** confirms escalation was justified, so the cost of a miss is
  revised *up* and the cost of a false alarm decays slightly back down;
* a **missed breach** — the peg broke while the agent sat at HOLD — is the
  expensive failure, and pushes the miss cost up hard.

Adjustments are multiplicative with a learning rate, and ``CostModel.clamp()``
keeps the weights inside sane bounds so a run of unlucky outcomes cannot drive
the agent into permanent silence or permanent panic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .utility import CostModel


@dataclass
class LearningStats:
    true_positives: int = 0
    false_positives: int = 0
    missed_breaches: int = 0
    updates: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        fired = self.true_positives + self.false_positives
        return self.true_positives / fired if fired else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.missed_breaches
        return self.true_positives / actual if actual else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "missed_breaches": self.missed_breaches,
            "updates": self.updates,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "history": self.history[-60:],
        }


class LearningElement:
    """Re-tunes the cost model from resolved alerts."""

    def __init__(self, cost: CostModel, lr: float = 0.08):
        self.cost = cost
        self.lr = lr
        self.stats = LearningStats()

    def _record(self, ts: int, reason: str) -> None:
        self.cost.clamp()
        self.stats.updates += 1
        self.stats.history.append({
            "ts": ts, "reason": reason,
            "miss_base": round(self.cost.miss_base, 2),
            "false_base": round(self.cost.false_base, 2),
            "precision": round(self.stats.precision, 4),
        })

    def on_true_positive(self, ts: int) -> None:
        self.stats.true_positives += 1
        # Escalating was right: raise what a miss is worth, relax the fatigue
        # penalty a little so the agent is not over-deterred next time.
        self.cost.miss_base *= (1.0 + self.lr)
        self.cost.false_base *= (1.0 - self.lr * 0.5)
        self._record(ts, "true_positive")

    def on_false_positive(self, ts: int) -> None:
        self.stats.false_positives += 1
        # Cried wolf: make false alarms more expensive so the bar to escalate rises.
        self.cost.false_base *= (1.0 + self.lr * 1.5)
        self._record(ts, "false_positive")

    def on_missed_breach(self, ts: int) -> None:
        self.stats.missed_breaches += 1
        # The expensive failure. Push hard.
        self.cost.miss_base *= (1.0 + self.lr * 2.5)
        self._record(ts, "missed_breach")

    def snapshot(self) -> Dict[str, object]:
        return {"cost_model": self.cost.to_dict(), "stats": self.stats.to_dict(), "lr": self.lr}
