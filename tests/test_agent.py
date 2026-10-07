"""Agent behaviour tests.

These assert the properties that make the agent an *agent* rather than a
threshold with extra steps:

* it stays quiet on a calm market (the property most likely to be lost by a
  careless cost-model change, and the one that makes the system usable);
* it escalates when the real models say risk is rising;
* an alert covers an episode, not a minute;
* the learning element actually moves the cost weights, in the right direction;
* adaptive sensing genuinely reduces inference while keeping perception complete.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sthirasense.agent.agent import DepegAgent                      # noqa: E402
from sthirasense.agent.learning import LearningElement              # noqa: E402
from sthirasense.agent.utility import Action, CostModel, choose     # noqa: E402
from sthirasense.models import DepegEnsemble                        # noqa: E402
from sthirasense.sim.engine import MarketFeed                       # noqa: E402


@pytest.fixture(scope="module")
def ens():
    return DepegEnsemble()


def run(ens, scenario: str, seed: int = 11):
    feed = MarketFeed(scenario, seed=seed)
    agent = DepegAgent(ens, feed)
    actions, scored = {}, 0
    while True:
        t = agent.tick()
        if t is None:
            break
        if t.warm and t.scored and t.decision:
            scored += 1
            a = t.decision.action.label
            actions[a] = actions.get(a, 0) + 1
    return agent, actions, scored


# --------------------------------------------------------------- utility model
def test_calm_market_prefers_silence():
    """At a probability below threshold with low stress, HOLD must win."""
    d = choose(p=1.0e-3, stress=5.0, cost=CostModel())
    assert d.action is Action.HOLD


def test_rising_risk_escalates_monotonically():
    """Holding stress fixed, raising probability must never de-escalate."""
    cost = CostModel()
    last = -1
    for p in (1e-4, 1e-3, 5e-3, 1e-2, 3e-2, 8e-2):
        a = int(choose(p, stress=60.0, cost=cost).action)
        assert a >= last, f"de-escalated at p={p}"
        last = a


def test_stress_amplifies_the_cost_of_a_miss():
    cost = CostModel()
    calm = choose(2e-2, stress=5.0, cost=cost).action
    stressed = choose(2e-2, stress=95.0, cost=cost).action
    assert int(stressed) >= int(calm)


def test_every_action_is_scored_not_just_the_winner():
    d = choose(1e-2, stress=40.0, cost=CostModel())
    assert len(d.scores) == len(Action)
    assert d.margin >= 0.0
    assert d.rationale


# ------------------------------------------------------------ learning element
def test_false_positive_makes_the_agent_more_reluctant():
    cost = CostModel()
    before = cost.false_base
    LearningElement(cost).on_false_positive(ts=0)
    assert cost.false_base > before


def test_true_positive_raises_the_cost_of_a_miss():
    cost = CostModel()
    before = cost.miss_base
    LearningElement(cost).on_true_positive(ts=0)
    assert cost.miss_base > before


def test_missed_breach_is_the_strongest_correction():
    a, b = CostModel(), CostModel()
    LearningElement(a).on_true_positive(ts=0)
    LearningElement(b).on_missed_breach(ts=0)
    assert b.miss_base > a.miss_base


def test_weights_stay_bounded_under_a_long_bad_run():
    cost = CostModel()
    le = LearningElement(cost)
    for _ in range(500):
        le.on_false_positive(ts=0)
    assert cost.false_base <= cost.max_false
    for _ in range(500):
        le.on_missed_breach(ts=0)
    assert cost.miss_base <= cost.max_miss


# ------------------------------------------------------------- end-to-end runs
def test_calm_scenario_raises_no_alerts(ens):
    agent, actions, scored = run(ens, "normal")
    assert scored > 0, "agent never scored anything"
    assert set(actions) == {"HOLD"}, f"agent broke silence on a calm market: {actions}"
    assert len(agent.state.alerts) == 0


def test_depeg_scenarios_escalate_and_alert(ens):
    for scenario in ("svb_bleed", "flash_crash", "terminal_collapse"):
        agent, actions, _ = run(ens, scenario)
        elevated = sum(v for k, v in actions.items() if k != "HOLD")
        assert elevated > 0, f"{scenario}: agent never escalated"
        assert len(agent.state.alerts) >= 1, f"{scenario}: no alert raised"


def test_alert_covers_an_episode_not_a_minute(ens):
    """Reopening per-tick produced hundreds of alerts for one event."""
    agent, _, _ = run(ens, "svb_bleed")
    assert len(agent.state.alerts) <= 5, (
        f"alert storm: {len(agent.state.alerts)} alerts for a single episode")


def test_learning_loop_closes_on_a_real_breach(ens):
    agent, _, _ = run(ens, "terminal_collapse")
    st = agent.learner.stats
    assert st.updates > 0, "learning element never fired"
    assert st.true_positives >= 1
    assert agent.cost.miss_base != CostModel().miss_base, "cost weights never moved"


def test_adaptive_sensing_saves_inference_without_losing_perception(ens):
    agent, _, scored = run(ens, "normal")
    status = agent.status()
    # Perception is complete...
    assert status["ticks"] > scored
    # ...but the models were invoked far less often than once per bar.
    assert status["sensing_efficiency"] > 0.5
    assert agent.engine.warm


def test_agent_only_sees_raw_bars(ens):
    """The feed must not leak scenario intent into the agent's input."""
    feed = MarketFeed("terminal_collapse", seed=5)
    bar = feed.next_bar()
    allowed = {"ts", "open", "high", "low", "close", "volume", "quote_volume",
               "trades", "taker_base", "taker_quote", "market_cap", "circulating_supply"}
    assert set(bar) == allowed
