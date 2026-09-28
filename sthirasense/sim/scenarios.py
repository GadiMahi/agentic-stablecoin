"""Market scenarios.

Each scenario is a trajectory over *raw observable quantities only* — price,
volume, order-flow skew, supply. None of them touch probabilities, stress
scores or agent state. The ML pipeline and the agent see exactly what they
would see from an exchange feed and have to work the rest out themselves.

That separation is the point: if the simulator could nudge the output, the demo
would prove nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List

# Baseline market shape, roughly calibrated to USDC on a major venue.
BASE_SUPPLY = 7.19e10
BASE_VOL = 6.0e4          # base-asset volume per minute
BASE_TRADES = 70


@dataclass
class Shape:
    """What the market looks like at one instant, before noise is applied."""
    price: float = 1.0
    vol_mult: float = 1.0      # multiple of baseline volume
    flow_skew: float = 0.0     # -1 all selling .. +1 all buying
    supply_drift: float = 0.0  # fractional supply change per minute
    note: str = ""


@dataclass
class Scenario:
    key: str
    name: str
    description: str
    duration_min: int
    shape_at: Callable[[int], Shape]
    recovers: bool


def _calm(_m: int) -> Shape:
    return Shape(price=1.0, vol_mult=1.0, flow_skew=0.0, supply_drift=0.0,
                 note="normal operation")


def _svb_bleed(m: int) -> Shape:
    """Reserve-backing scare: a slow bleed with building pressure, then recovery.

    Modelled on the USDC/SVB shape — hours of deterioration with rising volume
    and one-sided selling before the trough, which is the regime an early-warning
    system is actually meant to catch.
    """
    t = m / 60.0  # hours
    if t < 2.0:                       # rumour phase: flow turns, price barely moves
        f = t / 2.0
        return Shape(1.0 - 0.0008 * f, 1.0 + 1.4 * f, -0.18 * f, 0.0, "rumour: flow turning")
    if t < 6.0:                       # disclosure: the bleed
        f = (t - 2.0) / 4.0
        return Shape(1.0 - 0.0008 - 0.075 * (f ** 1.6), 2.4 + 4.6 * f,
                     -0.18 - 0.40 * f, -0.000006 * f, "disclosure: sustained selling")
    if t < 9.0:                       # trough
        f = (t - 6.0) / 3.0
        return Shape(0.9242 - 0.010 * math.sin(f * math.pi), 7.0 - 1.5 * f,
                     -0.58 + 0.10 * f, -0.000004, "trough: peak dislocation")
    f = min(1.0, (t - 9.0) / 5.0)     # recovery as reserves are confirmed
    return Shape(0.9242 + (1.0 - 0.9242) * (f ** 0.65), 5.5 - 4.2 * f,
                 -0.48 + 0.55 * f, 0.0, "recovery: backing confirmed")


def _flash_crash(m: int) -> Shape:
    """Violent break with no build-up — tests reaction speed, not lead time."""
    if m < 45:
        return Shape(1.0, 1.0, 0.0, 0.0, "quiet before")
    if m < 60:
        f = (m - 45) / 15.0
        return Shape(1.0 - 0.052 * (f ** 0.5), 1.0 + 11.0 * f, -0.75 * f, 0.0,
                     "liquidity vacuum")
    if m < 90:
        f = (m - 60) / 30.0
        return Shape(0.948 + 0.004 * f, 12.0 - 4.0 * f, -0.75 + 0.25 * f, 0.0,
                     "arbitrage stepping in")
    f = min(1.0, (m - 90) / 120.0)
    return Shape(0.952 + 0.048 * (f ** 0.5), 8.0 - 7.0 * f, -0.50 + 0.50 * f, 0.0,
                 "mean reversion")


def _terminal_collapse(m: int) -> Shape:
    """Algorithmic death spiral: reflexive, supply-expanding, no recovery."""
    t = m / 60.0
    if t < 1.5:
        f = t / 1.5
        return Shape(1.0 - 0.004 * f, 1.0 + 2.0 * f, -0.20 * f, 0.0, "peg strain")
    if t < 4.0:
        f = (t - 1.5) / 2.5
        return Shape(0.996 - 0.26 * (f ** 1.4), 3.0 + 9.0 * f, -0.30 - 0.45 * f,
                     0.000030 * f, "spiral: supply expanding")
    f = min(1.0, (t - 4.0) / 6.0)
    price = max(0.09, 0.736 * math.exp(-2.4 * f))
    return Shape(price, 12.0 - 6.0 * f, -0.80, 0.000090,
                 "collapse: redemption mechanism failed")


SCENARIOS: Dict[str, Scenario] = {
    "normal": Scenario(
        "normal", "Normal operation",
        "Calm market holding the peg. The contrast case: proves the agent stays "
        "quiet instead of crying wolf.",
        600, _calm, True),
    "svb_bleed": Scenario(
        "svb_bleed", "USDC / SVB-style slow bleed",
        "Reserve-backing scare. Hours of deterioration with rising volume and "
        "one-sided selling before the trough, then recovery. The realistic "
        "early-warning case.",
        900, _svb_bleed, True),
    "flash_crash": Scenario(
        "flash_crash", "Flash crash",
        "Sudden violent break with no build-up. Tests how fast the agent reacts "
        "when there is no lead time to find.",
        300, _flash_crash, True),
    "terminal_collapse": Scenario(
        "terminal_collapse", "Terminal collapse (UST-style)",
        "Algorithmic death spiral with expanding supply. The peg never returns.",
        720, _terminal_collapse, False),
}


def list_scenarios() -> List[Dict[str, object]]:
    return [{"key": s.key, "name": s.name, "description": s.description,
             "duration_min": s.duration_min, "recovers": s.recovers}
            for s in SCENARIOS.values()]
