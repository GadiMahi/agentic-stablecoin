"""Bar generator.

Turns a scenario trajectory into realistic OHLCV + market-cap bars, including
the 24h of warm-up history the feature engine needs before it can emit a
complete vector.

The generator is the *only* thing in the system that knows what scenario is
running. Everything downstream — feature engine, ensemble, agent — receives
nothing but bars.
"""
from __future__ import annotations

import random
from typing import Dict, Iterator, Optional

from ..config import WARMUP_BARS
from .scenarios import BASE_SUPPLY, BASE_TRADES, BASE_VOL, SCENARIOS, Scenario, Shape


class MarketFeed:
    """Emits one raw bar per call. Warm-up bars come first, then the scenario."""

    def __init__(self, scenario_key: str = "normal", seed: int = 42,
                 start_ts: int = 1_760_000_000):
        if scenario_key not in SCENARIOS:
            raise KeyError(f"unknown scenario {scenario_key!r}")
        self.scenario: Scenario = SCENARIOS[scenario_key]
        self.rng = random.Random(seed)
        self.start_ts = start_ts

        self.i = -WARMUP_BARS            # negative while warming up
        self.supply = BASE_SUPPLY
        self.prev_close = 1.0
        self._finished = False

    # ------------------------------------------------------------------ state
    @property
    def warming(self) -> bool:
        return self.i < 0

    @property
    def scenario_minute(self) -> int:
        return max(0, self.i)

    @property
    def finished(self) -> bool:
        return self._finished

    def progress(self) -> float:
        if self.warming:
            return 0.0
        return min(1.0, self.scenario_minute / self.scenario.duration_min)

    # -------------------------------------------------------------------- bar
    def _shape(self) -> Shape:
        if self.warming:
            return Shape(1.0, 1.0, 0.0, 0.0, "warm-up")
        return self.scenario.shape_at(self.scenario_minute)

    def next_bar(self) -> Optional[Dict[str, float]]:
        if self._finished:
            return None

        shape = self._shape()
        rng = self.rng

        # --- price: pull toward the scenario target, with micro-noise --------
        # Mean-reverting step toward the trajectory, so the series is continuous
        # rather than teleporting between target levels.
        noise = rng.gauss(0.0, 2.2e-5 * (1.0 + 7.0 * abs(shape.flow_skew)))
        close = self.prev_close + 0.22 * (shape.price - self.prev_close) + noise
        close = max(0.02, close)
        open_ = self.prev_close
        wick = abs(rng.gauss(0.0, 3.0e-5 * (1.0 + 9.0 * abs(shape.flow_skew))))
        high = max(open_, close) + wick
        low = min(open_, close) - wick
        self.prev_close = close

        # --- volume ----------------------------------------------------------
        vol = BASE_VOL * shape.vol_mult * rng.lognormvariate(0.0, 0.30)
        trades = max(1, int(BASE_TRADES * shape.vol_mult * rng.lognormvariate(0.0, 0.22)))
        quote_vol = vol * close

        # --- order flow ------------------------------------------------------
        # flow_skew < 0 means sellers dominate: taker *buy* volume falls.
        buy_frac = min(0.95, max(0.05, 0.5 + 0.42 * shape.flow_skew + rng.gauss(0, 0.03)))
        taker_base = vol * buy_frac
        taker_quote = quote_vol * buy_frac

        # --- supply / market cap --------------------------------------------
        self.supply *= (1.0 + shape.supply_drift + rng.gauss(0.0, 5e-7))
        mcap = self.supply * close

        bar: Dict[str, float] = {
            "ts": self.start_ts + (self.i + WARMUP_BARS) * 60,
            "open": open_, "high": high, "low": low, "close": close,
            "volume": vol, "quote_volume": quote_vol, "trades": trades,
            "taker_base": taker_base, "taker_quote": taker_quote,
            "market_cap": mcap, "circulating_supply": self.supply,
        }

        self.i += 1
        if not self.warming and self.scenario_minute > self.scenario.duration_min:
            self._finished = True
        return bar

    def __call__(self) -> Optional[Dict[str, float]]:
        """Callable form, so the agent can register this directly as a tool."""
        return self.next_bar()

    def __iter__(self) -> Iterator[Dict[str, float]]:
        while True:
            b = self.next_bar()
            if b is None:
                return
            yield b

    def info(self) -> Dict[str, object]:
        return {
            "scenario": self.scenario.key,
            "name": self.scenario.name,
            "warming": self.warming,
            "minute": self.scenario_minute,
            "duration_min": self.scenario.duration_min,
            "progress": round(self.progress(), 4),
            "phase": self._shape().note,
            "finished": self._finished,
        }
