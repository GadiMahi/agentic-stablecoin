"""Real-market feeds: a historical replay and a live stream.

Both expose the same interface as the simulator's ``MarketFeed`` (callable,
``next_bar()``, ``info()``, ``warming``, ``finished``) and emit bars with exactly
the same 12 raw fields, so the agent, the feature engine and the dashboard run
on real data without any change. As with the simulator, nothing here computes
a probability, a stress score or a decision.

* ``ReplayFeed`` downloads a past window once (Binance klines + CoinGecko
  supply), caches it under ``data_cache/`` so later runs work offline, and plays
  it back minute by minute at whatever speed the dashboard asks for.
* ``LiveFeed`` warms up on the last 24 hours of real bars, then waits for each
  new minute to close on Binance and hands it to the agent in real time.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import ROOT, WARMUP_BARS
from . import binance, coingecko
from .coingecko import SupplySeries
from .http import SourceError

CACHE_DIR = ROOT / "data_cache"
BAR_KEYS = ("ts", "open", "high", "low", "close", "volume", "quote_volume", "trades",
            "taker_base", "taker_quote", "market_cap", "circulating_supply")


def iso_to_ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def utc_label(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%d %b %Y %H:%M UTC")


@dataclass(frozen=True)
class RealSource:
    """A named real-data source the dashboard can start."""
    key: str
    name: str
    description: str
    kind: str                                   # "replay" | "live"
    symbols: Tuple[str, ...]                    # Binance pairs, tried in order
    coin_id: str                                # CoinGecko id
    start: Optional[str] = None                 # replay window (ISO, UTC)
    end: Optional[str] = None
    supply_estimate: Optional[float] = None     # used only if CoinGecko history is unavailable
    supply_note: str = ""

    @property
    def duration_min(self) -> Optional[int]:
        if self.start and self.end:
            return (iso_to_ts(self.end) - iso_to_ts(self.start)) // 60
        return None


def _full_bar(b: Dict[str, float], supply: float) -> Dict[str, float]:
    bar = {k: b[k] for k in BAR_KEYS[:10]}
    bar["market_cap"] = supply * b["close"]
    bar["circulating_supply"] = supply
    return bar


def _fetch_bars(symbols: Sequence[str], start_s: int, end_s: int,
                get: Optional[Callable[..., Any]]) -> Tuple[str, List[Dict[str, float]]]:
    """First pair in ``symbols`` with enough data for the window."""
    expected = (end_s - start_s) // 60 + 1
    tried: List[str] = []
    unreachable = 0
    for sym in symbols:
        try:
            raw = binance.fetch_klines(sym, start_s * 1000, end_s * 1000, get=get)
        except SourceError as e:
            tried.append(f"{sym}: {e}")
            unreachable += 1
            continue
        bars = binance.fill_gaps(raw, start_s, end_s)
        if raw and len(bars) >= 0.95 * expected and bars[0]["ts"] <= start_s + 3600:
            return sym, bars
        tried.append(f"{sym}: only {len(raw)} of {expected} minutes available")
    if unreachable == len(symbols):
        raise SourceError("Could not reach Binance. Check your internet connection and try "
                          "again. Details: " + "; ".join(tried))
    raise SourceError("No Binance pair had data for this window. " + "; ".join(tried))


class _Base:
    def __init__(self, source: RealSource):
        self.source = source
        self.bars: List[Dict[str, float]] = []
        self.n_warm = 0
        self.pos = 0
        self.symbol = ""
        self.supply: Optional[SupplySeries] = None
        self._finished = False

    @property
    def warming(self) -> bool:
        return self.pos < self.n_warm

    @property
    def scenario_minute(self) -> int:
        return max(0, self.pos - self.n_warm)

    @property
    def finished(self) -> bool:
        return self._finished

    def __call__(self) -> Optional[Dict[str, float]]:
        return self.next_bar()

    def stop(self) -> None:
        self._finished = True

    def _info(self, phase: str, progress: float) -> Dict[str, object]:
        return {
            "scenario": self.source.key, "name": self.source.name, "kind": self.source.kind,
            "warming": self.warming, "minute": self.scenario_minute,
            "duration_min": self.source.duration_min, "progress": round(progress, 4),
            "phase": phase, "finished": self._finished, "symbol": self.symbol,
            "supply_source": self.supply.source if self.supply else None,
        }


class ReplayFeed(_Base):
    """Plays back a real historical window, minute by minute."""

    def __init__(self, source: RealSource, get: Optional[Callable[..., Any]] = None,
                 cache_dir: Path = CACHE_DIR, use_cache: bool = True):
        super().__init__(source)
        assert source.start and source.end, "a replay needs a start and an end"
        start_s, end_s = iso_to_ts(source.start), iso_to_ts(source.end)
        warm_start = start_s - WARMUP_BARS * 60
        data = self._load(cache_dir, use_cache, warm_start, end_s, get)
        self.symbol = data["symbol"]
        self.supply = SupplySeries([tuple(p) for p in data["supply"]], data["supply_source"])
        self.bars = [_full_bar(b, self.supply.at(int(b["ts"]))) for b in data["bars"]]
        self.n_warm = sum(1 for b in self.bars if b["ts"] < start_s)

    def _load(self, cache_dir: Path, use_cache: bool, warm_start: int, end_s: int,
              get: Optional[Callable[..., Any]]) -> Dict[str, Any]:
        path = cache_dir / f"{self.source.key}.json"
        if use_cache and path.exists():
            return json.loads(path.read_text())
        symbol, bars = _fetch_bars(self.source.symbols, warm_start, end_s, get)
        try:
            supply = coingecko.supply_history(self.source.coin_id, warm_start - 7200, end_s, get=get)
            supply_source = "coingecko"
        except SourceError:
            if self.source.supply_estimate is None:
                raise
            # CoinGecko's free plans only reach back 365 days. Hold supply at a
            # documented estimate; the market cap still moves with the price.
            supply = [(warm_start, float(self.source.supply_estimate))]
            supply_source = "estimate"
        data = {"symbol": symbol, "start": self.source.start, "end": self.source.end,
                "bars": bars, "supply": supply, "supply_source": supply_source,
                "supply_note": self.source.supply_note if supply_source == "estimate" else ""}
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return data

    def next_bar(self) -> Optional[Dict[str, float]]:
        if self._finished or self.pos >= len(self.bars):
            self._finished = True
            return None
        bar = self.bars[self.pos]
        self.pos += 1
        if self.pos >= len(self.bars):
            self._finished = True
        return bar

    def progress(self) -> float:
        n = len(self.bars) - self.n_warm
        return min(1.0, self.scenario_minute / n) if n > 0 else 0.0

    def info(self) -> Dict[str, object]:
        i = min(max(self.pos - 1, 0), len(self.bars) - 1)
        phase = f"{utc_label(int(self.bars[i]['ts']))} · {self.symbol}" if self.bars else ""
        if self.warming:
            phase = "warm-up: previous 24h of real bars · " + phase
        return self._info(phase, self.progress())


class LiveFeed(_Base):
    """Real-time bars: 24h of history to warm up, then each new closed minute."""

    def __init__(self, source: RealSource, get: Optional[Callable[..., Any]] = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 poll_s: float = 2.0, grace_s: float = 3.0, supply_refresh_s: float = 600.0):
        super().__init__(source)
        self.get, self.clock, self.sleep = get, clock, sleep
        self.poll_s, self.grace_s, self.supply_refresh_s = poll_s, grace_s, supply_refresh_s
        self._stop = threading.Event()

        now = int(clock())
        last_closed = (now // 60) * 60 - 60          # open time of the last fully closed minute
        warm_start = last_closed - (WARMUP_BARS - 1) * 60
        self.symbol, raw = _fetch_bars(source.symbols, warm_start, last_closed, get)
        self.supply = self._initial_supply(warm_start, now)
        self._last_supply_check = now
        self.bars = [_full_bar(b, self.supply.at(int(b["ts"]))) for b in raw]
        self.n_warm = len(self.bars)
        self.last_ts = int(self.bars[-1]["ts"])

    def _initial_supply(self, warm_start: int, now: int) -> SupplySeries:
        try:
            pts = coingecko.supply_history(self.source.coin_id, warm_start - 7200, now, get=self.get)
            return SupplySeries(pts, "coingecko")
        except SourceError:
            pass
        try:
            supply, _ = coingecko.current_supply(self.source.coin_id, get=self.get)
            return SupplySeries([(warm_start, supply)], "coingecko")
        except SourceError as e:
            raise SourceError("CoinGecko could not be reached for circulating supply "
                              f"({e}). Try again, or set COINGECKO_API_KEY.") from e

    def stop(self) -> None:
        self._stop.set()
        super().stop()

    def _refresh_supply(self, now: int) -> None:
        if now - self._last_supply_check < self.supply_refresh_s:
            return
        self._last_supply_check = now
        try:
            supply, _ = coingecko.current_supply(self.source.coin_id, get=self.get)
            self.supply.add(now, supply)
        except SourceError:
            pass                                   # keep the last known value

    def _emit(self, b: Dict[str, float]) -> Dict[str, float]:
        bar = _full_bar(b, self.supply.at(int(b["ts"])))
        self.bars.append(bar)
        if len(self.bars) > WARMUP_BARS + 10:      # memory stays bounded in a long session
            drop = len(self.bars) - WARMUP_BARS
            del self.bars[:drop]
            self.n_warm -= drop
            self.pos -= drop
        self.pos += 1
        self.last_ts = int(b["ts"])
        return bar

    def next_bar(self) -> Optional[Dict[str, float]]:
        if self.pos < len(self.bars):              # still replaying the 24h warm-up
            bar = self.bars[self.pos]
            self.pos += 1
            return bar
        target = self.last_ts + 60
        while not self._stop.is_set():
            now = self.clock()
            wait = target + 60 + self.grace_s - now   # the minute must have closed
            if wait > 0:
                self.sleep(min(self.poll_s, wait))
                continue
            self._refresh_supply(int(now))
            try:
                raw = binance.fetch_klines(self.symbol, target * 1000, (target + 600) * 1000, get=self.get)
            except SourceError:
                self.sleep(self.poll_s * 5)
                continue
            closed = [b for b in raw if b["ts"] + 60 <= now]
            if closed and int(closed[0]["ts"]) == target:
                return self._emit(closed[0])
            if closed or now > target + 180:
                # No trade in that minute: carry the previous close forward.
                return self._emit(binance.fill_gaps([self._last_raw()], self.last_ts, target)[-1])
            self.sleep(self.poll_s)
        return None

    def _last_raw(self) -> Dict[str, float]:
        return {k: self.bars[-1][k] for k in BAR_KEYS[:10]}

    def info(self) -> Dict[str, object]:
        if self.warming:
            phase = f"warm-up: previous 24h of real bars · {self.symbol}"
        else:
            phase = f"live · {utc_label(self.last_ts)} · {self.symbol}"
        return self._info(phase, 0.0)
