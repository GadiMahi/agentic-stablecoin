"""Real-data sources (Binance + CoinGecko), tested against a fake server.

No network is used: ``get`` is replaced by a function that answers like the
real endpoints. The key properties checked are the ones the rest of the system
relies on: the exact 12-field raw-bar schema, a continuous minute series, a
full 24h warm-up, sensible fallbacks, and that the agent runs on the result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sthirasense.config import WARMUP_BARS                      # noqa: E402
from sthirasense.sources import binance, coingecko              # noqa: E402
from sthirasense.sources.feeds import (LiveFeed, RealSource,    # noqa: E402
                                       ReplayFeed, iso_to_ts)
from sthirasense.sources.http import SourceError               # noqa: E402

ALLOWED = {"ts", "open", "high", "low", "close", "volume", "quote_volume", "trades",
           "taker_base", "taker_quote", "market_cap", "circulating_supply"}


def kline(ts_s: int, close: float = 1.0, vol: float = 5e4) -> list:
    return [ts_s * 1000, f"{close:.5f}", f"{close + 1e-4:.5f}", f"{close - 1e-4:.5f}",
            f"{close:.5f}", f"{vol}", ts_s * 1000 + 59_999, f"{vol * close}", 70,
            f"{vol / 2}", f"{vol * close / 2}", "0"]


class FakeServer:
    """Answers Binance klines and CoinGecko endpoints from in-memory data."""

    def __init__(self, symbols=("USDCUSDT",), skip=(), cg_fails=False, refuse_first_base=False,
                 price=lambda ts: 1.0, upto=None):
        self.symbols, self.skip, self.cg_fails = set(symbols), set(skip), cg_fails
        self.refuse_first_base, self.price, self.upto = refuse_first_base, price, upto
        self.calls = []

    def __call__(self, url, params=None, headers=None):
        self.calls.append(url)
        if "klines" in url:
            if self.refuse_first_base and "api.binance.com" in url:
                raise SourceError("HTTP 451", status=451)
            if params["symbol"] not in self.symbols:
                return []
            start, end = params["startTime"] // 1000, params["endTime"] // 1000
            if self.upto is not None:
                end = min(end, self.upto())
            out = [kline(t, self.price(t)) for t in range(start, end + 1, 60) if t not in self.skip]
            return out[: params["limit"]]
        if self.cg_fails:
            raise SourceError("HTTP 401 from CoinGecko", status=401)
        if "market_chart/range" in url:
            pts = range(params["from"], params["to"] + 1, 3600)
            return {"prices": [[t * 1000, 1.0] for t in pts],
                    "market_caps": [[t * 1000, 7.0e10 + (t - params["from"])] for t in pts]}
        if "coins/markets" in url:
            return [{"circulating_supply": 7.1e10, "market_cap": 7.1e10, "current_price": 1.0}]
        raise AssertionError(url)


def replay_source(hours: int = 3, supply_estimate=None) -> RealSource:
    return RealSource(key="test_replay", name="Test replay", description="", kind="replay",
                      symbols=("USDCUSDT", "USDCBUSD"), coin_id="usd-coin",
                      start="2025-06-01T00:00:00Z", end=f"2025-06-01T{hours:02d}:00:00Z",
                      supply_estimate=supply_estimate)


# ----------------------------------------------------------------- binance
def test_parse_kline_maps_every_column():
    b = binance.parse_kline(kline(1_700_000_000, 0.99812))
    assert b["ts"] == 1_700_000_000 and b["close"] == pytest.approx(0.99812)
    assert b["trades"] == 70 and b["taker_base"] == pytest.approx(2.5e4)
    assert set(b) == ALLOWED - {"market_cap", "circulating_supply"}


def test_fetch_klines_pages_past_the_1000_bar_limit():
    srv = FakeServer()
    bars = binance.fetch_klines("USDCUSDT", 0, (2500 - 1) * 60_000, get=srv)
    assert len(bars) == 2500
    assert [b["ts"] for b in bars] == list(range(0, 2500 * 60, 60))
    assert sum("klines" in c for c in srv.calls) == 3


def test_falls_back_to_binance_vision_mirror_when_api_refuses():
    srv = FakeServer(refuse_first_base=True)
    bars = binance.fetch_klines("USDCUSDT", 0, 59 * 60_000, get=srv)
    assert len(bars) == 60
    assert any("data-api.binance.vision" in c for c in srv.calls)


def test_fill_gaps_makes_a_continuous_minute_series():
    raw = [binance.parse_kline(kline(t)) for t in (0, 60, 240)]
    out = binance.fill_gaps(raw, 0, 300)
    assert [b["ts"] for b in out] == [0, 60, 120, 180, 240, 300]
    assert out[2]["volume"] == 0 and out[2]["close"] == out[1]["close"]


# --------------------------------------------------------------- coingecko
def test_supply_is_market_cap_over_price_and_held_between_points():
    pts = coingecko.supply_history("usd-coin", 0, 7200, get=FakeServer())
    s = coingecko.SupplySeries(pts, "coingecko")
    assert s.at(0) == pytest.approx(7.0e10)
    assert s.at(3599) == pytest.approx(7.0e10)          # held until the next point
    assert s.at(3600) == pytest.approx(7.0e10 + 3600)


# ------------------------------------------------------------------ replay
def test_replay_emits_the_simulator_schema_with_full_warmup(tmp_path):
    feed = ReplayFeed(replay_source(), get=FakeServer(), cache_dir=tmp_path)
    assert feed.n_warm == WARMUP_BARS
    bars = list(iter(feed, None))
    assert len(bars) == WARMUP_BARS + 3 * 60 + 1
    assert all(set(b) == ALLOWED for b in bars)
    assert all(b2["ts"] - b1["ts"] == 60 for b1, b2 in zip(bars, bars[1:]))
    assert bars[-1]["market_cap"] == pytest.approx(bars[-1]["circulating_supply"] * bars[-1]["close"])
    assert feed.finished and feed.next_bar() is None
    assert feed.info()["supply_source"] == "coingecko"


def test_replay_tries_the_next_pair_when_the_first_has_no_data(tmp_path):
    feed = ReplayFeed(replay_source(), get=FakeServer(symbols=("USDCBUSD",)), cache_dir=tmp_path)
    assert feed.symbol == "USDCBUSD"


def test_replay_reports_clearly_when_no_pair_has_data(tmp_path):
    with pytest.raises(SourceError, match="No Binance pair"):
        ReplayFeed(replay_source(), get=FakeServer(symbols=()), cache_dir=tmp_path)


def test_replay_uses_the_documented_estimate_when_coingecko_history_is_unavailable(tmp_path):
    feed = ReplayFeed(replay_source(supply_estimate=41e9), get=FakeServer(cg_fails=True),
                      cache_dir=tmp_path)
    assert feed.supply.source == "estimate"
    assert feed.bars[0]["circulating_supply"] == 41e9


def test_replay_is_cached_and_then_works_offline(tmp_path):
    srv = FakeServer()
    ReplayFeed(replay_source(), get=srv, cache_dir=tmp_path)
    n = len(srv.calls)

    def offline(*a, **k):
        raise SourceError("no network")

    feed = ReplayFeed(replay_source(), get=offline, cache_dir=tmp_path)
    assert len(srv.calls) == n and len(feed.bars) == WARMUP_BARS + 181


def test_agent_runs_end_to_end_on_replayed_real_bars(tmp_path):
    from sthirasense.agent.agent import DepegAgent
    from sthirasense.models import DepegEnsemble

    start = iso_to_ts("2025-06-01T00:00:00Z")
    # A real-shaped dip: 1.5% below the peg in the second hour.
    srv = FakeServer(price=lambda t: 0.985 if start + 3600 <= t < start + 7200 else 1.0)
    agent = DepegAgent(DepegEnsemble(), ReplayFeed(replay_source(), get=srv, cache_dir=tmp_path))
    scored = []
    while (t := agent.tick()) is not None:
        if t.scored and t.prediction:
            scored.append(t)
    assert scored, "the models never ran on replayed data"
    assert any(t.breached for t in scored)


# -------------------------------------------------------------------- live
class FakeClock:
    def __init__(self, t: float):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def test_live_feed_warms_up_then_waits_for_each_closed_minute():
    clock = FakeClock(1_750_000_030.0)
    srv = FakeServer(upto=lambda: int(clock.t) // 60 * 60 - 60)    # only closed minutes exist
    feed = LiveFeed(RealSource(key="live", name="Live", description="", kind="live",
                               symbols=("USDCUSDT",), coin_id="usd-coin"),
                    get=srv, clock=clock, sleep=clock.sleep)
    warm = [feed.next_bar() for _ in range(WARMUP_BARS)]
    assert all(set(b) == ALLOWED for b in warm) and feed.warming is False
    t0 = clock.t
    bar = feed.next_bar()                               # blocks until the next minute closes
    assert bar["ts"] == warm[-1]["ts"] + 60
    assert clock.t >= bar["ts"] + 60 > t0 - 60
    nxt = feed.next_bar()
    assert nxt["ts"] == bar["ts"] + 60


def test_live_feed_stops_promptly():
    clock = FakeClock(1_750_000_030.0)
    feed = LiveFeed(RealSource(key="live", name="Live", description="", kind="live",
                               symbols=("USDCUSDT",), coin_id="usd-coin"),
                    get=FakeServer(upto=lambda: int(clock.t) // 60 * 60 - 60),
                    clock=clock, sleep=clock.sleep)
    for _ in range(WARMUP_BARS):
        feed.next_bar()
    feed.stop()
    assert feed.next_bar() is None


def test_live_feed_explains_when_coingecko_is_unreachable():
    with pytest.raises(SourceError, match="COINGECKO_API_KEY"):
        LiveFeed(RealSource(key="live", name="Live", description="", kind="live",
                            symbols=("USDCUSDT",), coin_id="usd-coin"),
                 get=FakeServer(cg_fails=True), clock=lambda: 1_750_000_030.0)


# --------------------------------------------------------------- registry
def test_dashboard_lists_simulated_replay_and_live_sources():
    from sthirasense.sources import list_sources
    kinds = {s["kind"] for s in list_sources()}
    assert kinds == {"simulated", "replay", "live"}
