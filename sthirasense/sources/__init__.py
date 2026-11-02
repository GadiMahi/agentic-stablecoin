"""Market data sources: the simulator plus real data from Binance and CoinGecko.

``build_feed(key)`` returns the feed for any source the dashboard lists, and
``list_sources()`` describes them all. Every feed emits the same 12 raw fields.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..sim.engine import MarketFeed
from ..sim.scenarios import SCENARIOS, list_scenarios
from .feeds import LiveFeed, RealSource, ReplayFeed
from .http import SourceError

REAL_SOURCES: Dict[str, RealSource] = {
    "real_usdc_svb_2023": RealSource(
        key="real_usdc_svb_2023",
        name="USDC / SVB, March 2023 (real data)",
        description=(
            "Replay of the real USDC depeg after Silicon Valley Bank failed: Binance "
            "1-minute bars from 10 to 14 March 2023 (UTC), preceded by 24 hours of "
            "real warm-up. Downloaded once, then cached for offline use."
        ),
        kind="replay",
        symbols=("USDCUSDT", "USDCBUSD"),
        coin_id="usd-coin",
        start="2023-03-10T00:00:00Z",
        end="2023-03-14T00:00:00Z",
        # CoinGecko's free plans only serve the last 365 days. If that history is
        # unavailable, supply is held at USDC's level just before the event:
        # "just north of $41 billion on March 8" (The Block, 11 Mar 2023).
        supply_estimate=41.0e9,
        supply_note="USDC supply held at ~$41B (The Block, 8 Mar 2023)",
    ),
    "live_usdc": RealSource(
        key="live_usdc",
        name="USDC live (Binance + CoinGecko)",
        description=(
            "Live USDC/USDT 1-minute bars from Binance with circulating supply from "
            "CoinGecko. Warms up on the last 24 hours, then the agent sees each new "
            "minute as it closes. Speed has no effect in live mode."
        ),
        kind="live",
        symbols=("USDCUSDT",),
        coin_id="usd-coin",
    ),
}


def list_sources() -> List[Dict[str, object]]:
    sims = [{**s, "kind": "simulated"} for s in list_scenarios()]
    real = [{"key": r.key, "name": r.name, "description": r.description,
             "duration_min": r.duration_min, "recovers": None, "kind": r.kind}
            for r in REAL_SOURCES.values()]
    return sims + real


def is_known(key: str) -> bool:
    return key in SCENARIOS or key in REAL_SOURCES


def build_feed(key: str, seed: int = 42, get=None):
    """The feed for ``key``. Real sources make network calls (or read the cache)."""
    if key in SCENARIOS:
        return MarketFeed(key, seed=seed)
    src: Optional[RealSource] = REAL_SOURCES.get(key)
    if src is None:
        raise KeyError(f"unknown source {key!r}")
    return ReplayFeed(src, get=get) if src.kind == "replay" else LiveFeed(src, get=get)


__all__ = ["REAL_SOURCES", "RealSource", "ReplayFeed", "LiveFeed", "SourceError",
           "build_feed", "list_sources", "is_known"]
