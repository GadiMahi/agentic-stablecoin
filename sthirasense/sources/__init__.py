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
    # Binance only listed USDC/USDT at 14:00 UTC on 11 March 2023, mid-crisis,
    # so the USDC/SVB depeg has no pre-event data to warm up on. FDUSD's depeg
    # is fully covered: FDUSD/USDT has traded on Binance since 2023.
    "real_fdusd_2025": RealSource(
        key="real_fdusd_2025",
        name="FDUSD depeg, April 2025 (real data)",
        description=(
            "Replay of a real depeg: on 2 April 2025 FDUSD fell to about $0.87 after "
            "its issuer was publicly accused of insolvency, and took about 36 hours "
            "to recover. Binance FDUSD/USDT 1-minute bars from 1 to 5 April 2025 "
            "(UTC), after 24 hours of real warm-up. Downloaded once, then cached."
        ),
        kind="replay",
        symbols=("FDUSDUSDT",),
        coin_id="first-digital-usd",
        start="2025-04-01T00:00:00Z",
        end="2025-04-05T00:00:00Z",
        # CoinGecko's free plans only serve the last 365 days. If that history is
        # unavailable, supply is held at the issuer's stated size at the time:
        # "worth $2.5 billion" (First Digital, quoted by Decrypt, 2 Apr 2025).
        supply_estimate=2.5e9,
        supply_note="FDUSD supply held at ~$2.5B (First Digital via Decrypt, Apr 2025)",
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
