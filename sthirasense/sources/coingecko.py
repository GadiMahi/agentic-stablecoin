"""CoinGecko: market capitalisation and circulating supply.

Binance has no supply data, so CoinGecko supplies the two remaining raw
fields. Supply moves slowly (issuance and redemptions), so hourly points held
constant within the hour are plenty for the 1-day supply-change feature.

A free Demo API key is optional: set ``COINGECKO_API_KEY`` to raise the rate
limit. Without a key the public endpoint works at a lower rate. On the free
plans historical data only reaches back 365 days.
"""
from __future__ import annotations

import bisect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import http
from .http import SourceError

BASE = "https://api.coingecko.com/api/v3"


def _headers() -> Dict[str, str]:
    key = os.environ.get("COINGECKO_API_KEY", "").strip()
    return {"x-cg-demo-api-key": key} if key else {}


def current_supply(coin_id: str, get: Optional[Callable[..., Any]] = None) -> Tuple[float, float]:
    """(circulating_supply, market_cap_usd) right now."""
    get = get or http.get_json
    data = get(f"{BASE}/coins/markets", {"vs_currency": "usd", "ids": coin_id}, _headers())
    if not data:
        raise SourceError(f"CoinGecko: unknown coin id {coin_id!r}")
    row = data[0]
    supply = row.get("circulating_supply")
    if not supply:
        raise SourceError(f"CoinGecko: no circulating supply for {coin_id!r}")
    return float(supply), float(row.get("market_cap") or 0.0)


def supply_history(coin_id: str, start_s: int, end_s: int,
                   get: Optional[Callable[..., Any]] = None) -> List[Tuple[int, float]]:
    """[(unix_seconds, circulating_supply)] between start_s and end_s.

    CoinGecko's range endpoint returns market caps and prices; supply is
    market_cap / price at each timestamp.
    """
    get = get or http.get_json
    data = get(f"{BASE}/coins/{coin_id}/market_chart/range",
               {"vs_currency": "usd", "from": int(start_s), "to": int(end_s)}, _headers())
    prices = {int(t) // 1000: float(p) for t, p in data.get("prices", []) if p}
    points: List[Tuple[int, float]] = []
    for t, mcap in data.get("market_caps", []):
        ts = int(t) // 1000
        price = prices.get(ts)
        if price and mcap:
            points.append((ts, float(mcap) / price))
    if not points:
        raise SourceError(f"CoinGecko: no market-cap history for {coin_id!r} in that range")
    points.sort()
    return points


class SupplySeries:
    """Circulating supply at any minute, holding the last known point."""

    def __init__(self, points: List[Tuple[int, float]], source: str):
        if not points:
            raise ValueError("SupplySeries needs at least one point")
        self.ts = [p[0] for p in points]
        self.values = [p[1] for p in points]
        self.source = source              # "coingecko" | "estimate"

    def at(self, ts: int) -> float:
        i = bisect.bisect_right(self.ts, ts) - 1
        return self.values[max(i, 0)]

    def add(self, ts: int, value: float) -> None:
        if self.ts and ts <= self.ts[-1]:
            return
        self.ts.append(ts)
        self.values.append(value)
