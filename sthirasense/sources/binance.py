"""Binance spot klines: price, volume, trades and taker flow per minute.

The kline endpoint returns exactly the order-book observables the models were
trained on, column for column:

    [open_time, open, high, low, close, volume, close_time, quote_volume,
     trades, taker_buy_base, taker_buy_quote, ignore]

No API key is needed. ``api.binance.com`` is tried first; if it refuses the
request (for example HTTP 451 in regions where Binance is restricted), the
public market-data mirror ``data-api.binance.vision`` is used instead.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from . import http
from .http import SourceError

BASE_URLS: Sequence[str] = ("https://api.binance.com", "https://data-api.binance.vision")
KLINES_PATH = "/api/v3/klines"
PAGE = 1000            # Binance's maximum klines per request
MINUTE_MS = 60_000


def parse_kline(row: Sequence[Any]) -> Dict[str, float]:
    """One kline row -> the order-book half of a raw bar (no supply fields)."""
    return {
        "ts": int(row[0]) // 1000,
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
        "quote_volume": float(row[7]),
        "trades": int(row[8]),
        "taker_base": float(row[9]),
        "taker_quote": float(row[10]),
    }


def _get(params: Dict[str, Any], get: Callable[..., Any]) -> List[List[Any]]:
    last: Optional[SourceError] = None
    for base in BASE_URLS:
        try:
            data = get(base + KLINES_PATH, params)
        except SourceError as e:
            last = e
            continue
        if isinstance(data, dict):             # Binance error object, e.g. invalid symbol
            raise SourceError(f"Binance: {data.get('msg', data)}")
        return data
    raise last or SourceError("Binance unreachable")


def fetch_klines(symbol: str, start_ms: int, end_ms: int,
                 get: Optional[Callable[..., Any]] = None) -> List[Dict[str, float]]:
    """All 1-minute bars with open time in [start_ms, end_ms], oldest first."""
    get = get or http.get_json
    out: List[Dict[str, float]] = []
    cursor = start_ms
    while cursor <= end_ms:
        rows = _get({"symbol": symbol, "interval": "1m", "startTime": cursor,
                     "endTime": end_ms, "limit": PAGE}, get)
        if not rows:
            break
        out.extend(parse_kline(r) for r in rows)
        nxt = int(rows[-1][0]) + MINUTE_MS
        if nxt <= cursor:                      # defensive: never loop on a stuck page
            break
        cursor = nxt
        if len(rows) < PAGE:
            break
    return out


def fill_gaps(bars: List[Dict[str, float]], start_s: int, end_s: int) -> List[Dict[str, float]]:
    """Return one bar per minute from start_s to end_s inclusive.

    A minute with no trades is absent from Binance's response; the feature
    engine expects a continuous series, so the gap is filled with a flat,
    zero-volume bar at the previous close. Minutes before the first real bar
    are not invented: the series starts at the first bar Binance returned.
    """
    by_ts = {b["ts"]: b for b in bars}
    filled: List[Dict[str, float]] = []
    prev: Optional[Dict[str, float]] = None
    for ts in range(start_s, end_s + 1, 60):
        b = by_ts.get(ts)
        if b is None:
            if prev is None:
                continue
            c = prev["close"]
            b = {"ts": ts, "open": c, "high": c, "low": c, "close": c, "volume": 0.0,
                 "quote_volume": 0.0, "trades": 0, "taker_base": 0.0, "taker_quote": 0.0}
        filled.append(b)
        prev = b
    return filled
