#!/usr/bin/env python3
"""Check the real-data connections and pre-download replays.

    python fetch_real_data.py check      # can we reach Binance and CoinGecko right now?
    python fetch_real_data.py replay     # download + cache the USDC/SVB March 2023 replay

Run ``check`` before a demo. ``replay`` saves the data under data_cache/, after
which the dashboard replays it with no network at all.
"""
from __future__ import annotations

import argparse
import sys
import time

from sthirasense.sources import REAL_SOURCES, SourceError, binance, coingecko
from sthirasense.sources.feeds import CACHE_DIR, ReplayFeed, utc_label


def check() -> int:
    ok = True
    now = int(time.time()) // 60 * 60
    print("[1] Binance 1-minute klines (USDCUSDT)")
    try:
        bars = binance.fetch_klines("USDCUSDT", (now - 600) * 1000, (now - 60) * 1000)
        b = bars[-1]
        print(f"    OK  {len(bars)} bars, latest {utc_label(int(b['ts']))}: close {b['close']:.5f}, "
              f"volume {b['volume']:,.0f}, trades {b['trades']}")
    except (SourceError, IndexError) as e:
        ok = False
        print(f"    FAIL  {e}")

    print("[2] CoinGecko circulating supply (usd-coin)")
    try:
        supply, mcap = coingecko.current_supply("usd-coin")
        print(f"    OK  supply {supply / 1e9:,.2f}B, market cap ${mcap / 1e9:,.2f}B")
    except SourceError as e:
        ok = False
        print(f"    FAIL  {e}")
        print("    tip: get a free Demo key at coingecko.com/en/api and set COINGECKO_API_KEY")

    print("\nBoth sources reachable: live mode will work." if ok else "\nFix the failure above before using live mode.")
    return 0 if ok else 1


def replay(key: str, refresh: bool) -> int:
    src = REAL_SOURCES[key]
    print(f"Downloading {src.name} ...")
    try:
        feed = ReplayFeed(src, use_cache=not refresh)
    except SourceError as e:
        print(f"FAIL  {e}")
        return 1
    scen = feed.bars[feed.n_warm:]
    low = min(scen, key=lambda b: b["low"])
    print(f"OK  pair {feed.symbol}, {feed.n_warm} warm-up + {len(scen)} replay minutes")
    print(f"    lowest price {low['low']:.4f} at {utc_label(int(low['ts']))}")
    print(f"    supply source: {feed.supply.source}")
    print(f"    cached in {CACHE_DIR / (key + '.json')}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="test Binance and CoinGecko connectivity")
    r = sub.add_parser("replay", help="download and cache a historical replay")
    r.add_argument("key", nargs="?", default="real_usdc_svb_2023", choices=[
        k for k, v in REAL_SOURCES.items() if v.kind == "replay"])
    r.add_argument("--refresh", action="store_true", help="ignore the cache and download again")
    args = ap.parse_args()
    sys.exit(check() if args.cmd == "check" else replay(args.key, args.refresh))


if __name__ == "__main__":
    main()
