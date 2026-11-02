#!/usr/bin/env python3
"""Check the real-data connections, pre-download replays, and run the agent on them.

    python fetch_real_data.py check      # can we reach Binance and CoinGecko right now?
    python fetch_real_data.py replay     # download + cache the FDUSD April 2025 replay
    python fetch_real_data.py run        # play the replay through the agent, print its timeline

Run ``check`` before a demo. ``replay`` saves the data under data_cache/, after
which the dashboard replays it with no network at all. ``run`` reports when the
agent escalated compared with when the price actually broke.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings

from sthirasense.sources import REAL_SOURCES, SourceError, binance, coingecko
from sthirasense.sources.feeds import CACHE_DIR, ReplayFeed, iso_to_ts, utc_label

REPLAYS = [k for k, v in REAL_SOURCES.items() if v.kind == "replay"]
BREACH = 0.005
CALM_S = 2 * 3600


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


def run(key: str) -> int:
    """Play the cached replay through the real agent and print what it did, when."""
    warnings.filterwarnings("ignore")
    from sthirasense.agent.agent import DepegAgent
    from sthirasense.models import DepegEnsemble

    try:
        feed = ReplayFeed(REAL_SOURCES[key])
    except SourceError as e:
        print(f"FAIL  {e}\n(run `python fetch_real_data.py replay` first)")
        return 1
    agent = DepegAgent(DepegEnsemble(), feed)
    start = iso_to_ts(REAL_SOURCES[key].start)
    first_level, first_break, low, peak_p, peak_stress = {}, None, None, 0.0, 0.0
    hold_since, calm_end, warn_after_calm = None, None, None   # calm = 2h+ straight at HOLD
    while (t := agent.tick()) is not None:
        if t.ts < start or not (t.scored and t.prediction and t.decision):
            if t.ts >= start and abs(t.bar["close"] - 1) >= BREACH and first_break is None:
                first_break = (int(t.ts), t.bar["close"])
            continue
        ts, close = int(t.ts), t.bar["close"]
        act = t.decision.action.label
        first_level.setdefault(act, (ts, close, t.prediction.depeg_proba, t.prediction.stress))
        if first_break is None:
            if act == "HOLD":
                hold_since = hold_since or ts
                if ts - hold_since >= CALM_S:
                    calm_end, warn_after_calm = ts, None
            else:
                hold_since = None
                if act in ("WARN", "CRITICAL") and calm_end is not None and warn_after_calm is None:
                    warn_after_calm = ts
        if abs(close - 1) >= BREACH and first_break is None:
            first_break = (ts, close)
        if low is None or t.bar["low"] < low[1]:
            low = (ts, t.bar["low"])
        peak_p = max(peak_p, t.prediction.depeg_proba)
        peak_stress = max(peak_stress, t.prediction.stress)

    s = agent.status()
    print(f"Agent on {REAL_SOURCES[key].name}  (pair {feed.symbol}, supply: {feed.supply.source})\n")
    print(f"  replay minutes            {len(feed.bars) - feed.n_warm}")
    print(f"  peak p(depeg, 24h)        {peak_p:.3e}   (threshold 1.8e-3)")
    print(f"  peak liquidity stress     {peak_stress:.1f}")
    for lvl in ("WATCH", "WARN", "CRITICAL"):
        if lvl in first_level:
            ts, c, p, st = first_level[lvl]
            print(f"  first {lvl:<9}           {utc_label(ts)}   price {c:.5f}   p {p:.2e}   stress {st:.1f}")
        else:
            print(f"  first {lvl:<9}           never")
    if first_break:
        print(f"  price first 0.5% off $1   {utc_label(first_break[0])}   price {first_break[1]:.5f}")
        # Lead time only counts a warning that followed 2h+ of calm at HOLD;
        # an agent that was elevated from the start has not anticipated anything.
        if warn_after_calm is not None:
            lead = (first_break[0] - warn_after_calm) / 60
            print(f"  warning lead time         {lead:+.0f} min  (fresh WARN after 2h+ calm, before the break)")
        else:
            print("  warning lead time         none: the agent never spent 2h calm at HOLD before the")
            print("                            break, so no fresh warning preceded it")
        crit = first_level.get("CRITICAL")
        if crit:
            print(f"  CRITICAL vs the break     {(crit[0] - first_break[0]) / 60:+.0f} min")
    else:
        print("  price first 0.5% off $1   never")
    if low:
        print(f"  lowest price              {utc_label(low[0])}   {low[1]:.5f}")
    alerts = agent.state.alerts
    print(f"  alerts                    {len(alerts)}  " + ", ".join(
        f"#{a.id} {a.level} {a.outcome or 'open'}" for a in alerts))
    print(f"  model runs saved          {s['sensing_efficiency'] * 100:.0f}%")
    print(f"  cost weights now          miss {agent.cost.miss_base:,.0f}, false alarm {agent.cost.false_base:.1f}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="test Binance and CoinGecko connectivity")
    r = sub.add_parser("replay", help="download and cache a historical replay")
    r.add_argument("key", nargs="?", default=REPLAYS[0], choices=REPLAYS)
    r.add_argument("--refresh", action="store_true", help="ignore the cache and download again")
    g = sub.add_parser("run", help="run the agent over a cached replay and print its timeline")
    g.add_argument("key", nargs="?", default=REPLAYS[0], choices=REPLAYS)
    args = ap.parse_args()
    if args.cmd == "check":
        sys.exit(check())
    if args.cmd == "replay":
        sys.exit(replay(args.key, args.refresh))
    sys.exit(run(args.key))


if __name__ == "__main__":
    main()
