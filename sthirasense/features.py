"""Live feature engine.

Reproduces, bar by bar, the exact feature set the trained models were fitted on.

The original research pipeline (``ml/preprocessing.py`` plus extra columns added
inside ``ml/train.py``) was a batch pandas job: it took a whole CSV and computed
rolling windows across it. For a live system we need the same numbers computed
incrementally as each new minute arrives.

This is a faithful re-implementation rather than a reuse of the batch code, and
``tests/test_feature_parity.py`` asserts the two agree to floating-point
tolerance on identical input. If that test fails, the models are being fed
something they were never trained on and every downstream probability is
meaningless.

Implementation note: series are held in pre-allocated numpy buffers with a write
head, so a rolling window is a *view* (``buf[n-w:n]``) and every statistic runs
in C. A pure-Python version of this loop was ~40x slower and could not keep up
with a replayed market at useful speed.

Raw bar schema — the only thing a data source (live or simulated) must supply:

    ts                  int    epoch seconds
    open high low close float  price
    volume              float  base asset volume
    quote_volume        float  quote asset volume
    trades              int    number of trades
    taker_base          float  taker buy base volume
    taker_quote         float  taker buy quote volume
    market_cap          float
    circulating_supply  float
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .config import PEG, WARMUP_BARS, WINDOWS

EPS = 1e-8

RAW_FIELDS = (
    "open", "high", "low", "close", "volume", "quote_volume",
    "trades", "taker_base", "taker_quote", "market_cap", "circulating_supply",
)

# Series we keep rolling buffers for: the raw fields plus two derived series.
_SERIES = RAW_FIELDS + ("trade_size", "peg_deviation")


class _Buffer:
    """Append-only numpy buffer with periodic compaction.

    Capacity is 2x the needed history; when the head reaches the end we copy the
    live tail back to the front. Amortised O(1) per append, and every window
    read is a contiguous view with no copy.
    """

    __slots__ = ("keep", "arr", "n")

    def __init__(self, keep: int):
        self.keep = keep
        self.arr = np.zeros(keep * 2, dtype=np.float64)
        self.n = 0

    def append(self, v: float) -> None:
        if self.n == self.arr.size:
            self.arr[: self.keep] = self.arr[self.n - self.keep:]
            self.n = self.keep
        self.arr[self.n] = v
        self.n += 1

    def tail(self, w: int) -> np.ndarray:
        return self.arr[self.n - w: self.n]

    def at(self, back: int) -> float:
        """Value `back` bars before the newest (0 = newest)."""
        return float(self.arr[self.n - 1 - back])

    @property
    def last(self) -> float:
        return float(self.arr[self.n - 1])


class FeatureEngine:
    """Maintains rolling buffers of raw bars and emits the 106-feature vector.

    Usage::

        eng = FeatureEngine(ensemble.feature_cols)
        for bar in stream:
            feats = eng.push(bar)       # None until warm
            if feats is not None:
                pred = ensemble.predict(eng.vector(feats), feats)
    """

    def __init__(self, feature_cols: List[str], keep: int = WARMUP_BARS + 8):
        self.feature_cols = list(feature_cols)
        self.keep = keep
        self._buf: Dict[str, _Buffer] = {k: _Buffer(keep) for k in _SERIES}
        self._count = 0
        self._last_ts = 0

    # ------------------------------------------------------------------ state
    @property
    def n_bars(self) -> int:
        return self._count

    @property
    def warm(self) -> bool:
        """True once a full 24h window exists, so no feature is a partial window."""
        return self._count >= WARMUP_BARS

    def warmup_progress(self) -> float:
        return min(1.0, self._count / WARMUP_BARS)

    # ------------------------------------------------------------------- push
    def push(self, bar: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Append a raw bar. Returns the full feature dict once warm, else None."""
        b = self._buf
        for k in RAW_FIELDS:
            b[k].append(float(bar[k]))
        b["trade_size"].append(float(bar["volume"]) / (float(bar["trades"]) + 1.0))
        b["peg_deviation"].append(abs(float(bar["close"]) - PEG))
        self._count += 1
        self._last_ts = int(bar["ts"])
        if not self.warm:
            return None
        return self._compute()

    # --------------------------------------------------------------- internals
    def _pct_change(self, key: str, periods: int) -> float:
        """Percent change over `periods` bars, in percent (x100)."""
        if self._count <= periods:
            return 0.0
        prev = self._buf[key].at(periods)
        if prev == 0.0:
            return 0.0
        return (self._buf[key].last / prev - 1.0) * 100.0

    def _compute(self) -> Dict[str, float]:
        b = self._buf
        f: Dict[str, float] = {}

        # ---- raw passthrough -------------------------------------------------
        for k in RAW_FIELDS:
            f[k] = b[k].last
        # `price` and `close` are the same quantity in the source dataset.
        f["price"] = f["close"]

        # ---- provider-style change columns -----------------------------------
        # In the training CSV these arrived pre-computed from the market-data
        # provider. Here they are derived from our own bar history, which is
        # self-consistent and avoids depending on a second data source.
        f["percent_change_1h"] = self._pct_change("close", 60)
        f["percent_change_24h"] = self._pct_change("close", 1440)
        f["volume_percent_change_24h"] = self._pct_change("volume", 1440)
        f["market_cap_percent_change_1d"] = self._pct_change("market_cap", 1440)
        f["circulating_supply_percent_change_1d"] = self._pct_change("circulating_supply", 1440)

        # ---- basic derived (preprocessing._create_basic_features) ------------
        o, h, l, c = f["open"], f["high"], f["low"], f["close"]
        hl = h - l
        f["daily_range"] = (h - l) / (o + EPS)
        f["body_size"] = abs(c - o) / (o + EPS)
        f["upper_shadow"] = (h - max(o, c)) / (hl + EPS)
        f["lower_shadow"] = (min(o, c) - l) / (hl + EPS)
        f["close_position"] = (c - l) / (hl + EPS)
        f["trade_size"] = b["trade_size"].last
        f["taker_buy_ratio"] = f["taker_base"] / (f["volume"] + EPS)
        f["taker_sell_ratio"] = f["taker_quote"] / (f["quote_volume"] + EPS)
        f["volume_imbalance"] = (f["taker_base"] - f["taker_quote"]) / (f["volume"] + EPS)
        f["mcap_to_supply_ratio"] = f["market_cap"] / (f["circulating_supply"] + EPS)
        f["price_to_mcap_billion"] = f["price"] / (f["market_cap"] / 1e9 + EPS)

        # ---- rolling windows (preprocessing._create_rolling_features) --------
        for w in WINDOWS:
            close_w = b["close"].tail(w)
            vol_w = b["volume"].tail(w)
            trd_w = b["trades"].tail(w)
            tsz_w = b["trade_size"].tail(w)
            mc_w = b["market_cap"].tail(w)
            cs_w = b["circulating_supply"].tail(w)

            f[f"close_mean_{w}m"] = float(close_w.mean())
            f[f"close_std_{w}m"] = float(close_w.std(ddof=1))   # pandas default
            f[f"close_min_{w}m"] = float(close_w.min())
            f[f"close_max_{w}m"] = float(close_w.max())

            f[f"volume_mean_{w}m"] = float(vol_w.mean())
            f[f"volume_std_{w}m"] = float(vol_w.std(ddof=1))
            f[f"volume_sum_{w}m"] = float(vol_w.sum())

            f[f"trades_mean_{w}m"] = float(trd_w.mean())
            f[f"trade_size_mean_{w}m"] = float(tsz_w.mean())

            f[f"market_cap_mean_{w}m"] = float(mc_w.mean())
            f[f"market_cap_std_{w}m"] = float(mc_w.std(ddof=1))
            f[f"circulating_supply_mean_{w}m"] = float(cs_w.mean())

        # ---- peg features (preprocessing._create_peg_features) ---------------
        dev = b["peg_deviation"].last
        f["peg_deviation"] = dev
        f["peg_deviation_pct"] = dev * 100.0
        f["peg_direction"] = c - PEG
        f["above_peg"] = 1.0 if c > PEG else 0.0
        f["below_peg"] = 1.0 if c < PEG else 0.0

        for w in WINDOWS:
            pw = b["peg_deviation"].tail(w)
            f[f"peg_deviation_mean_{w}m"] = float(pw.mean())
            f[f"peg_deviation_max_{w}m"] = float(pw.max())
            f[f"peg_deviation_std_{w}m"] = float(pw.std(ddof=1))

        d180 = b["peg_deviation"].tail(180)
        d1440 = b["peg_deviation"].tail(1440)
        f["peg_stress_1pct_3h"] = float((d180 > 0.01).sum())
        f["peg_stress_1pct_24h"] = float((d1440 > 0.01).sum())
        f["peg_stress_2pct_3h"] = float((d180 > 0.02).sum())
        f["peg_stress_2pct_24h"] = float((d1440 > 0.02).sum())

        # ---- interactions (preprocessing._create_interaction_features) -------
        f["volume_vs_3h"] = f["volume"] / (f["volume_mean_180m"] + EPS)
        f["volume_vs_24h"] = f["volume"] / (f["volume_mean_1440m"] + EPS)
        f["trade_size_vs_24h"] = f["trade_size"] / (f["trade_size_mean_1440m"] + EPS)
        f["mcap_to_volume_24h"] = f["market_cap"] / (f["volume_sum_1440m"] + EPS)

        # ---- momentum (preprocessing._create_momentum_features) --------------
        f["price_accel_1h_24h"] = f["percent_change_1h"] - f["percent_change_24h"]

        # ---- market dynamics (preprocessing._create_market_dynamics) ---------
        f["volume_spike_peg_stress"] = 1.0 if (f["volume_vs_24h"] > 2.0 and dev > 0.01) else 0.0
        f["extreme_volume_spike"] = 1.0 if f["volume_vs_24h"] > 3.0 else 0.0
        f["large_trade_anomaly"] = 1.0 if f["trade_size_vs_24h"] > 2.0 else 0.0

        # ---- extras added inside train.py.prepare_data ------------------------
        f["turnover_ratio"] = f["volume_mean_1440m"] / (f["market_cap_mean_1440m"] + 1e-9)
        f["price_momentum"] = self._pct_change("close", 60) / 100.0   # pct_change, not x100

        pd30 = b["peg_deviation"].at(30) if self._count > 30 else b["peg_deviation"].at(self._count - 1)
        f["peg_deviation_change"] = dev - pd30
        f["peg_deviation_volatility"] = float(b["peg_deviation"].tail(60).std(ddof=1))

        return f

    # ----------------------------------------------------------------- vector
    def vector(self, feats: Dict[str, float]) -> np.ndarray:
        """Order the feature dict into the exact column order the models expect."""
        out = np.empty(len(self.feature_cols), dtype=np.float64)
        for i, name in enumerate(self.feature_cols):
            v = feats.get(name, 0.0)
            if v is None or not math.isfinite(v):
                v = 0.0
            out[i] = v
        return out

    def missing_features(self, feats: Dict[str, float]) -> List[str]:
        """Any expected column this engine failed to produce. Should be empty."""
        return [n for n in self.feature_cols if n not in feats]
