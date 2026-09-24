"""Live feature engine.

Reproduces, bar by bar, the exact feature set the trained models were fitted on.

The original research pipeline (``ml/preprocessing.py`` + the extra columns added
inside ``ml/train.py``) was a batch pandas job: it took a whole CSV and computed
rolling windows across it. For a live system we need the same numbers computed
incrementally as each new minute arrives.

This module is deliberately a faithful re-implementation rather than a reuse of
the batch code: every formula below is copied from the training pipeline, and
``tests/test_feature_parity.py`` asserts that this engine and the original pandas
code agree to within floating-point tolerance on the same input. If that test
fails, the models are being fed something they were never trained on and every
downstream probability is meaningless.

Raw bar schema (the only thing a data source — live or simulated — must supply):

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
from collections import deque
from typing import Deque, Dict, List, Optional

from .config import PEG, WARMUP_BARS, WINDOWS

EPS = 1e-8

RAW_FIELDS = (
    "open", "high", "low", "close", "volume", "quote_volume",
    "trades", "taker_base", "taker_quote", "market_cap", "circulating_supply",
)


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs) -> float:
    """Sample standard deviation — matches pandas .rolling().std() (ddof=1)."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(max(var, 0.0))


class FeatureEngine:
    """Maintains a rolling buffer of raw bars and emits the 106-feature vector.

    Usage::

        eng = FeatureEngine(feature_cols)
        for bar in stream:
            vec = eng.push(bar)      # None until warm
            if vec is not None:
                proba = model.score(vec)
    """

    def __init__(self, feature_cols: List[str], maxlen: int = WARMUP_BARS + 16):
        self.feature_cols = list(feature_cols)
        self.maxlen = maxlen
        self.bars: Deque[Dict[str, float]] = deque(maxlen=maxlen)
        # Derived per-bar series we need rolling statistics over.
        self.trade_size: Deque[float] = deque(maxlen=maxlen)
        self.peg_dev: Deque[float] = deque(maxlen=maxlen)

    # ------------------------------------------------------------------ state
    @property
    def n_bars(self) -> int:
        return len(self.bars)

    @property
    def warm(self) -> bool:
        """True once a full 24h window exists, so no feature is a partial window."""
        return len(self.bars) >= WARMUP_BARS

    def warmup_progress(self) -> float:
        return min(1.0, len(self.bars) / WARMUP_BARS)

    # ------------------------------------------------------------------- push
    def push(self, bar: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Append a raw bar. Returns the full feature dict once warm, else None."""
        b = {k: float(bar[k]) for k in RAW_FIELDS}
        b["ts"] = int(bar["ts"])
        self.bars.append(b)
        self.trade_size.append(b["volume"] / (b["trades"] + 1))
        self.peg_dev.append(abs(b["close"] - PEG))
        if not self.warm:
            return None
        return self._compute()

    # --------------------------------------------------------------- internals
    def _series(self, key: str, n: int) -> List[float]:
        """Last n values of a raw field (oldest .. newest)."""
        bars = self.bars
        start = len(bars) - n
        return [bars[i][key] for i in range(start, len(bars))]

    def _tail(self, dq: Deque[float], n: int) -> List[float]:
        start = len(dq) - n
        return [dq[i] for i in range(start, len(dq))]

    def _pct_change(self, key: str, periods: int) -> float:
        """Percent change over `periods` bars, expressed in percent (x100)."""
        bars = self.bars
        if len(bars) <= periods:
            return 0.0
        prev = bars[-1 - periods][key]
        cur = bars[-1][key]
        if prev == 0:
            return 0.0
        return (cur / prev - 1.0) * 100.0

    def _compute(self) -> Dict[str, float]:
        bars = self.bars
        cur = bars[-1]
        f: Dict[str, float] = {}

        # ---- raw passthrough -------------------------------------------------
        for k in RAW_FIELDS:
            f[k] = cur[k]
        # `price` and `close` are the same quantity in the source dataset.
        f["price"] = cur["close"]

        # ---- CoinGecko-style change columns ---------------------------------
        # In the training CSV these arrived pre-computed from the market-data
        # provider. Here they are derived from our own bar history, which is
        # self-consistent and avoids depending on a second data source.
        f["percent_change_1h"] = self._pct_change("close", 60)
        f["percent_change_24h"] = self._pct_change("close", 1440)
        f["volume_percent_change_24h"] = self._pct_change("volume", 1440)
        f["market_cap_percent_change_1d"] = self._pct_change("market_cap", 1440)
        f["circulating_supply_percent_change_1d"] = self._pct_change("circulating_supply", 1440)

        # ---- basic derived (preprocessing._create_basic_features) -----------
        o, h, l, c = cur["open"], cur["high"], cur["low"], cur["close"]
        hl = h - l
        f["daily_range"] = (h - l) / (o + EPS)
        f["body_size"] = abs(c - o) / (o + EPS)
        f["upper_shadow"] = (h - max(o, c)) / (hl + EPS)
        f["lower_shadow"] = (min(o, c) - l) / (hl + EPS)
        f["close_position"] = (c - l) / (hl + EPS)
        f["trade_size"] = cur["volume"] / (cur["trades"] + 1)
        f["taker_buy_ratio"] = cur["taker_base"] / (cur["volume"] + EPS)
        f["taker_sell_ratio"] = cur["taker_quote"] / (cur["quote_volume"] + EPS)
        f["volume_imbalance"] = (cur["taker_base"] - cur["taker_quote"]) / (cur["volume"] + EPS)
        f["mcap_to_supply_ratio"] = cur["market_cap"] / (cur["circulating_supply"] + EPS)
        f["price_to_mcap_billion"] = f["price"] / (cur["market_cap"] / 1e9 + EPS)

        # ---- rolling windows (preprocessing._create_rolling_features) -------
        for w in WINDOWS:
            close_w = self._series("close", w)
            vol_w = self._series("volume", w)
            trd_w = self._series("trades", w)
            ts_w = self._tail(self.trade_size, w)
            mc_w = self._series("market_cap", w)
            cs_w = self._series("circulating_supply", w)

            f[f"close_mean_{w}m"] = _mean(close_w)
            f[f"close_std_{w}m"] = _std(close_w)
            f[f"close_min_{w}m"] = min(close_w)
            f[f"close_max_{w}m"] = max(close_w)

            f[f"volume_mean_{w}m"] = _mean(vol_w)
            f[f"volume_std_{w}m"] = _std(vol_w)
            f[f"volume_sum_{w}m"] = sum(vol_w)

            f[f"trades_mean_{w}m"] = _mean(trd_w)
            f[f"trade_size_mean_{w}m"] = _mean(ts_w)

            f[f"market_cap_mean_{w}m"] = _mean(mc_w)
            f[f"market_cap_std_{w}m"] = _std(mc_w)
            f[f"circulating_supply_mean_{w}m"] = _mean(cs_w)

        # ---- peg features (preprocessing._create_peg_features) --------------
        dev = abs(c - PEG)
        f["peg_deviation"] = dev
        f["peg_deviation_pct"] = dev * 100.0
        f["peg_direction"] = c - PEG
        f["above_peg"] = 1.0 if c > PEG else 0.0
        f["below_peg"] = 1.0 if c < PEG else 0.0

        for w in WINDOWS:
            pw = self._tail(self.peg_dev, w)
            f[f"peg_deviation_mean_{w}m"] = _mean(pw)
            f[f"peg_deviation_max_{w}m"] = max(pw)
            f[f"peg_deviation_std_{w}m"] = _std(pw)

        dev180 = self._tail(self.peg_dev, 180)
        dev1440 = self._tail(self.peg_dev, 1440)
        f["peg_stress_1pct_3h"] = float(sum(1 for x in dev180 if x > 0.01))
        f["peg_stress_1pct_24h"] = float(sum(1 for x in dev1440 if x > 0.01))
        f["peg_stress_2pct_3h"] = float(sum(1 for x in dev180 if x > 0.02))
        f["peg_stress_2pct_24h"] = float(sum(1 for x in dev1440 if x > 0.02))

        # ---- interactions (preprocessing._create_interaction_features) ------
        f["volume_vs_3h"] = cur["volume"] / (f["volume_mean_180m"] + EPS)
        f["volume_vs_24h"] = cur["volume"] / (f["volume_mean_1440m"] + EPS)
        f["trade_size_vs_24h"] = f["trade_size"] / (f["trade_size_mean_1440m"] + EPS)
        f["mcap_to_volume_24h"] = cur["market_cap"] / (f["volume_sum_1440m"] + EPS)

        # ---- momentum (preprocessing._create_momentum_features) -------------
        f["price_accel_1h_24h"] = f["percent_change_1h"] - f["percent_change_24h"]

        # ---- market dynamics (preprocessing._create_market_dynamics) --------
        f["volume_spike_peg_stress"] = 1.0 if (f["volume_vs_24h"] > 2.0 and dev > 0.01) else 0.0
        f["extreme_volume_spike"] = 1.0 if f["volume_vs_24h"] > 3.0 else 0.0
        f["large_trade_anomaly"] = 1.0 if f["trade_size_vs_24h"] > 2.0 else 0.0

        # ---- extras added inside train.py.prepare_data ----------------------
        f["turnover_ratio"] = f["volume_mean_1440m"] / (f["market_cap_mean_1440m"] + 1e-9)
        f["price_momentum"] = self._pct_change("close", 60) / 100.0  # pct_change(60), not x100

        # peg_deviation.diff(30)
        pd30 = self.peg_dev[-31] if len(self.peg_dev) > 30 else self.peg_dev[0]
        f["peg_deviation_change"] = dev - pd30
        # peg_deviation.rolling(60).std()
        f["peg_deviation_volatility"] = _std(self._tail(self.peg_dev, 60))

        return f

    # ----------------------------------------------------------------- vector
    def vector(self, feats: Dict[str, float]) -> List[float]:
        """Order the feature dict into the exact column order the models expect."""
        out: List[float] = []
        for name in self.feature_cols:
            v = feats.get(name)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                v = 0.0
            out.append(float(v))
        return out

    def missing_features(self, feats: Dict[str, float]) -> List[str]:
        """Any expected column this engine failed to produce. Should always be empty."""
        return [n for n in self.feature_cols if n not in feats]
