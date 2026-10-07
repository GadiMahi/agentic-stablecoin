"""Feature parity: live engine vs the original batch pandas pipeline.

The trained models were fitted on features produced by ``ml/preprocessing.py``
plus a handful of extra columns created inside ``ml/train.py``. The live
``FeatureEngine`` recomputes those incrementally. If the two ever disagree, the
models are being scored on inputs they never saw during training and every
probability downstream is meaningless.

This test builds a synthetic minute series, runs it through both paths, and
asserts agreement on all 106 columns the models actually consume.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from sthirasense.config import WINDOWS                      # noqa: E402
from sthirasense.features import FeatureEngine              # noqa: E402
from sthirasense.models import DepegEnsemble                # noqa: E402


N_BARS = 1700          # > 1440 warmup so the final rows are fully warm
EPS_REL = 1e-6


def make_bars(n: int = N_BARS, seed: int = 7) -> pd.DataFrame:
    """A plausible stablecoin minute series: tight around $1 with volume noise."""
    rng = np.random.default_rng(seed)
    close = 1.0 + np.cumsum(rng.normal(0, 2e-5, n))
    close = np.clip(close, 0.97, 1.03)
    spread = np.abs(rng.normal(0, 3e-5, n))
    openp = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(openp, close) + spread
    low = np.minimum(openp, close) - spread
    volume = np.abs(rng.lognormal(11.0, 0.6, n))
    trades = np.maximum(1, (volume / 900 + rng.normal(0, 12, n)).astype(int))
    taker_base = volume * rng.uniform(0.3, 0.7, n)
    quote_volume = volume * close
    taker_quote = quote_volume * rng.uniform(0.3, 0.7, n)
    supply = 7.19e10 + np.cumsum(rng.normal(0, 4e5, n))
    mcap = supply * close

    return pd.DataFrame({
        "ts": np.arange(n, dtype=np.int64) * 60,
        "open": openp, "high": high, "low": low, "close": close,
        "volume": volume, "quote_volume": quote_volume, "trades": trades,
        "taker_base": taker_base, "taker_quote": taker_quote,
        "market_cap": mcap, "circulating_supply": supply,
    })


def batch_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Reference path: the original preprocessing code, restricted to the
    windows the trained models actually use, plus train.py's extra columns."""
    import preprocessing  # from ml/

    d = df.copy()
    # Provider-supplied change columns, derived here exactly as the live engine does.
    d["price"] = d["close"]
    d["percent_change_1h"] = d["close"].pct_change(60) * 100
    d["percent_change_24h"] = d["close"].pct_change(1440) * 100
    d["volume_percent_change_24h"] = d["volume"].pct_change(1440) * 100
    d["market_cap_percent_change_1d"] = d["market_cap"].pct_change(1440) * 100
    d["circulating_supply_percent_change_1d"] = d["circulating_supply"].pct_change(1440) * 100

    eng = preprocessing.StablecoinFeatureEngine()
    eng.windows = list(WINDOWS)          # drop 7d/30d, as training did

    d = eng._create_basic_features(d)
    d = eng._create_rolling_features(d)
    d = eng._create_peg_features(d)

    # The interaction/momentum steps reference long-window columns whose derived
    # features were dropped before training. Inject placeholders so the original
    # code runs unmodified; nothing we assert on depends on them.
    for col in ("volume_mean_10080m", "volume_mean_43200m",
                "trade_size_mean_10080m", "volume_sum_10080m"):
        d[col] = 1.0
    for col in ("percent_change_7d", "percent_change_30d"):
        d[col] = 0.0

    d = eng._create_interaction_features(d)
    d = eng._create_momentum_features(d)
    d = eng._create_market_dynamics(d)

    # Extras added inside train.py.prepare_data
    d["turnover_ratio"] = d["volume_mean_1440m"] / (d["market_cap_mean_1440m"] + 1e-9)
    d["price_momentum"] = d["close"].pct_change(periods=60).fillna(0)
    d["peg_deviation_change"] = d["peg_deviation"].diff(periods=30).fillna(0)
    d["peg_deviation_volatility"] = d["peg_deviation"].rolling(window=60).std().fillna(0)
    return d


@pytest.fixture(scope="module")
def artefacts():
    bars = make_bars()
    ref = batch_pipeline(bars)
    cols = DepegEnsemble().feature_cols

    eng = FeatureEngine(cols)
    live_rows, live_idx = [], []
    for i, rec in enumerate(bars.to_dict("records")):
        out = eng.push(rec)
        if out is not None:
            live_rows.append(out)
            live_idx.append(i)
    return bars, ref, cols, live_rows, live_idx


def test_engine_produces_every_expected_column(artefacts):
    _, _, cols, live_rows, _ = artefacts
    assert live_rows, "engine never warmed up"
    missing = [c for c in cols if c not in live_rows[-1]]
    assert missing == [], f"live engine missing {len(missing)} columns: {missing[:10]}"


def test_warmup_boundary_is_exactly_1440_bars(artefacts):
    _, _, cols, _, live_idx = artefacts
    # First warm bar is index 1439 (the 1440th bar).
    assert live_idx[0] == 1439


@pytest.mark.parametrize("offset", [0, 25, 120, -1])
def test_live_matches_batch_on_all_106_features(artefacts, offset):
    _, ref, cols, live_rows, live_idx = artefacts
    k = offset if offset >= 0 else len(live_rows) - 1
    live = live_rows[k]
    row = ref.iloc[live_idx[k]]

    bad = []
    for c in cols:
        a, b = float(live[c]), float(row[c])
        if np.isnan(b):
            continue
        scale = max(abs(a), abs(b), 1e-9)
        if abs(a - b) / scale > EPS_REL:
            bad.append((c, a, b))
    assert not bad, "mismatched features:\n" + "\n".join(
        f"  {c}: live={a!r} batch={b!r}" for c, a, b in bad[:15]
    )


def test_vector_order_matches_model_contract(artefacts):
    _, _, cols, live_rows, _ = artefacts
    eng = FeatureEngine(cols)
    vec = eng.vector(live_rows[-1])
    assert len(vec) == len(cols) == 106
    assert all(float(v) == v for v in vec)
    assert all(np.isfinite(v) for v in vec)
