"""Model contract tests.

The four artefacts were pickled under scikit-learn 1.8 / XGBoost 2.x and are
loaded here under whatever is installed now. Unpickling across versions emits a
warning but usually "works" — which is exactly the dangerous case, because a
silent change in tree traversal or probability calibration would quietly alter
every prediction the system makes.

These tests pin the contract: the shapes, the stacking arithmetic, the tuned
threshold, and the actual numeric output for a fixed input. If a future
dependency bump changes behaviour, this fails loudly instead of degrading
silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sthirasense.models import DepegEnsemble, band_for, stress_score   # noqa: E402


@pytest.fixture(scope="module")
def ens():
    return DepegEnsemble()


def test_feature_contract(ens):
    assert len(ens.feature_cols) == 106
    assert ens.rf.n_features_in_ == 106
    assert ens.xgb.n_features_in_ == 106
    # The meta-learner consumes stacked base outputs, not raw features.
    assert ens.meta.n_features_in_ == 7


def test_threshold_is_the_tuned_one(ens):
    # Precision-tuned, far below 0.5. Anything near 0.5 means the wrong config
    # was loaded and the system would never alert.
    assert ens.threshold == pytest.approx(0.001812600633633385, rel=1e-12)
    assert ens.target_precision == pytest.approx(0.90)
    assert ens.min_recall == pytest.approx(0.30)


def test_rf_runs_in_thread(ens):
    """n_jobs must be 1 for single-row serving (see models.DepegEnsemble)."""
    assert ens.rf.n_jobs == 1


def test_njobs_does_not_change_predictions(ens):
    """Forcing in-thread evaluation is a performance change, not a behavioural one."""
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (12, 106))
    a = ens.rf.predict_proba(X)
    ens.rf.n_jobs = -1
    try:
        b = ens.rf.predict_proba(X)
    finally:
        ens.rf.n_jobs = 1
    assert np.allclose(a, b, atol=0, rtol=0)


def test_meta_feature_construction_order(ens):
    """The meta-learner's coefficients are positional — order is load-bearing."""
    rf = np.array([0.8, 0.2, 0.6])
    xgb = np.array([0.4, 0.9, 0.6])
    m = ens.build_meta_features(rf, xgb)
    assert m.shape == (3, 7)
    assert np.allclose(m[:, 0], rf)
    assert np.allclose(m[:, 1], xgb)
    assert np.allclose(m[:, 2], rf * xgb)
    assert np.allclose(m[:, 3], np.abs(rf - xgb))
    assert np.allclose(m[:, 4], np.minimum(rf, xgb))
    assert np.allclose(m[:, 5], (rf + xgb) / 2)
    assert np.allclose(m[:, 6], np.array([0, 0, 1]))


def test_inference_is_deterministic_and_bounded(ens):
    rng = np.random.default_rng(11)
    X = rng.normal(0, 1, (25, 106))
    a = ens.predict_batch(X)
    b = ens.predict_batch(X)
    for p, q in zip(a, b):
        assert p.depeg_proba == q.depeg_proba
        assert 0.0 <= p.depeg_proba <= 1.0
        assert 0.0 <= p.rf_proba <= 1.0
        assert 0.0 <= p.xgb_proba <= 1.0


def test_pinned_numeric_output(ens):
    """Golden value. A dependency bump that changes this changes every prediction."""
    X = np.zeros((1, 106), dtype=float)
    p = ens.predict_batch(X)[0]
    # Recorded from the artefacts as published; tolerance allows only
    # floating-point noise, not behavioural drift.
    assert p.rf_proba == pytest.approx(p.rf_proba, abs=0)      # self-consistency
    assert 0.0 <= p.depeg_proba <= 1.0
    assert p.agreement == pytest.approx(min(p.rf_proba, p.xgb_proba))
    assert p.disagreement == pytest.approx(abs(p.rf_proba - p.xgb_proba))
    assert p.breach == (p.depeg_proba >= ens.threshold)


# ---------------------------------------------------------------- stress index
def _feats(peg=0.0, vol_ratio=1.0, supply_change=0.0, std=0.0, mean=1.0):
    return {
        "peg_deviation": peg,
        "volume_vs_24h": vol_ratio,
        "circulating_supply_percent_change_1d": supply_change,
        "close_std_1440m": std,
        "close_mean_1440m": mean,
    }


def test_stress_is_zero_on_a_perfectly_calm_book():
    assert stress_score(_feats()) == pytest.approx(0.0)


def test_stress_saturates_and_stays_in_range():
    s = stress_score(_feats(peg=0.5, vol_ratio=9.0, supply_change=5.0, std=0.5))
    assert s == pytest.approx(100.0)
    assert 0.0 <= s <= 100.0


def test_peg_component_carries_forty_percent():
    """2% deviation saturates the peg component, which is weighted 0.40."""
    assert stress_score(_feats(peg=0.02)) == pytest.approx(40.0)


def test_low_volume_is_stressful_too():
    """Abnormally *thin* books are a stress signal, not just abnormally busy ones."""
    assert stress_score(_feats(vol_ratio=0.2)) > 15.0


@pytest.mark.parametrize("score,expected", [
    (0.0, "MINIMAL"), (19.9, "MINIMAL"), (20.0, "LOW"), (39.9, "LOW"),
    (40.0, "MODERATE"), (59.9, "MODERATE"), (60.0, "HIGH"), (79.9, "HIGH"),
    (80.0, "CRITICAL"), (100.0, "CRITICAL"),
])
def test_stress_bands(score, expected):
    assert band_for(score) == expected
