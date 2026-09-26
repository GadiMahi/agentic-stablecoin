"""Model layer: the trained stacked ensemble plus the liquidity-stress score.

Two distinct things live here.

``DepegEnsemble`` loads the four artefacts produced by ``ml/train.py`` and runs
the genuine inference path: RandomForest and XGBoost each score the 106-feature
vector, their outputs are assembled into the same 7 meta-features the
meta-learner was fitted on, and the LogisticRegression meta-model emits the final
calibrated probability. Nothing here is approximated — the stacking formula is
transcribed from ``ImprovedHybridDepegPredictor.create_meta_features``.

``stress_score`` is the deterministic 0-100 liquidity-stress index defined in
``ml/liquidity_stress_regression.py``. The trained regressor for it was not
published with the source repository, so we compute the index from its component
formula rather than inventing a model. The weights (peg 40%, volume 25%,
market-cap 20%, volatility 15%) are taken verbatim from that module.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import joblib
import numpy as np

from .config import MODEL_DIR, MODEL_PREFIX

# The artefacts were pickled under scikit-learn 1.8 / XGBoost 2.x. Loading them
# under a newer runtime emits version warnings but the tree structures and
# coefficients deserialize correctly — tests/test_model_contract.py pins the
# expected outputs so any genuine behavioural drift fails loudly instead of
# silently changing predictions.
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


@dataclass
class Prediction:
    """One scored minute."""
    rf_proba: float
    xgb_proba: float
    depeg_proba: float
    threshold: float
    breach: bool          # probability >= tuned threshold
    agreement: float      # min(rf, xgb) — both models must agree to be high
    disagreement: float   # |rf - xgb| — the ensemble's own uncertainty
    stress: float         # 0-100 liquidity stress index
    stress_band: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


STRESS_BANDS = (
    (20.0, "MINIMAL"),
    (40.0, "LOW"),
    (60.0, "MODERATE"),
    (80.0, "HIGH"),
    (101.0, "CRITICAL"),
)


def band_for(score: float) -> str:
    for ceiling, name in STRESS_BANDS:
        if score < ceiling:
            return name
    return "CRITICAL"


def stress_score(f: Dict[str, float]) -> float:
    """Liquidity stress index, 0-100.

    Transcribed from ``calculate_liquidity_stress_score`` in
    ``ml/liquidity_stress_regression.py``. The only adaptation: the source used
    ``rolling(24)`` on minute bars for the volume and volatility baselines, which
    is a 24-*minute* window. On a live minute feed we use the true 24-hour
    rolling statistics the feature engine already maintains, which is what the
    component comments in the original describe.
    """
    # Component 1 — peg deviation (40%). 2% deviation saturates the scale.
    peg_score = float(np.clip(f["peg_deviation"] * 5000.0, 0.0, 100.0))

    # Component 2 — volume stress (25%). Abnormally high *or* abnormally low.
    vr = f["volume_vs_24h"]
    vs_high = np.clip((vr - 1.0) * 33.0, 0.0, 100.0)
    vs_low = np.clip((1.0 - vr) * 143.0, 0.0, 100.0)
    volume_score = float(max(vs_high, vs_low))

    # Component 3 — supply/redemption pressure (20%). 0.5% daily change saturates.
    mcap_score = float(np.clip(abs(f["circulating_supply_percent_change_1d"]) * 200.0, 0.0, 100.0))

    # Component 4 — realised volatility (15%). 1% volatility saturates.
    mean_24h = f["close_mean_1440m"]
    vol = f["close_std_1440m"] / mean_24h if mean_24h else 0.0
    vol_score = float(np.clip(vol * 10000.0, 0.0, 100.0))

    score = peg_score * 0.40 + volume_score * 0.25 + mcap_score * 0.20 + vol_score * 0.15
    return float(np.clip(score, 0.0, 100.0))


class DepegEnsemble:
    """The trained hybrid stacked ensemble, loaded once and reused."""

    def __init__(self, model_dir=MODEL_DIR, prefix: str = MODEL_PREFIX):
        d = str(model_dir)
        p = f"{d}/{prefix}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.config: Dict = joblib.load(f"{p}_config.pkl")
            self.rf = joblib.load(f"{p}_rf.pkl")
            self.xgb = joblib.load(f"{p}_xgb.pkl")
            self.meta = joblib.load(f"{p}_meta.pkl")

        # The forest was pickled with n_jobs=-1. For single-row serving that
        # spawns a joblib pool per call and costs ~3.3x more wall-clock than
        # running in-thread, for bit-identical output (verified in
        # tests/test_model_contract.py). Streaming scores one row at a time, so
        # force in-thread evaluation.
        try:
            self.rf.n_jobs = 1
        except Exception:
            pass

        self.feature_cols: List[str] = list(self.config["feature_cols"])
        self.threshold: float = float(self.config["optimal_threshold"])
        self.target_precision: float = float(self.config.get("target_precision", 0.90))
        self.min_recall: float = float(self.config.get("min_recall", 0.30))

        # Fail fast on any shape mismatch rather than producing garbage later.
        if self.rf.n_features_in_ != len(self.feature_cols):
            raise ValueError(
                f"RF expects {self.rf.n_features_in_} features, config lists {len(self.feature_cols)}"
            )
        if self.meta.n_features_in_ != 7:
            raise ValueError(f"Meta-learner expects {self.meta.n_features_in_} inputs, expected 7")

    # ------------------------------------------------------------------ stack
    @staticmethod
    def build_meta_features(rf_proba: np.ndarray, xgb_proba: np.ndarray) -> np.ndarray:
        """The exact 7 meta-features from ImprovedHybridDepegPredictor.

        Order matters — the meta-learner's coefficients are positional.
        """
        return np.column_stack([
            rf_proba,                                    # 1 RF confidence
            xgb_proba,                                   # 2 XGB confidence
            rf_proba * xgb_proba,                        # 3 agreement signal
            np.abs(rf_proba - xgb_proba),                # 4 disagreement
            np.minimum(rf_proba, xgb_proba),             # 5 conservative view
            (rf_proba + xgb_proba) / 2.0,                # 6 average confidence
            ((rf_proba > 0.5).astype(int) & (xgb_proba > 0.5).astype(int)),  # 7 both positive
        ])

    # ------------------------------------------------------------------ score
    def predict_batch(self, X: Sequence[Sequence[float]]) -> List[Prediction]:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_p = self.rf.predict_proba(arr)[:, 1]
            xgb_p = self.xgb.predict_proba(arr)[:, 1]
            final = self.meta.predict_proba(self.build_meta_features(rf_p, xgb_p))[:, 1]

        out: List[Prediction] = []
        for i in range(arr.shape[0]):
            out.append(Prediction(
                rf_proba=float(rf_p[i]),
                xgb_proba=float(xgb_p[i]),
                depeg_proba=float(final[i]),
                threshold=self.threshold,
                breach=bool(final[i] >= self.threshold),
                agreement=float(min(rf_p[i], xgb_p[i])),
                disagreement=float(abs(rf_p[i] - xgb_p[i])),
                stress=0.0,
                stress_band="MINIMAL",
            ))
        return out

    def predict(self, vector: Sequence[float], feats: Dict[str, float]) -> Prediction:
        """Score one minute and attach its liquidity-stress index."""
        pred = self.predict_batch([vector])[0]
        pred.stress = stress_score(feats)
        pred.stress_band = band_for(pred.stress)
        return pred
