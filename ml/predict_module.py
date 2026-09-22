"""
DEPEG CLASSIFIER - INFERENCE MODULE
====================================

Hybrid stacked ensemble for predicting depeg events in next 24h.
No web framework needed - just import and call.

Usage:
------
from depeg_predictor import DepegPredictor

# Initialize once (loads models)
predictor = DepegPredictor()

# Use many times (fast predictions)
result = predictor.predict(your_features)
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

# Setup logging
logger = logging.getLogger(__name__)


class DepegPredictor:
    """
    In-process depeg event predictor using hybrid stacked ensemble
    
    Load once, use many times - optimized for speed
    """
    
    def __init__(self, 
                 rf_model_path: str = 'improved_hybrid_depeg_rf.pkl',
                 xgb_model_path: str = 'improved_hybrid_depeg_xgb.pkl',
                 meta_model_path: str = 'improved_hybrid_depeg_meta.pkl',
                 config_path: str = 'improved_hybrid_depeg_config.pkl'):
        """
        Initialize predictor (loads all models)
        
        Parameters:
        -----------
        rf_model_path : str
            Path to Random Forest model
        xgb_model_path : str
            Path to XGBoost model
        meta_model_path : str
            Path to meta-learner (Logistic Regression)
        config_path : str
            Path to configuration (features, threshold, etc.)
        """
        
        logger.info("Loading depeg prediction models...")
        
        try:
            # Load models
            self.rf_model = joblib.load(rf_model_path)
            self.xgb_model = joblib.load(xgb_model_path)
            self.meta_model = joblib.load(meta_model_path)
            
            # Load configuration
            config = joblib.load(config_path)
            self.feature_cols = config['feature_cols']
            self.optimal_threshold = config['optimal_threshold']
            self.target_precision = config.get('target_precision', 0.90)
            self.min_recall = config.get('min_recall', 0.30)
            
            logger.info(f"✅ RF Model loaded: {self.rf_model.n_estimators} trees")
            logger.info(f"✅ XGB Model loaded: {self.xgb_model.n_estimators} trees")
            logger.info(f"✅ Meta Model loaded (threshold: {self.optimal_threshold:.4f})")
            logger.info(f"✅ Required features: {len(self.feature_cols)}")
            
            # Store for metadata
            self.loaded_at = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise RuntimeError(f"Could not load models: {e}")
    
    def _clean_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Clean and convert features to proper numeric types"""
        cleaned = {}
        for key, value in features.items():
            # Handle empty strings, None, NaN
            if value == "" or value is None or (isinstance(value, float) and np.isnan(value)):
                cleaned[key] = 0.0
            else:
                try:
                    cleaned[key] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert '{key}': '{value}' to float, using 0.0")
                    cleaned[key] = 0.0
        return cleaned
    
    def _create_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Generate meta-features from base model predictions"""
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        
        # Meta-features (same as training)
        meta_features = np.column_stack([
            rf_proba,                           # RF confidence
            xgb_proba,                          # XGB confidence
            rf_proba * xgb_proba,               # Agreement signal
            np.abs(rf_proba - xgb_proba),       # Disagreement
            np.minimum(rf_proba, xgb_proba),    # Conservative view
            (rf_proba + xgb_proba) / 2,         # Average confidence
            (rf_proba > 0.5).astype(int) & (xgb_proba > 0.5).astype(int),  # Both predict positive
        ])
        
        return meta_features
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict depeg event probability for next 24h
        
        Parameters:
        -----------
        features : dict
            Your feature dictionary (must contain all required features)
            Example: {'close': 0.9998, 'volume': 1234567, ...}
        
        Returns:
        --------
        dict : Prediction result
            {
                'depeg_predicted': True/False,
                'depeg_probability': 0.8523,
                'confidence_level': 'HIGH',
                'threshold_used': 0.8500,
                'risk_classification': 'CRITICAL',
                'alert_required': True,
                'timestamp': '2024-01-15T14:30:00',
                'inference_time_ms': 15.2
            }
        """
        
        start_time = datetime.now()
        
        try:
            # Validate input
            if not isinstance(features, dict):
                raise ValueError(f"Features must be dict, got {type(features)}")
            
            # Clean features (handle empty strings, None, etc.)
            features = self._clean_features(features)
            
            # Check for missing required features
            missing = [f for f in self.feature_cols if f not in features]
            if missing:
                logger.warning(f"Missing {len(missing)} features, filling with 0: {missing[:5]}...")
                for feat in missing:
                    features[feat] = 0.0
            
            # Select and order features
            X = pd.DataFrame([features])[self.feature_cols]
            
            # Handle NaN (should be rare after cleaning, but just in case)
            if X.isna().any().any():
                logger.warning("NaN values detected after cleaning, filling with 0")
                X = X.fillna(0)
            
            # Handle inf values
            X = X.replace([np.inf, -np.inf], 0)
            
            # Create meta-features
            meta_features = self._create_meta_features(X)
            
            # Get probability from meta-model
            probability = float(self.meta_model.predict_proba(meta_features)[:, 1][0])
            
            # Binary prediction using optimal threshold
            prediction = int(probability >= self.optimal_threshold)
            
            # Confidence level based on probability
            if probability >= 0.90:
                confidence = "VERY_HIGH"
            elif probability >= 0.70:
                confidence = "HIGH"
            elif probability >= 0.50:
                confidence = "MEDIUM"
            elif probability >= 0.30:
                confidence = "LOW"
            else:
                confidence = "VERY_LOW"
            
            # Risk classification
            if prediction == 1:  # Depeg predicted
                if probability >= 0.95:
                    risk_class = "CRITICAL"
                    alert = True
                elif probability >= 0.85:
                    risk_class = "HIGH"
                    alert = True
                else:
                    risk_class = "MODERATE"
                    alert = True
            else:  # No depeg predicted
                if probability >= 0.50:
                    risk_class = "ELEVATED"
                    alert = False
                else:
                    risk_class = "NORMAL"
                    alert = False
            
            # Inference time
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Build result
            result = {
                'depeg_predicted': bool(prediction),
                'depeg_probability': round(probability, 6),
                'confidence_level': confidence,
                'threshold_used': round(self.optimal_threshold, 6),
                'risk_classification': risk_class,
                'alert_required': alert,
                'timestamp': datetime.now().isoformat(),
                'inference_time_ms': round(inference_time, 2)
            }
            
            logger.debug(
                f"Prediction: {'DEPEG' if prediction else 'SAFE'} "
                f"(p={probability:.4f}, {confidence}) in {inference_time:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Predict for multiple samples (more efficient)
        
        Parameters:
        -----------
        features_list : list of dict
            List of feature dictionaries
        
        Returns:
        --------
        list of dict : Predictions for each sample
        """
        
        results = []
        for features in features_list:
            result = self.predict(features)
            results.append(result)
        
        return results
    
    def get_required_features(self) -> List[str]:
        """Get list of required feature names"""
        return self.feature_cols.copy()
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': 'Hybrid Stacked Ensemble',
            'base_models': ['RandomForest', 'XGBoost'],
            'meta_model': 'LogisticRegression',
            'rf_trees': self.rf_model.n_estimators,
            'xgb_trees': self.xgb_model.n_estimators,
            'required_features': len(self.feature_cols),
            'optimal_threshold': self.optimal_threshold,
            'target_precision': self.target_precision,
            'min_recall': self.min_recall,
            'loaded_at': self.loaded_at.isoformat()
        }


# ============================================================================
# SINGLETON PATTERN (Optional - for even faster repeated use)
# ============================================================================

_global_predictor = None

def get_predictor() -> DepegPredictor:
    """
    Get global predictor instance (loads once, reuses many times)
    
    Usage:
    ------
    predictor = get_predictor()
    result = predictor.predict(features)
    """
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = DepegPredictor()
    return _global_predictor


# ============================================================================
# CONVENIENCE FUNCTION (Simplest usage)
# ============================================================================

def predict_depeg(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Convenience function for one-line predictions
    
    Usage:
    ------
    from depeg_predictor import predict_depeg
    
    result = predict_depeg(your_features)
    print(f"Depeg predicted: {result['depeg_predicted']}")
    
    Parameters:
    -----------
    features : dict
        Your features dictionary
    
    Returns:
    --------
    dict : Prediction result
    """
    predictor = get_predictor()
    return predictor.predict(features)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("DEPEG CLASSIFIER - DEMO")
    print("="*80)
    
    # Method 1: Direct instantiation
    print("\n1. Direct instantiation:")
    predictor = DepegPredictor()
    print(f"   Model info: {predictor.get_info()}")
    
    # Create sample features (you'd get these from your feature engineering pipeline)
    sample_features = {
        "open_time": "12-02-2026 09:04",
        "open": 1.0007,
        "high": 1.0007,
        "low": 1.0005,
        "close": 1.0006,
        "volume": 34702609,
        "close_time": 1770890000000,
        "quote_volume": 34723377.12,
        "trades": 731,
        "taker_base": 1176897,
        "taker_quote": 1177603.587,
        "ignore": 0,
        "percent_change_1h": 0,
        "percent_change_24h": 0.009995002,
        "percent_change_7d": -0.109813317,
        "percent_change_30d": -0.019984013,
        "price": 0.999411598,
        "market_cap": 183912000000,
        "circulating_supply": 184020000000,
        "market_cap_percent_change_1d": -0.002156755,
        "circulating_supply_percent_change_1d": -0.002069909,
        "market_cap_percent_change_7d": -0.007736991,
        "circulating_supply_percent_change_7d": -0.009189228,
        "market_cap_percent_change_30d": -0.015198175,
        "circulating_supply_percent_change_30d": -0.015704044,
        "volume_percent_change_24h": "",  # Empty string - will be handled
        "volume_percent_change_7d": "",   # Empty string - will be handled
        "volume_percent_change_30d": "",  # Empty string - will be handled
        "daily_range": 0.00019986009593422308,
        "body_size": 0.00009993004796711154,
        "upper_shadow": 0,
        "lower_shadow": 0.4999750012499375,
        "close_position": 0.4999750012499375,
        "trade_size": 47407.93579234973,
        "taker_buy_ratio": 0.033913790170646815,
        "taker_sell_ratio": 0.033913855294959855,
        "volume_imbalance": -0.000020361206847590553,
        "mcap_to_supply_ratio": 0.9994131072709488,
        "price_to_mcap_billion": 0.005434183728879345,
        "close_mean_180m": 1.0006083333333333,
        "close_std_180m": 0.00007001595991999625,
        "close_min_180m": 1.0005,
        "close_max_180m": 1.0007,
        "volume_mean_180m": 1936115.1888888888,
        "volume_std_180m": 4614007.935109943,
        "volume_sum_180m": 348500734,
        "trades_mean_180m": 259.44444444444446,
        "trade_size_mean_180m": 5373.094515500073,
        "market_cap_mean_180m": 183912000000,
        "market_cap_std_180m": 0,
        "circulating_supply_mean_180m": 184020000000,
        "close_mean_360m": 1.0006252777777778,
        "close_std_360m": 0.00006463896363136311,
        "close_min_360m": 1.0005,
        "close_max_360m": 1.0007,
        "volume_mean_360m": 1262621.7527777778,
        "volume_std_360m": 3513064.9530560174,
        "volume_sum_360m": 454543831,
        "trades_mean_360m": 234.03333333333333,
        "trade_size_mean_360m": 3863.0822671587916,
        "market_cap_mean_360m": 183912000000,
        "market_cap_std_360m": 0,
        "circulating_supply_mean_360m": 184020000000,
        "close_mean_720m": 1.000663472222222,
        "close_std_720m": 0.00009915722354065815,
        "close_min_720m": 1.0005,
        "close_max_720m": 1.001,
        "volume_mean_720m": 968729.1277777777,
        "volume_std_720m": 3355724.4140446703,
        "volume_sum_720m": 697484972,
        "trades_mean_720m": 217.05277777777778,
        "trade_size_mean_720m": 3127.424931027303,
        "market_cap_mean_720m": 184594013888.8889,
        "market_cap_std_720m": 1204409711.644504,
        "circulating_supply_mean_720m": 184736284722.22223,
        "close_mean_1440m": 1.0007599305555555,
        "close_std_1440m": 0.00018497786987924084,
        "close_min_1440m": 1.0003,
        "close_max_1440m": 1.0011,
        "volume_mean_1440m": 1134772.7222222222,
        "volume_std_1440m": 3385597.658944666,
        "volume_sum_1440m": 1634072720,
        "trades_mean_1440m": 232.89652777777778,
        "trade_size_mean_1440m": 3551.588624559335,
        "market_cap_mean_1440m": 185656006944.44446,
        "market_cap_std_1440m": 1361400144.914383,
        "circulating_supply_mean_1440m": 185851642361.1111,
        "peg_deviation": 0.0005999999999999339,
        "peg_deviation_pct": 0.05999999999999339,
        "peg_direction": 0.0005999999999999339,
        "above_peg": 1,
        "below_peg": 0,
        "peg_deviation_mean_180m": 0.0006083333333332663,
        "peg_deviation_max_180m": 0.0006999999999999229,
        "peg_deviation_std_180m": 0.00007001595987251303,
        "peg_deviation_mean_360m": 0.0006252777777777089,
        "peg_deviation_max_360m": 0.0006999999999999229,
        "peg_deviation_std_360m": 0.00006463896352097056,
        "peg_deviation_mean_720m": 0.0006634722222221491,
        "peg_deviation_max_720m": 0.0009999999999998899,
        "peg_deviation_std_720m": 0.00009915722363605646,
        "peg_deviation_mean_1440m": 0.0007599305555554924,
        "peg_deviation_max_1440m": 0.001100000000000101,
        "peg_deviation_std_1440m": 0.00018497787005020085,
        "peg_stress_1pct_3h": 0,
        "peg_stress_1pct_24h": 0,
        "peg_stress_2pct_3h": 0,
        "peg_stress_2pct_24h": 0,
        "volume_vs_3h": 17.923834903601573,
        "volume_vs_24h": 30.581109609368887,
        "volume_vs_7d": 22.761109024310834,
        "volume_vs_30d": 28.1305592693891,
        "trade_size_vs_24h": 13.348374714455678,
        "trade_size_vs_7d": 14.238030296063378,
        "mcap_to_volume_24h": 112.54823469545468,
        "mcap_to_volume_7d": 11.966876999369886,
        "price_accel_1h_24h": -0.009995002,
        "price_accel_24h_7d": 0.119808319,
        "price_accel_7d_30d": -0.089829304,
        "volume_spike_peg_stress": 0,
        "extreme_volume_spike": 1,
        "large_trade_anomaly": 1,
    }
    
    # Predict
    result = predictor.predict(sample_features)
    
    print("\n2. Prediction result:")
    print(f"   Depeg Predicted: {result['depeg_predicted']}")
    print(f"   Probability: {result['depeg_probability']:.4f}")
    print(f"   Confidence: {result['confidence_level']}")
    print(f"   Risk Class: {result['risk_classification']}")
    print(f"   Alert Required: {result['alert_required']}")
    print(f"   Inference Time: {result['inference_time_ms']:.1f}ms")
    
    # Method 2: Convenience function
    print("\n3. Using convenience function:")
    result2 = predict_depeg(sample_features)
    print(f"   Depeg: {result2['depeg_predicted']} (p={result2['depeg_probability']:.4f})")
    
    # Method 3: Batch prediction
    print("\n4. Batch prediction:")
    batch_features = [sample_features] * 5
    batch_results = predictor.predict_batch(batch_features)
    print(f"   Predicted {len(batch_results)} samples")
    
    print("\n" + "="*80)
    print("INTEGRATION EXAMPLE")
    print("="*80)
    print("""
# In your existing backend server:

from depeg_predictor import DepegPredictor

# Initialize once at server startup
depeg_predictor = DepegPredictor()

# In your data processing function (called every minute):
def process_new_data():
    # 1. Fetch raw data from API
    raw_data = fetch_from_binance_gecko()
    
    # 2. Compute your features (you already have this)
    features = compute_all_features(raw_data)
    
    # 3. Predict (fast! ~15ms)
    result = depeg_predictor.predict(features)
    
    # 4. Use result
    if result['alert_required']:
        send_alert(f"DEPEG RISK: {result['risk_classification']}")
    
    # 5. Store to database
    db.save({
        'timestamp': result['timestamp'],
        'depeg_predicted': result['depeg_predicted'],
        'depeg_probability': result['depeg_probability'],
        'risk_classification': result['risk_classification']
    })
    
    return result
    """)