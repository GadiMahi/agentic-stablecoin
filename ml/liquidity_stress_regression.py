"""
LIQUIDITY STRESS REGRESSION MODEL FOR STABLECOINS
==================================================

Instead of binary classification (depeg/no depeg), this predicts:
- Continuous liquidity stress score (0-100)
- Peg deviation magnitude
- Time-to-depeg estimation
- Confidence intervals

More useful for risk monitoring than binary alerts!

Architecture: XGBoost Regressor + Quantile Regression
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# LIQUIDITY STRESS SCORE DEFINITION
# ============================================================================

def calculate_liquidity_stress_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate continuous liquidity stress score (0-100)
    
    Components:
    -----------
    1. Peg Deviation (40% weight) - Distance from $1.00
    2. Volume Stress (25% weight) - Abnormal volume changes
    3. Market Cap Stress (20% weight) - Supply/redemption pressure
    4. Volatility Stress (15% weight) - Price instability
    
    Returns:
    --------
    pd.Series: Stress score 0-100
        0-20: Minimal stress (healthy)
        20-40: Low stress (monitor)
        40-60: Moderate stress (caution)
        60-80: High stress (warning)
        80-100: Critical stress (imminent risk)
    """
    
    print("\n📊 Calculating Liquidity Stress Score...")
    
    # Component 1: Peg Deviation Score (0-100)
    peg_deviation = np.abs(df['close'] - 1.0)
    peg_score = np.clip(peg_deviation * 5000, 0, 100)  # 2% deviation = 100
    
    # Component 2: Volume Stress Score (0-100)
    # Extreme volume (>3x normal) or extremely low volume (<0.3x normal)
    df['volume_ratio_24h'] = df['volume'] / (df['volume'].rolling(24).mean() + 1e-8)
    volume_stress_high = np.clip((df['volume_ratio_24h'] - 1) * 33, 0, 100)  # >4x = 100
    volume_stress_low = np.clip((1 - df['volume_ratio_24h']) * 143, 0, 100)  # <0.3x = 100
    volume_score = np.maximum(volume_stress_high, volume_stress_low)
    
    # Component 3: Market Cap Stress Score (0-100)
    # Rapid supply changes indicate redemption pressure
    supply_change_1d = np.abs(df['circulating_supply_percent_change_1d'])
    mcap_score = np.clip(supply_change_1d * 200, 0, 100)  # 0.5% change = 100
    
    # Component 4: Volatility Stress Score (0-100)
    # Using your Realized_Daily_Volatility if available, otherwise calculate
    if 'Realized_Daily_Volatility' in df.columns:
        volatility = df['Realized_Daily_Volatility']
    else:
        volatility = df['close'].rolling(24).std() / df['close'].rolling(24).mean()
    
    vol_score = np.clip(volatility * 10000, 0, 100)  # 1% volatility = 100
    
    # Combined Score (weighted average)
    stress_score = (
        peg_score * 0.40 +
        volume_score * 0.25 +
        mcap_score * 0.20 +
        vol_score * 0.15
    )
    
    # Ensure 0-100 range
    stress_score = np.clip(stress_score, 0, 100)
    
    print(f"   Stress Score Statistics:")
    print(f"   Mean:   {stress_score.mean():.2f}")
    print(f"   Median: {stress_score.median():.2f}")
    print(f"   95th %: {stress_score.quantile(0.95):.2f}")
    print(f"   Max:    {stress_score.max():.2f}")
    
    return stress_score


# ============================================================================
# MULTI-TARGET REGRESSION MODEL
# ============================================================================

class LiquidityStressRegressor:
    """
    Professional-grade regression model for liquidity stress prediction
    
    Predicts multiple targets:
    1. Stress score (next 24h average)
    2. Maximum stress (next 24h peak)
    3. Peg deviation (maximum in next 24h)
    4. Confidence intervals (quantile regression)
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        # Multiple models for different targets
        self.stress_model = None      # Mean stress predictor
        self.max_stress_model = None  # Peak stress predictor
        self.peg_model = None         # Peg deviation predictor
        
        # Quantile models (for confidence intervals)
        self.quantile_10_model = None  # 10th percentile
        self.quantile_90_model = None  # 90th percentile
        
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def create_regression_targets(self, 
                                  df: pd.DataFrame,
                                  stress_col: str = 'stress_score',
                                  lookahead_hours: int = 24) -> pd.DataFrame:
        """
        Create multiple regression targets for next N hours
        """
        
        print(f"\n🎯 Creating Regression Targets (lookahead: {lookahead_hours}h)...")
        
        # Target 1: Average stress in next 24h
        df['target_stress_mean_24h'] = (
            df[stress_col]
            .shift(-1)
            .rolling(lookahead_hours)
            .mean()
        )
        
        # Target 2: Maximum stress in next 24h
        df['target_stress_max_24h'] = (
            df[stress_col]
            .shift(-1)
            .rolling(lookahead_hours)
            .max()
        )
        
        # Target 3: Maximum peg deviation in next 24h
        df['target_peg_deviation_max_24h'] = (
            np.abs(df['close'] - 1.0)
            .shift(-1)
            .rolling(lookahead_hours)
            .max()
        )
        
        # Target 4: Stress volatility in next 24h
        df['target_stress_volatility_24h'] = (
            df[stress_col]
            .shift(-1)
            .rolling(lookahead_hours)
            .std()
        )
        
        print(f"   Created 4 regression targets")
        print(f"   Target ranges:")
        print(f"     Stress Mean:     {df['target_stress_mean_24h'].min():.2f} - {df['target_stress_mean_24h'].max():.2f}")
        print(f"     Stress Max:      {df['target_stress_max_24h'].min():.2f} - {df['target_stress_max_24h'].max():.2f}")
        print(f"     Peg Dev Max:     {df['target_peg_deviation_max_24h'].min():.4f} - {df['target_peg_deviation_max_24h'].max():.4f}")
        
        return df
    
    def train(self,
             X_train: pd.DataFrame,
             y_train_dict: Dict[str, pd.Series],
             X_val: pd.DataFrame,
             y_val_dict: Dict[str, pd.Series],
             feature_names: list = None):
        """
        Train multiple XGBoost regressors
        
        Parameters:
        -----------
        X_train, X_val : Features
        y_train_dict, y_val_dict : Dict of targets
            {'stress_mean': Series, 'stress_max': Series, ...}
        """
        
        print("\n" + "="*80)
        print("TRAINING LIQUIDITY STRESS REGRESSION MODELS")
        print("="*80)
        
        self.feature_names = feature_names if feature_names else list(X_train.columns)
        
        # Scale features (important for regression!)
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        # Configuration for regression
        base_params = {
            'n_estimators': 500,
            'max_depth': 5,              # Can be deeper for regression
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.5,
            'reg_alpha': 0.3,
            'reg_lambda': 1.0,
            'objective': 'reg:squarederror',  # MSE loss
            'eval_metric': ['rmse', 'mae'],
            'early_stopping_rounds': 30,
            'random_state': self.random_state,
            'tree_method': 'hist'
        }
        
        # Train Model 1: Mean Stress Predictor
        print("\n[1/5] Training Mean Stress Predictor...")
        self.stress_model = xgb.XGBRegressor(**base_params)
        self.stress_model.fit(
            X_train_scaled, y_train_dict['stress_mean'],
            eval_set=[(X_val_scaled, y_val_dict['stress_mean'])],
            verbose=50
        )
        
        # Train Model 2: Max Stress Predictor
        print("\n[2/5] Training Max Stress Predictor...")
        self.max_stress_model = xgb.XGBRegressor(**base_params)
        self.max_stress_model.fit(
            X_train_scaled, y_train_dict['stress_max'],
            eval_set=[(X_val_scaled, y_val_dict['stress_max'])],
            verbose=50
        )
        
        # Train Model 3: Peg Deviation Predictor
        print("\n[3/5] Training Peg Deviation Predictor...")
        self.peg_model = xgb.XGBRegressor(**base_params)
        self.peg_model.fit(
            X_train_scaled, y_train_dict['peg_deviation'],
            eval_set=[(X_val_scaled, y_val_dict['peg_deviation'])],
            verbose=50
        )
        
        # Train Model 4: 10th Percentile (Lower Bound)
        print("\n[4/5] Training 10th Percentile Model (Lower Confidence)...")
        quantile_10_params = base_params.copy()
        quantile_10_params['objective'] = 'reg:quantileerror'
        quantile_10_params['quantile_alpha'] = 0.10
        
        self.quantile_10_model = xgb.XGBRegressor(**quantile_10_params)
        self.quantile_10_model.fit(
            X_train_scaled, y_train_dict['stress_mean'],
            eval_set=[(X_val_scaled, y_val_dict['stress_mean'])],
            verbose=50
        )
        
        # Train Model 5: 90th Percentile (Upper Bound)
        print("\n[5/5] Training 90th Percentile Model (Upper Confidence)...")
        quantile_90_params = base_params.copy()
        quantile_90_params['objective'] = 'reg:quantileerror'
        quantile_90_params['quantile_alpha'] = 0.90
        
        self.quantile_90_model = xgb.XGBRegressor(**quantile_90_params)
        self.quantile_90_model.fit(
            X_train_scaled, y_train_dict['stress_mean'],
            eval_set=[(X_val_scaled, y_val_dict['stress_mean'])],
            verbose=50
        )
        
        print("\n✅ All models trained successfully!")
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions with all models
        
        Returns:
        --------
        dict: {
            'stress_mean': mean stress prediction,
            'stress_max': peak stress prediction,
            'peg_deviation': max peg deviation,
            'lower_bound': 10th percentile,
            'upper_bound': 90th percentile
        }
        """
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        predictions = {
            'stress_mean': self.stress_model.predict(X_scaled),
            'stress_max': self.max_stress_model.predict(X_scaled),
            'peg_deviation': self.peg_model.predict(X_scaled),
            'lower_bound': self.quantile_10_model.predict(X_scaled),
            'upper_bound': self.quantile_90_model.predict(X_scaled)
        }
        
        return predictions
    
    def evaluate(self, 
                X_test: pd.DataFrame,
                y_test_dict: Dict[str, pd.Series],
                plot: bool = True) -> Dict:
        """
        Comprehensive evaluation of regression models
        """
        
        print("\n" + "="*80)
        print("REGRESSION MODEL EVALUATION")
        print("="*80)
        
        # Get predictions
        preds = self.predict(X_test)
        
        # Calculate metrics for each target
        metrics = {}
        
        for target_name in ['stress_mean', 'stress_max', 'peg_deviation']:
            y_true = y_test_dict[target_name]
            y_pred = preds[target_name]
            
            # Remove NaN
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            # Calculate metrics
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            r2 = r2_score(y_true_clean, y_pred_clean)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100
            
            metrics[target_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape
            }
        
        # Print results
        print(f"\n{'Model':<30} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'MAPE':>10}")
        print("-" * 80)
        
        for target_name, m in metrics.items():
            assessment = '✅' if m['R²'] > 0.6 else '⚠️' if m['R²'] > 0.4 else '❌'
            print(f"{target_name:<30} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['R²']:>10.4f} {m['MAPE']:>9.2f}% {assessment}")
        
        # Coverage of confidence intervals
        in_bounds = (
            (y_test_dict['stress_mean'] >= preds['lower_bound']) & 
            (y_test_dict['stress_mean'] <= preds['upper_bound'])
        )
        coverage = in_bounds.sum() / len(in_bounds) * 100
        
        print(f"\nConfidence Interval Coverage: {coverage:.1f}%")
        print(f"  (Target: ~80% for 10th-90th percentile range)")
        
        if plot:
            self._plot_evaluation(y_test_dict, preds)
        
        return metrics
    
    def _plot_evaluation(self, y_test_dict, preds):
        """Generate evaluation plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Actual vs Predicted (Mean Stress)
        y_true = y_test_dict['stress_mean'].values
        y_pred = preds['stress_mean']
        
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        
        axes[0, 0].scatter(y_true[mask], y_pred[mask], alpha=0.5, s=10)
        axes[0, 0].plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Stress Score', fontsize=12)
        axes[0, 0].set_ylabel('Predicted Stress Score', fontsize=12)
        axes[0, 0].set_title('Mean Stress: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        residuals = y_true[mask] - y_pred[mask]
        axes[0, 1].scatter(y_pred[mask], residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Stress Score', fontsize=12)
        axes[0, 1].set_ylabel('Residuals', fontsize=12)
        axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Time series with confidence intervals
        time_idx = np.arange(min(1000, len(y_true)))
        axes[0, 2].plot(time_idx, y_true[time_idx], label='Actual', alpha=0.7, linewidth=1)
        axes[0, 2].plot(time_idx, y_pred[time_idx], label='Predicted', alpha=0.7, linewidth=1)
        axes[0, 2].fill_between(
            time_idx,
            preds['lower_bound'][time_idx],
            preds['upper_bound'][time_idx],
            alpha=0.2,
            label='80% Confidence Interval'
        )
        axes[0, 2].set_xlabel('Time (hours)', fontsize=12)
        axes[0, 2].set_ylabel('Stress Score', fontsize=12)
        axes[0, 2].set_title('Time Series with Confidence Intervals', fontsize=14, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Max Stress Distribution
        y_max_true = y_test_dict['stress_max'].values
        y_max_pred = preds['stress_max']
        
        mask_max = ~(np.isnan(y_max_true) | np.isnan(y_max_pred))
        
        axes[1, 0].hist(y_max_true[mask_max], bins=50, alpha=0.5, label='Actual', density=True)
        axes[1, 0].hist(y_max_pred[mask_max], bins=50, alpha=0.5, label='Predicted', density=True)
        axes[1, 0].set_xlabel('Max Stress Score', fontsize=12)
        axes[1, 0].set_ylabel('Density', fontsize=12)
        axes[1, 0].set_title('Max Stress Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Peg Deviation Predictions
        y_peg_true = y_test_dict['peg_deviation'].values
        y_peg_pred = preds['peg_deviation']
        
        mask_peg = ~(np.isnan(y_peg_true) | np.isnan(y_peg_pred))
        
        axes[1, 1].scatter(y_peg_true[mask_peg] * 100, y_peg_pred[mask_peg] * 100, alpha=0.5, s=10)
        max_val = max(y_peg_true[mask_peg].max(), y_peg_pred[mask_peg].max()) * 100
        axes[1, 1].plot([0, max_val], [0, max_val], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Peg Deviation (%)', fontsize=12)
        axes[1, 1].set_ylabel('Predicted Peg Deviation (%)', fontsize=12)
        axes[1, 1].set_title('Peg Deviation: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Feature Importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.stress_model.feature_importances_
        }).sort_values('importance', ascending=True).tail(20)
        
        axes[1, 2].barh(range(len(importance)), importance['importance'])
        axes[1, 2].set_yticks(range(len(importance)))
        axes[1, 2].set_yticklabels(importance['feature'], fontsize=9)
        axes[1, 2].set_xlabel('Importance', fontsize=12)
        axes[1, 2].set_title('Top 20 Feature Importance', fontsize=14, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('regression_evaluation.png', dpi=150, bbox_inches='tight')
        print("\n📊 Evaluation plots saved to: regression_evaluation.png")
        plt.show()
    
    def get_risk_level(self, stress_score: float) -> str:
        """Convert stress score to risk level"""
        if stress_score < 20:
            return "MINIMAL (Healthy)"
        elif stress_score < 40:
            return "LOW (Monitor)"
        elif stress_score < 60:
            return "MODERATE (Caution)"
        elif stress_score < 80:
            return "HIGH (Warning)"
        else:
            return "CRITICAL (Imminent Risk)"
    
    def save_model(self, filepath_prefix: str = 'liquidity_stress_model'):
        """Save all models"""
        import pickle
        
        models = {
            'stress_model': self.stress_model,
            'max_stress_model': self.max_stress_model,
            'peg_model': self.peg_model,
            'quantile_10_model': self.quantile_10_model,
            'quantile_90_model': self.quantile_90_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(f'{filepath_prefix}.pkl', 'wb') as f:
            pickle.dump(models, f)
        
        print(f"\n💾 Models saved to: {filepath_prefix}.pkl")


# ============================================================================
# COMPLETE TRAINING PIPELINE
# ============================================================================

def train_liquidity_stress_regressor(df: pd.DataFrame,
                                     feature_cols: list,
                                     lookahead_hours: int = 24):
    """
    Complete training pipeline for liquidity stress regression
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with all features (must be sorted by time!)
    feature_cols : list
        Feature column names
    lookahead_hours : int
        Prediction horizon (default: 24)
    
    Returns:
    --------
    LiquidityStressRegressor : Trained model
    """
    
    print("\n" + "="*80)
    print("LIQUIDITY STRESS REGRESSION - COMPLETE PIPELINE")
    print("="*80)
    
    # Step 1: Calculate stress score
    df['stress_score'] = calculate_liquidity_stress_score(df)
    
    # Step 2: Create regression targets
    df = model.create_regression_targets(df, 'stress_score', lookahead_hours)
    
    # Step 3: Clean data
    df_clean = df.dropna()
    print(f"\nRows after cleaning: {len(df_clean):,}")
    
    # Step 4: Split data (time-aware)
    train_size = int(len(df_clean) * 0.6)
    val_size = int(len(df_clean) * 0.2)
    
    train_df = df_clean.iloc[:train_size]
    val_df = df_clean.iloc[train_size:train_size+val_size]
    test_df = df_clean.iloc[train_size+val_size:]
    
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    
    y_train = {
        'stress_mean': train_df['target_stress_mean_24h'],
        'stress_max': train_df['target_stress_max_24h'],
        'peg_deviation': train_df['target_peg_deviation_max_24h']
    }
    
    y_val = {
        'stress_mean': val_df['target_stress_mean_24h'],
        'stress_max': val_df['target_stress_max_24h'],
        'peg_deviation': val_df['target_peg_deviation_max_24h']
    }
    
    y_test = {
        'stress_mean': test_df['target_stress_mean_24h'],
        'stress_max': test_df['target_stress_max_24h'],
        'peg_deviation': test_df['target_peg_deviation_max_24h']
    }
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    # Step 5: Initialize and train
    model = LiquidityStressRegressor()
    model.train(X_train, y_train, X_val, y_val, feature_names=feature_cols)
    
    # Step 6: Evaluate
    metrics = model.evaluate(X_test, y_test, plot=True)
    
    # Step 7: Save
    model.save_model('liquidity_stress_model')
    
    return model, metrics


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    print("""
USAGE:
======

# Step 1: Get features from feature engineering
from complete_feature_engineering import run_complete_pipeline

X_train, X_val, X_test, y_train, y_val, y_test, features = run_complete_pipeline(
    'your_data.csv'
)

# Combine for regression training
df_full = pd.concat([X_train, X_val, X_test]).reset_index(drop=True)
df_full['close'] = ...  # Need close price for stress calculation

# Step 2: Train regression model
from liquidity_stress_regression import train_liquidity_stress_regressor

model, metrics = train_liquidity_stress_regressor(
    df=df_full,
    feature_cols=features,
    lookahead_hours=24
)

# Step 3: Make predictions
predictions = model.predict(X_new)

print(f"Predicted stress score: {predictions['stress_mean'][0]:.1f}")
print(f"Risk level: {model.get_risk_level(predictions['stress_mean'][0])}")
print(f"Confidence interval: [{predictions['lower_bound'][0]:.1f}, {predictions['upper_bound'][0]:.1f}]")
print(f"Max stress next 24h: {predictions['stress_max'][0]:.1f}")
print(f"Max peg deviation: {predictions['peg_deviation'][0]*100:.3f}%")
    """)
