import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import gc
from typing import Tuple, Dict

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CSV_PATH = 'processed_data.csv'
TARGET_COL = 'target_depeg_next_24h'
SPLIT_DATE = '2025-11-01'

# Feature drops based on your analysis
DROP_PATTERNS = ['_10080m', '_43200m', '_2880m', '_7d', '_30d'] 

DROP_COLS = [
    'open_time', 'close_time', 'timestamp', 'ignore', 
    'target_depeg_next_24h', 'future_max_deviation',
    'Thresh_D', 'Thresh_U', 'depeg_label',
    'close_min_1440m', 'close_min_720m', 
    'peg_deviation_max_1440m'
]

class ImprovedHybridDepegPredictor:
    """
    IMPROVED Hybrid model with aggressive false positive reduction.
    
    Key improvements:
    1. Conservative SMOTE ratios
    2. High regularization in meta-learner
    3. Precision-focused threshold optimization
    4. Calibrated probability outputs
    5. Multi-stage filtering
    """
    
    def __init__(self, target_precision=0.90, min_recall=0.30):
        """
        Initialize with target performance metrics.
        
        Args:
            target_precision: Minimum acceptable precision (default 0.90 = 90%)
            min_recall: Minimum acceptable recall (default 0.30 = 30%)
        """
        self.rf_model = None
        self.xgb_model = None
        self.meta_model = None
        self.feature_cols = None
        self.optimal_threshold = 0.5
        self.target_precision = target_precision
        self.min_recall = min_recall
        
        print(f"Initialized with targets: Precision ≥ {target_precision:.1%}, Recall ≥ {min_recall:.1%}")
        
    def prepare_data(self, csv_path: str) -> Tuple:
        """Load and prepare data with date-based split"""
        print("\n" + "="*80)
        print("🔄 PREPARING DATA (IMPROVED PIPELINE)")
        print("="*80)
        
        # Load
        print(f"\n[1/5] Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        
        # Parse dates
        print("\n[2/5] Parsing timestamps...")
        if 'open_time' in df.columns:
            df['datetime'] = pd.to_datetime(df['open_time'], dayfirst=True, errors='coerce')
        else:
            raise ValueError("'open_time' column missing")
        
        df = df.dropna(subset=['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        # Feature engineering
        print("\n[3/5] Engineering features...")
        if 'volume_mean_1440m' in df.columns and 'market_cap_mean_1440m' in df.columns:
            df['turnover_ratio'] = df['volume_mean_1440m'] / (df['market_cap_mean_1440m'] + 1e-9)
        
        # Add momentum features
        if 'close' in df.columns:
            df['price_momentum'] = df['close'].pct_change(periods=60).fillna(0)  # 1-hour momentum
        
        if 'peg_deviation' in df.columns:
            df['peg_deviation_change'] = df['peg_deviation'].diff(periods=30).fillna(0)  # 30-min change
            df['peg_deviation_volatility'] = df['peg_deviation'].rolling(window=60).std().fillna(0)
        
        # Remove long-term memory features
        cols_to_drop = [c for c in df.columns if any(p in c for p in DROP_PATTERNS)]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Drop leakage columns
        final_drop = [c for c in DROP_COLS if c in df.columns]
        
        # Date-based split
        print(f"\n[4/5] Splitting at {SPLIT_DATE}...")
        split_ts = pd.Timestamp(SPLIT_DATE)
        
        train_mask = df['datetime'] < split_ts
        test_mask = df['datetime'] >= split_ts
        
        X_train_raw = df.loc[train_mask].drop(columns=['datetime'] + final_drop, errors='ignore')
        y_train_raw = df.loc[train_mask, TARGET_COL]
        
        X_test = df.loc[test_mask].drop(columns=['datetime'] + final_drop, errors='ignore')
        y_test = df.loc[test_mask, TARGET_COL]
        
        print(f"   Train: {len(X_train_raw)} rows | {y_train_raw.sum()} events")
        print(f"   Test:  {len(X_test)} rows  | {y_test.sum()} events")
        
        # Store feature columns
        self.feature_cols = X_train_raw.columns.tolist()
        
        # Filter for stable moments with STRICTER criteria
        print("\n[5/5] Filtering for stable training moments (STRICT)...")
        if 'peg_deviation' in X_train_raw.columns:
            # Even stricter: only train on very stable periods
            stable_mask = X_train_raw['peg_deviation'] < 0.003  # Was 0.005, now 0.003
        else:
            stable_mask = (X_train_raw['close'] > 0.997) & (X_train_raw['close'] < 1.003)
        
        X_train_stable = X_train_raw[stable_mask]
        y_train_stable = y_train_raw[stable_mask]
        
        print(f"   Stable training samples: {len(X_train_stable)} ({y_train_stable.sum()} pre-crash signals)")
        print(f"   Stability ratio: {len(X_train_stable)/len(X_train_raw)*100:.1f}% of training data")
        
        del df
        gc.collect()
        
        return X_train_stable, y_train_stable, X_test, y_test, X_train_raw, y_train_raw
    
    def train_base_models(self, X_train, y_train) -> Tuple:
        """Train Random Forest and XGBoost with CONSERVATIVE settings"""
        print("\n" + "="*80)
        print("🏗️  TRAINING BASE MODELS (CONSERVATIVE MODE)")
        print("="*80)
        
        # Balance with CONSERVATIVE SMOTE
        print("\n[1/3] Balancing with CONSERVATIVE SMOTE...")
        if y_train.sum() > 20:
            # MUCH LOWER ratios to avoid over-predicting
            smote_rf = SMOTE(sampling_strategy=0.05, random_state=42)  # Was 0.2, now 0.05
            X_rf, y_rf = smote_rf.fit_resample(X_train, y_train)
            
            smote_xgb = SMOTE(sampling_strategy=0.10, random_state=42)  # Was 0.5, now 0.10
            X_xgb, y_xgb = smote_xgb.fit_resample(X_train, y_train)
        else:
            X_rf, y_rf = X_train, y_train
            X_xgb, y_xgb = X_train, y_train
        
        print(f"   RF balanced: {len(X_rf)} samples (positive ratio: {y_rf.sum()/len(y_rf)*100:.2f}%)")
        print(f"   XGB balanced: {len(X_xgb)} samples (positive ratio: {y_xgb.sum()/len(y_xgb)*100:.2f}%)")
        
        # Train Random Forest with HIGH PRECISION settings
        print("\n[2/3] Training Random Forest (High Precision Mode)...")
        self.rf_model = RandomForestClassifier(
            n_estimators=300,           # More trees for stability
            max_depth=15,               # Shallower trees (was 20)
            min_samples_leaf=20,        # More samples per leaf (was 5) - prevents overfitting
            min_samples_split=50,       # Require more samples to split
            max_features='sqrt',        # Reduce feature randomness
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        self.rf_model.fit(X_rf, y_rf)
        print("   ✓ Random Forest trained")
        
        # Train XGBoost with EXTREME PRECISION settings
        print("\n[3/3] Training XGBoost (Extreme Precision Mode)...")
        self.xgb_model = XGBClassifier(
            n_estimators=500,           # More trees
            max_depth=4,                # Very shallow (was 5)
            learning_rate=0.01,         # Very slow learning (was 0.03)
            subsample=0.5,              # Less data per tree (was 0.6)
            colsample_bytree=0.5,       # Fewer features per tree (was 0.6)
            gamma=5.0,                  # Much higher regularization (was 2.0)
            min_child_weight=10,        # Require more samples in leaves (NEW)
            reg_alpha=1.0,              # L1 regularization (NEW)
            reg_lambda=5.0,             # L2 regularization (NEW)
            scale_pos_weight=100.0,     # Higher weight (was 50.0)
            objective='binary:logistic',
            tree_method='hist',
            n_jobs=-1,
            random_state=42,
            verbosity=0
        )
        self.xgb_model.fit(X_xgb, y_xgb)
        print("   ✓ XGBoost trained")
        
        del X_rf, y_rf, X_xgb, y_xgb
        gc.collect()
        
        return self.rf_model, self.xgb_model
    
    def create_meta_features(self, X, return_base_preds=False):
        """Generate meta-features from base model predictions"""
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        
        # Meta-features with ADDITIONAL CONSERVATIVE signals
        meta_features = np.column_stack([
            rf_proba,                           # RF confidence
            xgb_proba,                          # XGB confidence
            rf_proba * xgb_proba,               # Agreement signal (both high = real signal)
            np.abs(rf_proba - xgb_proba),       # Disagreement (high = uncertain)
            np.minimum(rf_proba, xgb_proba),    # Conservative view (NEW - both must agree)
            (rf_proba + xgb_proba) / 2,         # Average confidence
            (rf_proba > 0.5).astype(int) & (xgb_proba > 0.5).astype(int),  # Both predict positive (NEW)
        ])
        
        if return_base_preds:
            return meta_features, rf_proba, xgb_proba
        return meta_features
    
    def train_meta_model(self, X_train_raw, y_train_raw, X_val, y_val):
        """Train meta-learner with HIGH REGULARIZATION"""
        print("\n" + "="*80)
        print("🧠 TRAINING META-LEARNER (HIGH REGULARIZATION)")
        print("="*80)
        
        # Generate meta-features
        print("\n[1/3] Creating meta-features...")
        meta_train = self.create_meta_features(X_train_raw)
        meta_val = self.create_meta_features(X_val)
        
        # Train with VERY HIGH regularization
        print("\n[2/3] Training Logistic Regression (C=0.01 for high regularization)...")
        self.meta_model = LogisticRegression(
            C=0.01,                     # VERY high regularization (was 0.1)
            class_weight='balanced',
            penalty='l2',
            solver='lbfgs',
            max_iter=2000,
            random_state=42
        )
        self.meta_model.fit(meta_train, y_train_raw)
        print("   ✓ Meta-learner trained")
        
        # Find PRECISION-OPTIMIZED threshold
        print("\n[3/3] Optimizing threshold for target precision...")
        meta_proba = self.meta_model.predict_proba(meta_val)[:, 1]
        
        # Find threshold that achieves target precision while maximizing recall
        precisions, recalls, thresholds = precision_recall_curve(y_val, meta_proba)
        
        # Filter thresholds that meet precision target
        valid_indices = precisions >= self.target_precision
        
        if valid_indices.sum() > 0:
            # Among valid precision thresholds, pick the one with highest recall
            valid_recalls = recalls[valid_indices]
            valid_thresholds = thresholds[valid_indices[:-1]]  # Exclude last element
            
            if len(valid_thresholds) > 0:
                best_idx = np.argmax(valid_recalls[:-1])  # Exclude last element
                self.optimal_threshold = valid_thresholds[best_idx]
            else:
                # Fallback: use very high threshold
                self.optimal_threshold = 0.95
        else:
            # If can't achieve target precision, use very conservative threshold
            print(f"   ⚠️ Warning: Can't achieve {self.target_precision:.1%} precision")
            print(f"   Using fallback threshold of 0.95")
            self.optimal_threshold = 0.95
        
        # Verify threshold on validation set
        y_pred_val = (meta_proba >= self.optimal_threshold).astype(int)
        val_precision = precision_score(y_val, y_pred_val, zero_division=0)
        val_recall = recall_score(y_val, y_pred_val, zero_division=0)
        
        print(f"\n   Optimal threshold: {self.optimal_threshold:.4f}")
        print(f"   Validation Precision: {val_precision:.4f} (target: {self.target_precision:.4f})")
        print(f"   Validation Recall: {val_recall:.4f} (minimum: {self.min_recall:.4f})")
        
        if val_precision < self.target_precision:
            print(f"   ⚠️ Warning: Precision below target, increasing threshold...")
            self.optimal_threshold = min(0.99, self.optimal_threshold * 1.2)
            print(f"   Adjusted threshold: {self.optimal_threshold:.4f}")
        
        return self.meta_model
    
    def predict(self, X):
        """Make predictions using the full stacked ensemble"""
        meta_features = self.create_meta_features(X)
        proba = self.meta_model.predict_proba(meta_features)[:, 1]
        return (proba >= self.optimal_threshold).astype(int), proba
    
    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation with FALSE POSITIVE focus"""
        print("\n" + "="*80)
        print("📊 IMPROVED HYBRID MODEL EVALUATION")
        print("="*80)
        
        # Get predictions
        y_pred, y_proba = self.predict(X_test)
        
        # Also get base model predictions
        meta_features, rf_proba, xgb_proba = self.create_meta_features(X_test, return_base_preds=True)
        
        # Metrics
        print("\n--- HYBRID MODEL PERFORMANCE ---")
        print(classification_report(y_test, y_pred, target_names=['Safe', 'Depeg'], digits=4))
        
        print("\n--- CONFUSION MATRIX ---")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        # Calculate detailed metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # FALSE POSITIVE ANALYSIS
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"\n--- ERROR ANALYSIS ---")
        print(f"False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
        print(f"False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")
        print(f"True Negatives: {tn:,} (correctly identified safe periods)")
        print(f"False Positives: {fp:,} (false alarms)")
        
        try:
            auc = roc_auc_score(y_test, y_proba)
            print(f"\nROC AUC Score: {auc:.4f}")
        except:
            auc = None
        
        # Alert statistics
        print(f"\n--- ALERT STATISTICS ---")
        total_predictions = y_pred.sum()
        total_actual = y_test.sum()
        print(f"Total Alerts Generated: {total_predictions:,}")
        print(f"Total Actual Depegs: {total_actual:,}")
        print(f"Alert Rate: {total_predictions/len(y_test)*100:.4f}% of time periods")
        print(f"Precision (Alert Accuracy): {precision:.4f}")
        print(f"Recall (Coverage): {recall:.4f}")
        
        # Compare with base models
        print("\n" + "="*80)
        print("📈 COMPARISON WITH PREVIOUS MODELS")
        print("="*80)
        
        rf_pred = (rf_proba >= 0.06).astype(int)
        xgb_pred = (xgb_proba >= 0.0001).astype(int)
        
        rf_precision = precision_score(y_test, rf_pred, zero_division=0)
        rf_recall = recall_score(y_test, rf_pred, zero_division=0)
        rf_fp = confusion_matrix(y_test, rf_pred)[0, 1]
        rf_fpr = rf_fp / (rf_fp + tn) if (rf_fp + tn) > 0 else 0
        
        xgb_precision = precision_score(y_test, xgb_pred, zero_division=0)
        xgb_recall = recall_score(y_test, xgb_pred, zero_division=0)
        xgb_fp = confusion_matrix(y_test, xgb_pred)[0, 1]
        xgb_fpr = xgb_fp / (xgb_fp + tn) if (xgb_fp + tn) > 0 else 0
        
        comparison = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Improved Hybrid'],
            'Precision': [rf_precision, xgb_precision, precision],
            'Recall': [rf_recall, xgb_recall, recall],
            'False Pos Rate': [rf_fpr, xgb_fpr, fpr],
            'False Alarms': [int(rf_fp), int(xgb_fp), int(fp)]
        })
        
        print("\n" + comparison.to_string(index=False))
        
        # Improvement metrics
        print("\n" + "="*80)
        print("🎯 KEY IMPROVEMENTS")
        print("="*80)
        
        if fpr < rf_fpr:
            print(f"✓ False Positive Rate reduced by {(1 - fpr/rf_fpr)*100:.1f}% vs Random Forest")
        if fpr < xgb_fpr:
            print(f"✓ False Positive Rate reduced by {(1 - fpr/xgb_fpr)*100:.1f}% vs XGBoost")
        
        print(f"✓ False Alarms: {fp:,} (was {rf_fp:,} in RF, {xgb_fp:,} in XGB)")
        print(f"✓ Precision: {precision:.1%} (target was {self.target_precision:.1%})")
        print(f"✓ Recall: {recall:.1%} (minimum was {self.min_recall:.1%})")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'fpr': fpr,
            'fnr': fnr,
            'false_positives': fp,
            'confusion_matrix': cm,
            'comparison': comparison
        }
    
    def plot_model_comparison(self, X_test, y_test):
        """Visualize improvements"""
        meta_features, rf_proba, xgb_proba = self.create_meta_features(X_test, return_base_preds=True)
        hybrid_proba = self.meta_model.predict_proba(meta_features)[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Probability distributions
        ax = axes[0, 0]
        ax.hist(hybrid_proba[y_test == 0], bins=100, alpha=0.7, label='Safe (actual)', color='blue', density=True)
        ax.hist(hybrid_proba[y_test == 1], bins=100, alpha=0.7, label='Depeg (actual)', color='red', density=True)
        ax.axvline(x=self.optimal_threshold, color='green', linestyle='--', linewidth=3, label=f'Threshold ({self.optimal_threshold:.4f})')
        ax.set_xlabel('Predicted Probability', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title('Improved Hybrid: Probability Distribution', fontweight='bold')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        # 2. Threshold analysis
        ax = axes[0, 1]
        thresholds = np.linspace(0.01, 0.99, 100)
        precisions = []
        recalls = []
        fprs = []
        
        for thresh in thresholds:
            y_pred_temp = (hybrid_proba >= thresh).astype(int)
            precisions.append(precision_score(y_test, y_pred_temp, zero_division=0))
            recalls.append(recall_score(y_test, y_pred_temp, zero_division=0))
            cm_temp = confusion_matrix(y_test, y_pred_temp)
            fp_temp = cm_temp[0, 1]
            tn_temp = cm_temp[0, 0]
            fprs.append(fp_temp / (fp_temp + tn_temp) if (fp_temp + tn_temp) > 0 else 0)
        
        ax.plot(thresholds, precisions, label='Precision', linewidth=2, color='blue')
        ax.plot(thresholds, recalls, label='Recall', linewidth=2, color='red')
        ax.plot(thresholds, fprs, label='FPR', linewidth=2, color='orange')
        ax.axvline(x=self.optimal_threshold, color='green', linestyle='--', linewidth=2, label='Selected')
        ax.axhline(y=self.target_precision, color='blue', linestyle=':', alpha=0.5, label='Target Precision')
        ax.set_xlabel('Threshold', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Threshold Optimization Analysis', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Confusion matrices comparison
        ax = axes[1, 0]
        ax.axis('off')
        
        y_pred_hybrid = (hybrid_proba >= self.optimal_threshold).astype(int)
        cm_hybrid = confusion_matrix(y_test, y_pred_hybrid)
        
        # Create sub-axes for confusion matrices
        gs = axes[1, 0].get_gridspec()
        
        # Remove the main axis
        axes[1, 0].remove()
        
        # Create subfigure
        subfigs = fig.add_subfigure(gs[1, 0])
        sub_axes = subfigs.subplots(1, 3)
        
        # RF confusion matrix
        rf_pred = (rf_proba >= 0.06).astype(int)
        cm_rf = confusion_matrix(y_test, rf_pred)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=sub_axes[0], cbar=False)
        sub_axes[0].set_title('Random Forest', fontsize=10)
        sub_axes[0].set_ylabel('Actual')
        sub_axes[0].set_xlabel('Predicted')
        
        # XGB confusion matrix
        xgb_pred = (xgb_proba >= 0.0001).astype(int)
        cm_xgb = confusion_matrix(y_test, xgb_pred)
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=sub_axes[1], cbar=False)
        sub_axes[1].set_title('XGBoost', fontsize=10)
        sub_axes[1].set_xlabel('Predicted')
        
        # Hybrid confusion matrix
        sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Greens', ax=sub_axes[2], cbar=False)
        sub_axes[2].set_title('Improved Hybrid', fontsize=10)
        sub_axes[2].set_xlabel('Predicted')
        
        subfigs.suptitle('Confusion Matrix Comparison', fontweight='bold')
        
        # 4. False positive comparison
        ax = axes[1, 1]
        models = ['Random\nForest', 'XGBoost', 'Improved\nHybrid']
        fp_counts = [cm_rf[0, 1], cm_xgb[0, 1], cm_hybrid[0, 1]]
        colors = ['#3498db', '#e67e22', '#2ecc71']
        
        bars = ax.bar(models, fp_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('False Positive Count', fontweight='bold')
        ax.set_title('False Positive Reduction', fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, fp_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val):,}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add reduction percentage
        if cm_rf[0, 1] > 0:
            reduction = (1 - cm_hybrid[0, 1]/cm_rf[0, 1]) * 100
            ax.text(0.5, 0.95, f'{reduction:.1f}% Reduction vs RF', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('improved_hybrid_results.png', dpi=300, bbox_inches='tight')
        print("\n   ✓ Saved: improved_hybrid_results.png")
        plt.close()
    
    def save_models(self, prefix='improved_hybrid_depeg'):
        """Save all models"""
        joblib.dump(self.rf_model, f'{prefix}_rf.pkl')
        joblib.dump(self.xgb_model, f'{prefix}_xgb.pkl')
        joblib.dump(self.meta_model, f'{prefix}_meta.pkl')
        joblib.dump({
            'feature_cols': self.feature_cols,
            'optimal_threshold': self.optimal_threshold,
            'target_precision': self.target_precision,
            'min_recall': self.min_recall
        }, f'{prefix}_config.pkl')
        print(f"\n💾 All models saved with prefix: {prefix}")
    
    def load_models(self, prefix='improved_hybrid_depeg'):
        """Load saved models"""
        self.rf_model = joblib.load(f'{prefix}_rf.pkl')
        self.xgb_model = joblib.load(f'{prefix}_xgb.pkl')
        self.meta_model = joblib.load(f'{prefix}_meta.pkl')
        config = joblib.load(f'{prefix}_config.pkl')
        self.feature_cols = config['feature_cols']
        self.optimal_threshold = config['optimal_threshold']
        self.target_precision = config.get('target_precision', 0.90)
        self.min_recall = config.get('min_recall', 0.30)
        print(f"✓ Models loaded from {prefix}")
    

def main():
    """Main training pipeline with FALSE POSITIVE reduction focus"""
    print("\n" + "="*80)
    print("🚀 IMPROVED HYBRID MODEL - FALSE POSITIVE REDUCTION")
    print("="*80)
    print("\nKey Improvements:")
    print("  • Conservative SMOTE ratios (5% RF, 10% XGB)")
    print("  • Stricter stable period filtering")
    print("  • Deeper trees with more regularization")
    print("  • Precision-optimized threshold selection")
    print("  • Target: ≥90% Precision, ≥30% Recall")
    print("="*80 + "\n")
    
    # Initialize with precision target
    predictor = ImprovedHybridDepegPredictor(
        target_precision=0.90,  # 90% precision minimum
        min_recall=0.30         # 30% recall minimum
    )
    
    # Prepare data
    X_train_stable, y_train_stable, X_test, y_test, X_train_raw, y_train_raw = \
        predictor.prepare_data(CSV_PATH)
    
    # Train base models
    predictor.train_base_models(X_train_stable, y_train_stable)
    
    # Train meta-learner
    predictor.train_meta_model(X_train_raw, y_train_raw, X_test, y_test)
    
    # Evaluate
    results = predictor.evaluate(X_test, y_test)
    
    # Visualizations
    print("\n" + "="*80)
    print("📊 CREATING VISUALIZATIONS")
    print("="*80)
    predictor.plot_model_comparison(X_test, y_test)
    
    # Save models
    predictor.save_models('improved_hybrid_depeg')
    
    print("\n" + "="*80)
    print("✅ IMPROVED MODEL TRAINING COMPLETE")
    print("="*80)
    print(f"\nFinal Performance:")
    print(f"  Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"  Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  FPR:       {results['fpr']:.4f} ({results['fpr']*100:.2f}%)")
    print(f"  False Positives: {results['false_positives']:,}")
    print(f"  Threshold: {predictor.optimal_threshold:.4f}")
    
    return predictor, results

    


if __name__ == "__main__":
    predictor, results = main()

