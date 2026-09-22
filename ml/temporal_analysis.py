import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple
import json
from collections import defaultdict
import joblib

warnings.filterwarnings('ignore')

class RealModelTemporalEvaluator:
    """
    Temporal analysis using ACTUAL trained hybrid model.
    No simulation - uses real model predictions.
    """
    
    def __init__(self, csv_path='processed_data.csv', model_prefix='improved_hybrid_depeg'):
        """Initialize with dataset and trained model"""
        print("="*80)
        print("🔬 TEMPORAL MODEL EVALUATION - USING REAL TRAINED MODEL")
        print("="*80)
        
        self.csv_path = csv_path
        self.model_prefix = model_prefix
        self.df = None
        self.results_by_period = []
        self.aggregated_metrics = {}
        
        # Load trained models
        self.load_models()
        
    def load_models(self):
        """Load the actual trained hybrid models"""
        print("\n[Loading Models] Loading trained hybrid ensemble...")
        
        try:
            self.rf_model = joblib.load(f'{self.model_prefix}_rf.pkl')
            self.xgb_model = joblib.load(f'{self.model_prefix}_xgb.pkl')
            self.meta_model = joblib.load(f'{self.model_prefix}_meta.pkl')
            
            config = joblib.load(f'{self.model_prefix}_config.pkl')
            self.feature_cols = config['feature_cols']
            self.optimal_threshold = config['optimal_threshold']
            
            print(f"   ✓ Models loaded successfully")
            print(f"   ✓ Threshold: {self.optimal_threshold:.4f}")
            print(f"   ✓ Features: {len(self.feature_cols)}")
            
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: Model files not found!")
            print(f"   Please run 'python improved_hybrid_model.py' first to train the models.")
            raise
        
    def create_meta_features(self, X):
        """Generate meta-features from base model predictions"""
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        
        # Same meta-features as training
        meta_features = np.column_stack([
            rf_proba,
            xgb_proba,
            rf_proba * xgb_proba,
            np.abs(rf_proba - xgb_proba),
            np.minimum(rf_proba, xgb_proba),
            (rf_proba + xgb_proba) / 2,
            (rf_proba > 0.5).astype(int) & (xgb_proba > 0.5).astype(int),
        ])
        
        return meta_features
    
    def predict(self, X):
        """Make predictions using the trained ensemble"""
        # Ensure we have all required features
        missing_cols = set(self.feature_cols) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        
        # Select features in correct order
        X_features = X[self.feature_cols]
        
        # Clean data
        X_features = X_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Get meta-features and predict
        meta_features = self.create_meta_features(X_features)
        proba = self.meta_model.predict_proba(meta_features)[:, 1]
        predictions = (proba >= self.optimal_threshold).astype(int)
        
        return predictions, proba
        
    def load_and_prepare_data(self):
        """Load data and parse timestamps"""
        print("\n[1/6] Loading dataset...")
        
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        
        # Parse timestamps
        if 'open_time' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['open_time'], dayfirst=True, errors='coerce')
        else:
            raise ValueError("'open_time' column not found")
        
        self.df = self.df.dropna(subset=['datetime'])
        self.df = self.df.sort_values('datetime').reset_index(drop=True)
        
        # Get date range
        self.start_date = self.df['datetime'].min()
        self.end_date = self.df['datetime'].max()
        self.total_days = (self.end_date - self.start_date).days
        
        print(f"   Dataset range: {self.start_date} to {self.end_date}")
        print(f"   Total days: {self.total_days}")
        print(f"   Total samples: {len(self.df)}")
        
        if self.total_days < 365:
            print(f"   ⚠️ Warning: Dataset has {self.total_days} days (< 365)")
        
        return self.df
    
    def create_6hour_periods(self):
        """Create 6-hour time windows"""
        print("\n[2/6] Creating 6-hour periods...")
        
        periods = []
        current = self.start_date
        
        max_days = min(365, self.total_days)
        end_analysis = self.start_date + timedelta(days=max_days)
        
        while current < end_analysis:
            period_end = current + timedelta(hours=6)
            periods.append({
                'start': current,
                'end': period_end,
                'period_id': len(periods)
            })
            current = period_end
        
        print(f"   Created {len(periods)} 6-hour periods")
        print(f"   Analysis window: {self.start_date} to {end_analysis}")
        
        self.periods = periods
        return periods
    
    def evaluate_period(self, period_data, period_info):
        """Evaluate model performance for a single 6-hour period using REAL model"""
        if len(period_data) == 0:
            return None
        
        # Prepare features (don't drop target yet)
        X = period_data.copy()
        y = period_data['target_depeg_next_24h']
        
        # Get REAL model predictions
        try:
            y_pred, y_prob = self.predict(X)
        except Exception as e:
            print(f"   Warning: Prediction failed for period {period_info['period_id']}: {e}")
            return None
        
        # Calculate metrics
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, 
            accuracy_score, confusion_matrix, roc_auc_score
        )
        
        # Handle edge cases
        if y.sum() == 0:
            precision = 1.0 if y_pred.sum() == 0 else 0.0
            recall = np.nan
            auc = np.nan
        elif y_pred.sum() == 0:
            precision = np.nan
            recall = 0.0
            auc = np.nan
        else:
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y, y_prob)
            except:
                auc = np.nan
        
        if np.isnan(precision) or np.isnan(recall):
            f1 = np.nan
        else:
            f1 = f1_score(y, y_pred, zero_division=0)
        
        accuracy = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Market conditions
        market_volatility = period_data['close'].std() if 'close' in period_data.columns else 0
        avg_volume = period_data['volume'].mean() if 'volume' in period_data.columns else 0
        avg_peg_dev = period_data['peg_deviation'].mean() if 'peg_deviation' in period_data.columns else 0
        
        results = {
            'period_id': period_info['period_id'],
            'start_time': period_info['start'],
            'end_time': period_info['end'],
            'day_of_year': period_info['start'].timetuple().tm_yday,
            'hour_of_day': period_info['start'].hour,
            'samples': len(period_data),
            'actual_depegs': int(y.sum()),
            'predicted_depegs': int(y_pred.sum()),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'auc': auc,
            'fpr': fpr,
            'fnr': fnr,
            'avg_risk_score': y_prob.mean(),
            'max_risk_score': y_prob.max(),
            'market_volatility': market_volatility,
            'avg_volume': avg_volume,
            'avg_peg_deviation': avg_peg_dev
        }
        
        return results
    
    def run_temporal_analysis(self):
        """Run evaluation across all 6-hour periods using REAL model"""
        print("\n[3/6] Running temporal analysis with REAL MODEL...")
        
        results = []
        
        for i, period in enumerate(self.periods):
            # Get data for this period
            mask = (self.df['datetime'] >= period['start']) & \
                   (self.df['datetime'] < period['end'])
            period_data = self.df[mask]
            
            # Evaluate using REAL model
            result = self.evaluate_period(period_data, period)
            
            if result is not None:
                results.append(result)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"   Processed {i+1}/{len(self.periods)} periods...")
        
        self.results_by_period = results
        print(f"   ✓ Completed {len(results)} period evaluations with REAL MODEL")
        
        return results
    
    def calculate_aggregated_metrics(self):
        """Calculate overall and time-based aggregated metrics"""
        print("\n[4/6] Calculating aggregated metrics...")
        
        df_results = pd.DataFrame(self.results_by_period)
        
        # Remove NaN values for calculation
        df_clean = df_results.dropna(subset=['precision', 'recall', 'f1_score'])
        
        # Overall metrics
        overall = {
            'total_periods': len(df_results),
            'total_samples': df_results['samples'].sum(),
            'total_actual_depegs': df_results['actual_depegs'].sum(),
            'total_predicted_depegs': df_results['predicted_depegs'].sum(),
            'total_tp': df_results['true_positives'].sum(),
            'total_tn': df_results['true_negatives'].sum(),
            'total_fp': df_results['false_positives'].sum(),
            'total_fn': df_results['false_negatives'].sum(),
            'avg_precision': df_clean['precision'].mean(),
            'std_precision': df_clean['precision'].std(),
            'avg_recall': df_clean['recall'].mean(),
            'std_recall': df_clean['recall'].std(),
            'avg_f1': df_clean['f1_score'].mean(),
            'std_f1': df_clean['f1_score'].std(),
            'avg_accuracy': df_results['accuracy'].mean(),
            'avg_fpr': df_results['fpr'].mean(),
            'avg_fnr': df_results['fnr'].mean(),
            'avg_risk_score': df_results['avg_risk_score'].mean(),
        }
        
        # Calculate overall precision/recall from totals
        if overall['total_predicted_depegs'] > 0:
            overall['overall_precision'] = overall['total_tp'] / overall['total_predicted_depegs']
        else:
            overall['overall_precision'] = 0.0
        
        if overall['total_actual_depegs'] > 0:
            overall['overall_recall'] = overall['total_tp'] / overall['total_actual_depegs']
        else:
            overall['overall_recall'] = 0.0
        
        if overall['overall_precision'] + overall['overall_recall'] > 0:
            overall['overall_f1'] = 2 * (overall['overall_precision'] * overall['overall_recall']) / \
                                   (overall['overall_precision'] + overall['overall_recall'])
        else:
            overall['overall_f1'] = 0.0
        
        # By hour of day
        by_hour = df_results.groupby('hour_of_day').agg({
            'precision': 'mean',
            'recall': 'mean',
            'f1_score': 'mean',
            'accuracy': 'mean',
            'actual_depegs': 'sum',
            'predicted_depegs': 'sum',
            'samples': 'sum'
        }).to_dict('index')
        
        # By day of week
        df_results['day_of_week'] = pd.to_datetime(df_results['start_time']).dt.dayofweek
        by_dow = df_results.groupby('day_of_week').agg({
            'precision': 'mean',
            'recall': 'mean',
            'f1_score': 'mean',
            'actual_depegs': 'sum',
            'samples': 'sum'
        }).to_dict('index')
        
        # By month
        df_results['month'] = pd.to_datetime(df_results['start_time']).dt.month
        by_month = df_results.groupby('month').agg({
            'precision': 'mean',
            'recall': 'mean',
            'f1_score': 'mean',
            'actual_depegs': 'sum',
            'samples': 'sum'
        }).to_dict('index')
        
        # Performance stability
        stability = {
            'precision_coefficient_of_variation': (overall['std_precision'] / overall['avg_precision']) 
                                                   if overall['avg_precision'] > 0 else 0,
            'recall_coefficient_of_variation': (overall['std_recall'] / overall['avg_recall']) 
                                               if overall['avg_recall'] > 0 else 0,
            'periods_with_depegs': (df_results['actual_depegs'] > 0).sum(),
            'periods_with_predictions': (df_results['predicted_depegs'] > 0).sum(),
            'periods_with_high_accuracy': (df_results['accuracy'] > 0.95).sum(),
        }
        
        self.aggregated_metrics = {
            'overall': overall,
            'by_hour_of_day': by_hour,
            'by_day_of_week': by_dow,
            'by_month': by_month,
            'stability': stability,
            'df_results': df_results
        }
        
        print("   ✓ Aggregation complete")
        
        return self.aggregated_metrics
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("📊 REAL MODEL TEMPORAL ANALYSIS SUMMARY")
        print("="*80)
        
        overall = self.aggregated_metrics['overall']
        
        print("\n--- OVERALL PERFORMANCE (USING TRAINED MODEL) ---")
        print(f"Total Periods Analyzed:     {overall['total_periods']:,}")
        print(f"Total Samples:              {overall['total_samples']:,}")
        print(f"Total Actual Depegs:        {overall['total_actual_depegs']:,}")
        print(f"Total Predicted Depegs:     {overall['total_predicted_depegs']:,}")
        print(f"\nConfusion Matrix (Aggregated):")
        print(f"  True Positives:           {overall['total_tp']:,}")
        print(f"  True Negatives:           {overall['total_tn']:,}")
        print(f"  False Positives:          {overall['total_fp']:,}")
        print(f"  False Negatives:          {overall['total_fn']:,}")
        
        print(f"\n--- KEY METRICS ---")
        print(f"Overall Precision:          {overall['overall_precision']:.4f} ({overall['overall_precision']*100:.2f}%)")
        print(f"Overall Recall:             {overall['overall_recall']:.4f} ({overall['overall_recall']*100:.2f}%)")
        print(f"Overall F1-Score:           {overall['overall_f1']:.4f}")
        print(f"\nAverage Precision (±std):   {overall['avg_precision']:.4f} ± {overall['std_precision']:.4f}")
        print(f"Average Recall (±std):      {overall['avg_recall']:.4f} ± {overall['std_recall']:.4f}")
        print(f"Average F1-Score (±std):    {overall['avg_f1']:.4f} ± {overall['std_f1']:.4f}")
        print(f"Average Accuracy:           {overall['avg_accuracy']:.4f}")
        
        print(f"\n--- ERROR RATES ---")
        print(f"Average False Positive Rate: {overall['avg_fpr']:.4f} ({overall['avg_fpr']*100:.2f}%)")
        print(f"Average False Negative Rate: {overall['avg_fnr']:.4f} ({overall['avg_fnr']*100:.2f}%)")
        
        print(f"\n--- ALERT ANALYSIS ---")
        alert_rate = overall['total_predicted_depegs'] / overall['total_samples'] * 100
        print(f"Alert Rate:                 {alert_rate:.4f}% of time")
        print(f"Alerts per Period (avg):    {overall['total_predicted_depegs']/overall['total_periods']:.2f}")
        
        # Stability
        stability = self.aggregated_metrics['stability']
        print(f"\n--- TEMPORAL STABILITY ---")
        print(f"Periods with Depegs:        {stability['periods_with_depegs']} "
              f"({stability['periods_with_depegs']/overall['total_periods']*100:.2f}%)")
        print(f"Periods with Predictions:   {stability['periods_with_predictions']} "
              f"({stability['periods_with_predictions']/overall['total_periods']*100:.2f}%)")
        print(f"High Accuracy Periods:      {stability['periods_with_high_accuracy']} "
              f"({stability['periods_with_high_accuracy']/overall['total_periods']*100:.2f}%)")
        print(f"Precision Stability (CV):   {stability['precision_coefficient_of_variation']:.4f}")
        print(f"Recall Stability (CV):      {stability['recall_coefficient_of_variation']:.4f}")
        
        # Model threshold info
        print(f"\n--- MODEL CONFIGURATION ---")
        print(f"Optimal Threshold:          {self.optimal_threshold:.4f}")
        print(f"Model Type:                 Stacked Ensemble (RF + XGB + Meta)")
    
    def create_visualizations(self):
        """Create visualization (reusing the existing logic from original script)"""
        print("\n[5/6] Creating visualizations...")
        
        df_results = self.aggregated_metrics['df_results']
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Performance over time
        ax1 = fig.add_subplot(gs[0, :])
        df_clean = df_results.dropna(subset=['f1_score'])
        ax1.plot(df_clean['period_id'], df_clean['precision'], label='Precision', alpha=0.7, linewidth=1)
        ax1.plot(df_clean['period_id'], df_clean['recall'], label='Recall', alpha=0.7, linewidth=1)
        ax1.plot(df_clean['period_id'], df_clean['f1_score'], label='F1-Score', alpha=0.7, linewidth=2, color='green')
        ax1.axhline(y=self.aggregated_metrics['overall']['avg_f1'], color='red', linestyle='--', label='Avg F1', linewidth=2)
        ax1.set_xlabel('Period ID (6-hour intervals)', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('REAL MODEL: Performance Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # 2. Depegs detected
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.bar(df_results['period_id'], df_results['actual_depegs'], label='Actual', alpha=0.6, color='red', width=1)
        ax2.bar(df_results['period_id'], df_results['predicted_depegs'], label='Predicted', alpha=0.6, color='blue', width=1)
        ax2.set_xlabel('Period ID', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Depeg Detection', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. By hour
        ax3 = fig.add_subplot(gs[1, 1])
        by_hour = self.aggregated_metrics['by_hour_of_day']
        hours = sorted(by_hour.keys())
        f1_by_hour = [by_hour[h]['f1_score'] for h in hours]
        colors = plt.cm.viridis(np.linspace(0, 1, len(hours)))
        bars = ax3.bar(hours, f1_by_hour, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Hour of Day', fontweight='bold')
        ax3.set_ylabel('Average F1-Score', fontweight='bold')
        ax3.set_title('Performance by Time', fontsize=12, fontweight='bold')
        ax3.set_xticks(hours)
        ax3.set_xticklabels([f'{h:02d}:00' for h in hours])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Confusion matrix
        ax4 = fig.add_subplot(gs[1, 2])
        overall = self.aggregated_metrics['overall']
        cm_data = np.array([
            [overall['total_tn'], overall['total_fp']],
            [overall['total_fn'], overall['total_tp']]
        ])
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax4,
                   xticklabels=['Safe', 'Depeg'],
                   yticklabels=['Safe', 'Depeg'])
        ax4.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Actual')
        ax4.set_xlabel('Predicted')
        
        # 5. Risk distribution
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(df_results['avg_risk_score'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax5.axvline(x=self.optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({self.optimal_threshold:.4f})')
        ax5.set_xlabel('Risk Score', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('Risk Score Distribution', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Precision vs Recall
        ax6 = fig.add_subplot(gs[2, 1])
        df_clean = df_results.dropna(subset=['precision', 'recall'])
        scatter = ax6.scatter(df_clean['recall'], df_clean['precision'], c=df_clean['f1_score'], 
                            cmap='viridis', alpha=0.6, edgecolors='black', s=30)
        ax6.set_xlabel('Recall', fontweight='bold')
        ax6.set_ylabel('Precision', fontweight='bold')
        ax6.set_title('Precision vs Recall', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax6, label='F1')
        ax6.grid(True, alpha=0.3)
        
        # 7. Error rates
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.plot(df_results['period_id'], df_results['fpr'], label='FPR', alpha=0.7, color='orange')
        ax7.plot(df_results['period_id'], df_results['fnr'], label='FNR', alpha=0.7, color='red')
        ax7.set_xlabel('Period ID', fontweight='bold')
        ax7.set_ylabel('Rate', fontweight='bold')
        ax7.set_title('Error Rates', fontsize=12, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Summary table
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        summary_data = [
            ['Metric', 'Value'],
            ['Overall Precision', f"{overall['overall_precision']:.1%}"],
            ['Overall Recall', f"{overall['overall_recall']:.1%}"],
            ['Overall F1-Score', f"{overall['overall_f1']:.4f}"],
            ['False Positive Rate', f"{overall['avg_fpr']:.2%}"],
            ['False Negatives', f"{overall['total_fn']:,}"],
            ['Model Threshold', f"{self.optimal_threshold:.4f}"],
            ['Alert Rate', f"{overall['total_predicted_depegs']/overall['total_samples']*100:.2f}%"],
        ]
        
        table = ax8.table(cellText=summary_data, cellLoc='left', loc='center', colWidths=[0.3, 0.7])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        for i in range(2):
            table[(0, i)].set_facecolor('#2ecc71')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(summary_data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        ax8.set_title('Real Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        fig.suptitle('Real Trained Model: Temporal Analysis', fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('real_model_temporal_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved: real_model_temporal_analysis.png")
        plt.close()
    
    def save_results(self):
        """Save results"""
        print("\n[6/6] Saving results...")
        
        df_results = self.aggregated_metrics['df_results']
        df_results.to_csv('real_model_temporal_analysis.csv', index=False)
        print("   ✓ Saved: real_model_temporal_analysis.csv")
        
        metrics_to_save = {
            'overall': self.aggregated_metrics['overall'],
            'by_hour_of_day': {str(k): v for k, v in self.aggregated_metrics['by_hour_of_day'].items()},
            'by_day_of_week': {str(k): v for k, v in self.aggregated_metrics['by_day_of_week'].items()},
            'by_month': {str(k): v for k, v in self.aggregated_metrics['by_month'].items()},
            'stability': self.aggregated_metrics['stability'],
            'model_info': {
                'threshold': self.optimal_threshold,
                'model_type': 'Stacked Ensemble (RF + XGB + Meta)',
                'model_prefix': self.model_prefix
            }
        }
        
        with open('real_model_temporal_summary.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=2, default=str)
        print("   ✓ Saved: real_model_temporal_summary.json")
        
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        self.load_and_prepare_data()
        self.create_6hour_periods()
        self.run_temporal_analysis()
        self.calculate_aggregated_metrics()
        self.print_summary()
        self.create_visualizations()
        self.save_results()
        
        print("\n" + "="*80)
        print("✅ REAL MODEL TEMPORAL ANALYSIS COMPLETE")
        print("="*80)
        print("\nGenerated files:")
        print("  1. real_model_temporal_analysis.png - Performance dashboard")
        print("  2. real_model_temporal_analysis.csv - Detailed results")
        print("  3. real_model_temporal_summary.json - Aggregated metrics")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("🚀 REAL MODEL TEMPORAL ANALYSIS")
    print("="*80)
    print("\nThis script uses your ACTUAL TRAINED hybrid model")
    print("(not simulation) to test performance over 6-hour periods.")
    print("\n" + "="*80 + "\n")
    
    evaluator = RealModelTemporalEvaluator(
        csv_path='processed_data.csv',
        model_prefix='improved_hybrid_depeg'
    )
    evaluator.run_complete_analysis()


if __name__ == "__main__":
    main()