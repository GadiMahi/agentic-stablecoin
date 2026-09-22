import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Make sure these match the filenames you saved earlier
MODEL_PREFIX = 'improved_hybrid_depeg' 

def load_and_analyze(prefix, top_n=20):
    print("="*80)
    print(f"📂 LOADING MODELS FROM: {prefix}...")
    print("="*80)

    try:
        # 1. Load the objects
        rf_model = joblib.load(f'{prefix}_rf.pkl')
        xgb_model = joblib.load(f'{prefix}_xgb.pkl')
        meta_model = joblib.load(f'{prefix}_meta.pkl')
        config = joblib.load(f'{prefix}_config.pkl')
        
        feature_cols = config['feature_cols']
        
        print("✓ Models loaded successfully")
        print(f"✓ Feature count: {len(feature_cols)}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file. {e}")
        return

    # 2. Extract Feature Importance
    # -------------------------------------------------------
    # Random Forest (Gini Importance)
    rf_imp = rf_model.feature_importances_
    
    # XGBoost (Gain/Weight)
    xgb_imp = xgb_model.feature_importances_

    # Create DataFrame
    df_imp = pd.DataFrame({
        'Feature': feature_cols,
        'RF_Raw': rf_imp,
        'XGB_Raw': xgb_imp
    })

    # 3. Normalize & Calculate Hybrid Score
    # -------------------------------------------------------
    # Normalize 0-1 to make them comparable
    df_imp['RF_Norm'] = df_imp['RF_Raw'] / df_imp['RF_Raw'].max()
    df_imp['XGB_Norm'] = df_imp['XGB_Raw'] / df_imp['XGB_Raw'].max()
    
    # Calculate consensus (Hybrid Score)
    df_imp['Hybrid_Score'] = (df_imp['RF_Norm'] + df_imp['XGB_Norm']) / 2
    
    # Sort
    df_imp = df_imp.sort_values('Hybrid_Score', ascending=False).reset_index(drop=True)

    # 4. Print Top Features (Text Report)
    # -------------------------------------------------------
    print("\n" + "="*80)
    print(f"🏆 TOP {top_n} MOST PREDICTIVE FEATURES")
    print("="*80)
    print(f"{'Rank':<5} | {'Feature Name':<40} | {'Hybrid':<8} | {'RF':<8} | {'XGB':<8}")
    print("-" * 85)
    
    for i in range(top_n):
        row = df_imp.iloc[i]
        print(f"{i+1:<5} | {row['Feature']:<40} | {row['Hybrid_Score']:.4f}   | {row['RF_Norm']:.4f}   | {row['XGB_Norm']:.4f}")

    # 5. Visualize (Plots)
    # -------------------------------------------------------
    plt.figure(figsize=(15, 10))
    
    # Plot A: Hybrid Importance
    plt.subplot(2, 1, 1)
    sns.barplot(x='Hybrid_Score', y='Feature', data=df_imp.head(top_n), palette='viridis')
    plt.title(f'Top {top_n} Features (Combined RF + XGB Importance)', fontweight='bold')
    plt.xlabel('Hybrid Score (Normalized)')
    plt.grid(axis='x', alpha=0.3)
    
    # Plot B: Meta-Learner Weights (Decision Logic)
    plt.subplot(2, 1, 2)
    
    # Define meta-feature names (Must match training order)
    meta_features = [
        'RF_Proba', 'XGB_Proba', 'Agreement (RF*XGB)', 
        'Disagreement |RF-XGB|', 'Conservative Min', 
        'Average', 'Double_Confirm'
    ]
    
    if hasattr(meta_model, 'coef_'):
        # Extract weights
        weights = meta_model.coef_[0]
        
        # Check if lengths match (Meta model might have different features if code changed)
        if len(weights) == len(meta_features):
            df_meta = pd.DataFrame({'Signal': meta_features, 'Weight': weights})
            df_meta = df_meta.sort_values('Weight', ascending=False)
            
            # Color logic: Green for positive (trust), Red for negative (suppress)
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_meta['Weight']]
            
            sns.barplot(x='Weight', y='Signal', data=df_meta, palette=colors)
            plt.title('Meta-Learner Logic: Which Model Does It Trust?', fontweight='bold')
            plt.axvline(0, color='black', linewidth=1)
            plt.grid(axis='x', alpha=0.3)
        else:
            plt.text(0.5, 0.5, "Meta-feature mismatch (Code version diff)", 
                     ha='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('saved_model_feature_importance.png', dpi=300)
    print(f"\n✅ Plot saved to: saved_model_feature_importance.png")
    plt.show()

if __name__ == "__main__":
    load_and_analyze(MODEL_PREFIX, top_n=20)