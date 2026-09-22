"""
COMPLETE STABLECOIN DEPEG DETECTION - FEATURE ENGINEERING
Minute-level data version (Fully corrected)
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# FEATURE ENGINEERING CLASS
# ============================================================================

class StablecoinFeatureEngine:
    """
    Complete feature engineering pipeline
    Minute-level version
    """

    def __init__(self):
        # Rolling windows converted to minutes
        # 3h, 6h, 12h, 24h, 48h, 7d, 30d
        self.windows = [
            180,     # 3h
            360,     # 6h
            720,     # 12h
            1440,    # 24h
            2880,    # 48h
            10080,   # 7d
            43200    # 30d
        ]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        print("=" * 80)
        print("STABLECOIN FEATURE ENGINEERING PIPELINE (Minute Data)")
        print("=" * 80)
        print(f"\nInput shape: {df.shape}")

        df = df.sort_values('open_time').reset_index(drop=True)

        print("\n[Step 1/6] Creating basic derived features...")
        df = self._create_basic_features(df)

        print("\n[Step 2/6] Creating rolling window features...")
        df = self._create_rolling_features(df)

        print("\n[Step 3/6] Creating peg deviation features...")
        df = self._create_peg_features(df)

        print("\n[Step 4/6] Creating interaction features...")
        df = self._create_interaction_features(df)

        print("\n[Step 5/6] Creating momentum features...")
        df = self._create_momentum_features(df)

        print("\n[Step 6/6] Creating market dynamics features...")
        df = self._create_market_dynamics(df)

        print("\nFeature engineering complete!")
        print(f"Output shape: {df.shape}")

        return df

    # ------------------------------------------------------------------------

    def _create_basic_features(self, df):

        df['daily_range'] = (df['high'] - df['low']) / (df['open'] + 1e-8)
        df['body_size'] = np.abs(df['close'] - df['open']) / (df['open'] + 1e-8)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-8)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-8)
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

        df['trade_size'] = df['volume'] / (df['trades'] + 1)
        df['taker_buy_ratio'] = df['taker_base'] / (df['volume'] + 1e-8)
        df['taker_sell_ratio'] = df['taker_quote'] / (df['quote_volume'] + 1e-8)
        df['volume_imbalance'] = (df['taker_base'] - df['taker_quote']) / (df['volume'] + 1e-8)

        df['mcap_to_supply_ratio'] = df['market_cap'] / (df['circulating_supply'] + 1e-8)
        df['price_to_mcap_billion'] = df['price'] / (df['market_cap'] / 1e9 + 1e-8)

        print("   Created 11 basic features")
        return df

    # ------------------------------------------------------------------------

    def _create_rolling_features(self, df):

        rolling_features = {}

        for window in self.windows:

            rolling_features[f'close_mean_{window}m'] = df['close'].rolling(window).mean()
            rolling_features[f'close_std_{window}m'] = df['close'].rolling(window).std()
            rolling_features[f'close_min_{window}m'] = df['close'].rolling(window).min()
            rolling_features[f'close_max_{window}m'] = df['close'].rolling(window).max()

            rolling_features[f'volume_mean_{window}m'] = df['volume'].rolling(window).mean()
            rolling_features[f'volume_std_{window}m'] = df['volume'].rolling(window).std()
            rolling_features[f'volume_sum_{window}m'] = df['volume'].rolling(window).sum()

            rolling_features[f'trades_mean_{window}m'] = df['trades'].rolling(window).mean()
            rolling_features[f'trade_size_mean_{window}m'] = df['trade_size'].rolling(window).mean()

            rolling_features[f'market_cap_mean_{window}m'] = df['market_cap'].rolling(window).mean()
            rolling_features[f'market_cap_std_{window}m'] = df['market_cap'].rolling(window).std()
            rolling_features[f'circulating_supply_mean_{window}m'] = df['circulating_supply'].rolling(window).mean()

        df = pd.concat([df, pd.DataFrame(rolling_features, index=df.index)], axis=1)

        print(f"   Created {len(rolling_features)} rolling features")
        return df

    # ------------------------------------------------------------------------

    def _create_peg_features(self, df):

        df['peg_deviation'] = np.abs(df['close'] - 1.0)
        df['peg_deviation_pct'] = df['peg_deviation'] * 100
        df['peg_direction'] = df['close'] - 1.0
        df['above_peg'] = (df['close'] > 1.0).astype(int)
        df['below_peg'] = (df['close'] < 1.0).astype(int)

        peg_rolling = {}
        for window in self.windows:
            peg_rolling[f'peg_deviation_mean_{window}m'] = df['peg_deviation'].rolling(window).mean()
            peg_rolling[f'peg_deviation_max_{window}m'] = df['peg_deviation'].rolling(window).max()
            peg_rolling[f'peg_deviation_std_{window}m'] = df['peg_deviation'].rolling(window).std()

        df = pd.concat([df, pd.DataFrame(peg_rolling, index=df.index)], axis=1)

        # Stress indicators (converted properly)
        df['peg_stress_1pct_3h'] = (df['peg_deviation'] > 0.01).rolling(180).sum()
        df['peg_stress_1pct_24h'] = (df['peg_deviation'] > 0.01).rolling(1440).sum()
        df['peg_stress_2pct_3h'] = (df['peg_deviation'] > 0.02).rolling(180).sum()
        df['peg_stress_2pct_24h'] = (df['peg_deviation'] > 0.02).rolling(1440).sum()

        print("   Peg features created")
        return df

    # ------------------------------------------------------------------------

    def _create_interaction_features(self, df):

        df['volume_vs_3h'] = df['volume'] / (df['volume_mean_180m'] + 1e-8)
        df['volume_vs_24h'] = df['volume'] / (df['volume_mean_1440m'] + 1e-8)
        df['volume_vs_7d'] = df['volume'] / (df['volume_mean_10080m'] + 1e-8)
        df['volume_vs_30d'] = df['volume'] / (df['volume_mean_43200m'] + 1e-8)

        df['trade_size_vs_24h'] = df['trade_size'] / (df['trade_size_mean_1440m'] + 1e-8)
        df['trade_size_vs_7d'] = df['trade_size'] / (df['trade_size_mean_10080m'] + 1e-8)

        df['mcap_to_volume_24h'] = df['market_cap'] / (df['volume_sum_1440m'] + 1e-8)
        df['mcap_to_volume_7d'] = df['market_cap'] / (df['volume_sum_10080m'] + 1e-8)

        print("   Interaction features created")
        return df

    # ------------------------------------------------------------------------

    def _create_momentum_features(self, df):

        df['price_accel_1h_24h'] = df['percent_change_1h'] - df['percent_change_24h']
        df['price_accel_24h_7d'] = df['percent_change_24h'] - df['percent_change_7d']
        df['price_accel_7d_30d'] = df['percent_change_7d'] - df['percent_change_30d']

        print("   Momentum features created")
        return df

    # ------------------------------------------------------------------------

    def _create_market_dynamics(self, df):

        df['volume_spike_peg_stress'] = ((df['volume_vs_24h'] > 2.0) & (df['peg_deviation'] > 0.01)).astype(int)
        df['extreme_volume_spike'] = (df['volume_vs_24h'] > 3.0).astype(int)
        df['large_trade_anomaly'] = (df['trade_size_vs_24h'] > 2.0).astype(int)

        print("   Market dynamics features created")
        return df


# ============================================================================
# DYNAMIC TARGET (Minute Version)
# ============================================================================

def create_target_variable(df: pd.DataFrame, 
                          lookahead_hours: int = 24,
                          alpha: float = 1/3):

    print("=" * 80)
    print("CREATING DYNAMIC DEPEG TARGET (Carey 2023)")
    print("=" * 80)

    window_30d = 30 * 24 * 60  # 43,200 minutes
    df['volume_30d'] = df['volume'].rolling(window_30d).sum()

    df['Thresh_D'] = 1 - (10 / (df['volume_30d'] ** alpha + 1e-8))
    df['Thresh_U'] = 1 + (10 / (df['volume_30d'] ** alpha + 1e-8))

    lookahead_minutes = lookahead_hours * 60

    future_low = df['low'].shift(-1).rolling(lookahead_minutes).min()
    future_high = df['high'].shift(-1).rolling(lookahead_minutes).max()

    target_col = f'target_depeg_next_{lookahead_hours}h'

    df[target_col] = (
        (future_low <= df['Thresh_D']) |
        (future_high >= df['Thresh_U'])
    ).astype(int)

    print(f"Total depeg events: {df[target_col].sum()}")
    print(df[target_col].value_counts())

    return df, target_col


# ============================================================================
# RUN PIPELINE (UNCHANGED BEHAVIOR)
# ============================================================================

def run_complete_pipeline(input_file: str,
                          output_file: str = None,
                          lookahead_hours: int = 24):

    print("=" * 80)
    print("STABLECOIN DEPEG DETECTION - COMPLETE PIPELINE")
    print("=" * 80)

    df = pd.read_csv(input_file)

    engine = StablecoinFeatureEngine()
    df = engine.fit_transform(df)

    df, target_col = create_target_variable(df, lookahead_hours)

    if output_file:
        df.to_csv(output_file, index=False)

    df = df.dropna()

    feature_cols = [col for col in df.columns if col not in ['open_time', 'close_time', target_col, 'ignore']]

    train_end = int(len(df) * 0.6)
    val_end = int(len(df) * 0.8)

    X_train = df.iloc[:train_end][feature_cols]
    y_train = df.iloc[:train_end][target_col]

    X_val = df.iloc[train_end:val_end][feature_cols]
    y_val = df.iloc[train_end:val_end][target_col]

    X_test = df.iloc[val_end:][feature_cols]
    y_test = df.iloc[val_end:][target_col]

    print("Pipeline complete")
    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols