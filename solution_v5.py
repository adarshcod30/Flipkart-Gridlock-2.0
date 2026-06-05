#!/usr/bin/env python3
"""
Flipkart Gridlock 2.0 — Traffic Demand Prediction v5 (Autoregressive Time Series)
Strategy:
  - The dataset is a strict time series. Traffic at t is highly correlated (0.97) with t-1.
  - We use recursive (step-by-step) autoregressive forecasting for the test set.
  - Features: lag_1 (15m), lag_2 (30m), lag_4 (60m), and lag_96 (exact same time yesterday).
  - Train CatBoost and LightGBM on the train set using these exact historical lags.
  - For the test set, predict the first time step (2:15), use those predictions to form the lags for the next step (2:30), and repeat until 13:45.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import CatBoostRegressor
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
SEED = 42

_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_BASE32_MAP = {c: i for i, c in enumerate(_BASE32)}
def geohash_decode(gh):
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    is_lon = True
    for c in gh:
        val = _BASE32_MAP.get(c, 0)
        for bit in [16, 8, 4, 2, 1]:
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if val & bit: lon_range[0] = mid
                else: lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if val & bit: lat_range[0] = mid
                else: lat_range[1] = mid
            is_lon = not is_lon
    return (lat_range[0] + lat_range[1]) / 2, (lon_range[0] + lon_range[1]) / 2

print("=" * 80)
print("STEP 1: LOAD DATA AND BASE FEATURES")
print("=" * 80)

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

train['is_train'] = 1
test['is_train'] = 0
test['demand'] = np.nan
df = pd.concat([train, test], axis=0, ignore_index=True)

# Decode Geohash
coords = df['geohash'].apply(geohash_decode)
df['latitude'] = coords.apply(lambda x: x[0])
df['longitude'] = coords.apply(lambda x: x[1])

# Time features
ts_parts = df['timestamp'].str.split(':', expand=True).astype(int)
df['hour'] = ts_parts[0]
df['minute'] = ts_parts[1]
df['time_idx_daily'] = df['hour'] * 4 + df['minute'] // 15
df['global_time_id'] = df['day'] * 96 + df['time_idx_daily']

# Missing values
df['RoadType'] = df['RoadType'].fillna('Unknown')
df['Weather'] = df['Weather'].fillna('Unknown')
train_temp_median = df.loc[df['is_train']==1, 'Temperature'].median()
df['Temperature'] = df['Temperature'].fillna(train_temp_median)

# Encodings
cat_cols = ['geohash', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
for c in cat_cols:
    df[f'{c}_enc'] = LabelEncoder().fit_transform(df[c].astype(str))

# Create baseline dictionary to fetch true values
demand_dict = df.set_index(['geohash', 'global_time_id'])['demand'].to_dict()

# ============================================================
print("STEP 2: PREPARE TRAINING DATA")
print("=" * 80)

def get_lag(gh, tid, lag):
    return demand_dict.get((gh, tid - lag), np.nan)

print("Calculating lag features for training...")
# For training, we can compute lags vectorized by mapping
df_train = df[df['is_train'] == 1].copy()
df_train['lag_1'] = df_train.apply(lambda r: get_lag(r['geohash'], r['global_time_id'], 1), axis=1)
df_train['lag_2'] = df_train.apply(lambda r: get_lag(r['geohash'], r['global_time_id'], 2), axis=1)
df_train['lag_4'] = df_train.apply(lambda r: get_lag(r['geohash'], r['global_time_id'], 4), axis=1)
df_train['lag_96'] = df_train.apply(lambda r: get_lag(r['geohash'], r['global_time_id'], 96), axis=1)

feature_cols = [
    'day', 'NumberofLanes', 'Temperature', 'latitude', 'longitude', 
    'hour', 'minute', 'time_idx_daily', 
    'geohash_enc', 'RoadType_enc', 'Weather_enc', 'LargeVehicles_enc', 'Landmarks_enc',
    'lag_1', 'lag_2', 'lag_4', 'lag_96'
]

print("Training features:", feature_cols)

X_train = df_train[feature_cols].values
y_train = df_train['demand'].values

print("Training LightGBM Autoregressive Model...")
m_lgb = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.03, num_leaves=127, max_depth=-1,
    n_jobs=-1, random_state=SEED, verbose=-1
)
m_lgb.fit(X_train, y_train)

print("Training CatBoost Autoregressive Model...")
m_cb = CatBoostRegressor(
    iterations=1000, learning_rate=0.03, depth=8,
    loss_function='RMSE', random_seed=SEED, verbose=0
)
m_cb.fit(X_train, y_train)

# ============================================================
print("\n" + "=" * 80)
print("STEP 3: RECURSIVE TEST SET FORECASTING")
print("=" * 80)

test_time_ids = sorted(df[df['is_train'] == 0]['global_time_id'].unique())
print(f"Forecasting exactly {len(test_time_ids)} sequential time steps...")

df_test = df[df['is_train'] == 0].copy()
# We will predict step by step
for t in test_time_ids:
    # Get rows for this time step
    mask = df_test['global_time_id'] == t
    current_rows = df_test[mask]
    
    # Calculate lags using the DYNAMIC demand_dict (which includes previous predictions)
    lags_1 = current_rows['geohash'].apply(lambda gh: demand_dict.get((gh, t - 1), np.nan)).values
    lags_2 = current_rows['geohash'].apply(lambda gh: demand_dict.get((gh, t - 2), np.nan)).values
    lags_4 = current_rows['geohash'].apply(lambda gh: demand_dict.get((gh, t - 4), np.nan)).values
    lags_96 = current_rows['geohash'].apply(lambda gh: demand_dict.get((gh, t - 96), np.nan)).values
    
    # We must handle NaNs for missing history. If lag_1 is missing, we use lag_96 (same time yesterday) as a fallback.
    # If lag_96 is missing, we use the global median (rare).
    global_med = np.nanmedian(y_train)
    
    # Helper to fill NaNs
    def fill_lags(lag_array, fallback):
        out = np.where(pd.isna(lag_array), fallback, lag_array)
        return np.where(pd.isna(out), global_med, out)
        
    lags_96 = fill_lags(lags_96, global_med)
    lags_4 = fill_lags(lags_4, lags_96)
    lags_2 = fill_lags(lags_2, lags_4)
    lags_1 = fill_lags(lags_1, lags_2)
    
    # Build feature matrix
    X_step = current_rows[feature_cols[:-4]].values
    X_step = np.column_stack([X_step, lags_1, lags_2, lags_4, lags_96])
    
    # Predict
    pred_lgb = m_lgb.predict(X_step)
    pred_cb = m_cb.predict(X_step)
    pred_blend = (pred_lgb + pred_cb) / 2.0
    pred_blend = np.clip(pred_blend, 0, 1)  # Demand is [0, 1]
    
    # Save predictions back to dictionary and dataframe
    geohashes = current_rows['geohash'].values
    for gh, p in zip(geohashes, pred_blend):
        demand_dict[(gh, t)] = p
        
    df_test.loc[mask, 'demand'] = pred_blend
    print(f"  Predicted step {t} (Hour {t%96 // 4}:{t%96 % 4 * 15:02d}), mean demand: {np.mean(pred_blend):.4f}")

# ============================================================
print("\n" + "=" * 80)
print("STEP 4: SAVE SUBMISSION")
print("=" * 80)

submission = pd.DataFrame({
    'Index': df_test['Index'].values.astype(int),
    'demand': df_test['demand'].values
})

out_path = os.path.join(BASE_DIR, 'submission_v5_autoregressive.csv')
submission.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
print("DONE!")
