#!/usr/bin/env python3
"""
Flipkart Gridlock 2.0 — Traffic Demand Prediction v7
Breakthrough: Historical Demand Profile Embeddings

Instead of relying on the tree to partition spatial bounding boxes, we transform the time-series problem into an embedding problem. 
We explicitly provide the ENTIRE 104-dimensional historical demand profile (all 96 intervals of Day 48 + 8 morning intervals of Day 49) of each geohash as features.
This gives the model perfect historical context for any geohash.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import lightgbm as lgb
import xgboost as xgb
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

# Missing values
df['RoadType'] = df['RoadType'].fillna('Unknown')
df['Weather'] = df['Weather'].fillna('Unknown')
df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())

# Build 104-Dimensional Historical Profiles!
print("Building Historical Demand Profiles...")
d48 = df[(df['is_train'] == 1) & (df['day'] == 48)]
d49_morning = df[(df['is_train'] == 1) & (df['day'] == 49) & (df['time_idx_daily'] < 9)]

d48_pivot = d48.pivot_table(index='geohash', columns='time_idx_daily', values='demand', aggfunc='mean')
d48_pivot.columns = [f'd48_t{c}' for c in d48_pivot.columns]

d49_pivot = d49_morning.pivot_table(index='geohash', columns='time_idx_daily', values='demand', aggfunc='mean')
d49_pivot.columns = [f'd49_t{c}' for c in d49_pivot.columns]

df = df.merge(d48_pivot, on='geohash', how='left')
df = df.merge(d49_pivot, on='geohash', how='left')

# Fill NaNs in profiles with global medians
profile_cols = [f'd48_t{i}' for i in range(96)] + [f'd49_t{i}' for i in range(9)]
global_med = df[df['is_train']==1]['demand'].median()
df[profile_cols] = df[profile_cols].fillna(global_med)

# Encodings
cat_cols = ['geohash', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
for c in cat_cols:
    df[f'{c}_enc'] = LabelEncoder().fit_transform(df[c].astype(str))

feature_cols = [
    'day', 'NumberofLanes', 'Temperature', 'latitude', 'longitude', 
    'hour', 'minute', 'time_idx_daily',
    'geohash_enc', 'RoadType_enc', 'Weather_enc', 'LargeVehicles_enc', 'Landmarks_enc'
] + profile_cols

print(f"Total Features: {len(feature_cols)}")

X_train = df[df['is_train'] == 1][feature_cols].values
y_train = df[df['is_train'] == 1]['demand'].values
X_test = df[df['is_train'] == 0][feature_cols].values

# ============================================================
print("\n" + "=" * 80)
print("STEP 2: TRAIN ENSEMBLE MODELS")
print("=" * 80)

print("1. Training Profile-Aware LightGBM...")
m_lgb = lgb.LGBMRegressor(
    n_estimators=1500, learning_rate=0.03, num_leaves=255, max_depth=-1,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=SEED, verbose=-1
)
m_lgb.fit(X_train, y_train)
pred_lgb = m_lgb.predict(X_test)

print("2. Training Profile-Aware XGBoost...")
m_xgb = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=12,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=SEED
)
m_xgb.fit(X_train, y_train)
pred_xgb = m_xgb.predict(X_test)

print("3. Training Profile-Aware CatBoost...")
m_cb = CatBoostRegressor(
    iterations=1500, learning_rate=0.05, depth=10,
    loss_function='RMSE', random_seed=SEED, verbose=0
)
m_cb.fit(X_train, y_train)
pred_cb = m_cb.predict(X_test)

# ============================================================
print("\n" + "=" * 80)
print("STEP 3: ENSEMBLE AND SAVE")
print("=" * 80)

pred_blend = (pred_lgb * 0.4) + (pred_xgb * 0.3) + (pred_cb * 0.3)
pred_blend = np.clip(pred_blend, 0, 1)

submission = pd.DataFrame({
    'Index': df[df['is_train'] == 0]['Index'].values.astype(int),
    'demand': pred_blend
})

out_path = os.path.join(BASE_DIR, 'submission_v7_profile.csv')
submission.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
print("DONE!")
