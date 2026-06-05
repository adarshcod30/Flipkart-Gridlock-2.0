#!/usr/bin/env python3
"""
Flipkart Gridlock 2.0 — Traffic Demand Prediction v6
Ultimate Ensemble with KNN Imputation & Categorical Target Embeddings

This script addresses the irreducible noise ceiling (currently ~91.2%) by:
1. Using KNN Imputation to accurately reconstruct missing Temperatures (instead of median), saving the 3.2% of test rows that were dropping the score.
2. Training massive unregularized XGBoost, LightGBM, and CatBoost models.
3. Ensembling the results to smooth out spatial staircasing.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

df['hour_sin'] = np.sin(2 * np.pi * df['hour']/23.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/23.0)

# Missing values - Category
df['RoadType'] = df['RoadType'].fillna('Unknown')
df['Weather'] = df['Weather'].fillna('Unknown')

# KNN Imputation for Temperature
print("Imputing Temperature NaNs using KNN...")
knn_imputer = KNNImputer(n_neighbors=5)
df['Temperature'] = knn_imputer.fit_transform(df[['latitude', 'longitude', 'time_idx_daily', 'Temperature']])[:, 3]

# Encodings
cat_cols = ['geohash', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
for c in cat_cols:
    df[f'{c}_enc'] = LabelEncoder().fit_transform(df[c].astype(str))

# Safe Target Encoding from Day 48 ONLY
print("Computing Day 48 Target Encodings...")
day48_data = df[(df['is_train'] == 1) & (df['day'] == 48)]
gh_mean = day48_data.groupby('geohash')['demand'].mean()
df['geohash_day48_mean'] = df['geohash'].map(gh_mean).fillna(gh_mean.mean())

feature_cols = [
    'day', 'NumberofLanes', 'Temperature', 'latitude', 'longitude', 
    'hour', 'minute', 'time_idx_daily', 'hour_sin', 'hour_cos',
    'geohash_enc', 'RoadType_enc', 'Weather_enc', 'LargeVehicles_enc', 'Landmarks_enc',
    'geohash_day48_mean'
]

print("Features:", feature_cols)

X_train = df[df['is_train'] == 1][feature_cols].values
y_train = df[df['is_train'] == 1]['demand'].values
X_test = df[df['is_train'] == 0][feature_cols].values

# ============================================================
print("\n" + "=" * 80)
print("STEP 2: TRAIN ENSEMBLE MODELS")
print("=" * 80)

print("1. Training Deep LightGBM...")
m_lgb = lgb.LGBMRegressor(
    n_estimators=1500, learning_rate=0.03, num_leaves=255, max_depth=-1,
    subsample=0.9, colsample_bytree=0.9,
    n_jobs=-1, random_state=SEED, verbose=-1
)
m_lgb.fit(X_train, y_train)
pred_lgb = m_lgb.predict(X_test)

print("2. Training Deep XGBoost...")
m_xgb = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=12,
    subsample=0.9, colsample_bytree=0.9,
    n_jobs=-1, random_state=SEED
)
m_xgb.fit(X_train, y_train)
pred_xgb = m_xgb.predict(X_test)

print("3. Training Deep CatBoost...")
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

out_path = os.path.join(BASE_DIR, 'submission_v6_final.csv')
submission.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
print("DONE!")
