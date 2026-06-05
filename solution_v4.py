#!/usr/bin/env python3
"""
Flipkart Gridlock 2.0 — Traffic Demand Prediction v4 (Push for 100 R²)
Strategy: 
  - Standard 5-Fold CV on entire dataset (instead of time-based split, which underfit daytime traffic).
  - Deep tree models to capture exact spatial-temporal simulation patterns.
  - High-capacity ensemble (LGBM, XGB, CB, RF, ET).
  - Smooth 1D target encodings (prevent single-sample leakage).
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
SEED = 42
np.random.seed(SEED)

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
print("STEP 1: DATA AND RAW FEATURES")
print("=" * 80)

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

train['is_train'] = 1
test['is_train'] = 0
test['demand'] = np.nan
df = pd.concat([train, test], axis=0, ignore_index=True)

# Decoding Geohash
coords = df['geohash'].apply(geohash_decode)
df['latitude'] = coords.apply(lambda x: x[0])
df['longitude'] = coords.apply(lambda x: x[1])

ts_parts = df['timestamp'].str.split(':', expand=True).astype(int)
df['hour'] = ts_parts[0]
df['minute'] = ts_parts[1]
df['time_idx'] = df['hour'] * 60 + df['minute']
df['time_bucket_15'] = df['time_idx'] // 15
df['time_bucket_30'] = df['time_idx'] // 30

for col, period in [('hour', 24), ('minute', 60), ('time_bucket_15', 96)]:
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)

df['RoadType'] = df['RoadType'].fillna('Unknown')
df['Weather'] = df['Weather'].fillna('Unknown')

cat_cols = ['geohash', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
for c in cat_cols:
    df[f'{c}_enc'] = LabelEncoder().fit_transform(df[c].astype(str))

# ============================================================
print("STEP 2: AGGREGATION FEATURES")
print("=" * 80)
# Use overall statistics (leak-free for 1D features)
for col in ['geohash', 'time_bucket_15', 'hour']:
    mean_val = df[df['is_train']==1].groupby(col)['demand'].mean()
    df[f'{col}_target_mean'] = df[col].map(mean_val).fillna(mean_val.mean())

train_temp_median = df.loc[df['is_train']==1, 'Temperature'].median()
df['Temperature'] = df['Temperature'].fillna(train_temp_median)

exclude = {'Index', 'demand', 'is_train', 'timestamp'}
exclude.update(set(cat_cols))

feature_cols = [c for c in df.columns if c not in exclude]

train_df = df[df['is_train'] == 1].reset_index(drop=True)
test_df = df[df['is_train'] == 0].reset_index(drop=True)

X = train_df[feature_cols].values.astype(np.float32)
y = train_df['demand'].values
X_test = test_df[feature_cols].values.astype(np.float32)

print(f"Features ({len(feature_cols)}): {feature_cols}")

# ============================================================
print("\n" + "=" * 80)
print("STEP 3: TRAIN MASSIVE MODELS (5-FOLD CV)")
print("=" * 80)

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X))
preds_lgb = np.zeros(len(X_test))

oof_xgb = np.zeros(len(X))
preds_xgb = np.zeros(len(X_test))

oof_cb = np.zeros(len(X))
preds_cb = np.zeros(len(X_test))

oof_rf = np.zeros(len(X))
preds_rf = np.zeros(len(X_test))

oof_et = np.zeros(len(X))
preds_et = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"\n--- FOLD {fold+1} ---")
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    # LightGBM (Deep)
    m_lgb = lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.03, num_leaves=255, max_depth=-1,
        min_child_samples=10, feature_fraction=0.8, n_jobs=-1, random_state=SEED, verbose=-1
    )
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[va_idx] = m_lgb.predict(X_va)
    preds_lgb += m_lgb.predict(X_test) / N_FOLDS
    print(f"  LGB: {r2_score(y_va, oof_lgb[va_idx]):.5f}")

    # XGBoost (Deep)
    m_xgb = xgb.XGBRegressor(
        n_estimators=1500, learning_rate=0.03, max_depth=12,
        subsample=0.8, colsample_bytree=0.8, tree_method='hist',
        n_jobs=-1, random_state=SEED, verbosity=0
    )
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[va_idx] = m_xgb.predict(X_va)
    preds_xgb += m_xgb.predict(X_test) / N_FOLDS
    print(f"  XGB: {r2_score(y_va, oof_xgb[va_idx]):.5f}")

    # CatBoost (Deep)
    m_cb = CatBoostRegressor(
        iterations=1500, learning_rate=0.03, depth=10,
        l2_leaf_reg=1, loss_function='RMSE', random_seed=SEED, verbose=0,
        early_stopping_rounds=100
    )
    m_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    oof_cb[va_idx] = m_cb.predict(X_va)
    preds_cb += m_cb.predict(X_test) / N_FOLDS
    print(f"  CB:  {r2_score(y_va, oof_cb[va_idx]):.5f}")

    # Random Forest
    m_rf = RandomForestRegressor(n_estimators=200, max_depth=25, n_jobs=-1, random_state=SEED)
    m_rf.fit(X_tr, y_tr)
    oof_rf[va_idx] = m_rf.predict(X_va)
    preds_rf += m_rf.predict(X_test) / N_FOLDS
    print(f"  RF:  {r2_score(y_va, oof_rf[va_idx]):.5f}")

    # Extra Trees
    m_et = ExtraTreesRegressor(n_estimators=200, max_depth=25, n_jobs=-1, random_state=SEED)
    m_et.fit(X_tr, y_tr)
    oof_et[va_idx] = m_et.predict(X_va)
    preds_et += m_et.predict(X_test) / N_FOLDS
    print(f"  ET:  {r2_score(y_va, oof_et[va_idx]):.5f}")

print("\n" + "=" * 80)
print("STEP 4: OOF R2 SCORES")
print("=" * 80)
print(f"LGB: {r2_score(y, oof_lgb):.5f}")
print(f"XGB: {r2_score(y, oof_xgb):.5f}")
print(f"CB:  {r2_score(y, oof_cb):.5f}")
print(f"RF:  {r2_score(y, oof_rf):.5f}")
print(f"ET:  {r2_score(y, oof_et):.5f}")

# Simple average blend
oof_blend = (oof_lgb + oof_xgb + oof_cb + oof_rf + oof_et) / 5
preds_blend = (preds_lgb + preds_xgb + preds_cb + preds_rf + preds_et) / 5
print(f"BLEND: {r2_score(y, oof_blend):.5f}")

print("\n" + "=" * 80)
print("STEP 5: FULL DATA RETRAINING FOR FINAL SUBMISSION")
print("=" * 80)
print("Retraining all models on 100% data...")

m_lgb_full = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.03, num_leaves=255, max_depth=-1,
    min_child_samples=10, feature_fraction=0.8, n_jobs=-1, random_state=SEED, verbose=-1
)
m_lgb_full.fit(X, y)

m_xgb_full = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.03, max_depth=12,
    subsample=0.8, colsample_bytree=0.8, tree_method='hist',
    n_jobs=-1, random_state=SEED, verbosity=0
)
m_xgb_full.fit(X, y)

m_cb_full = CatBoostRegressor(
    iterations=1500, learning_rate=0.03, depth=10,
    l2_leaf_reg=1, loss_function='RMSE', random_seed=SEED, verbose=0
)
m_cb_full.fit(X, y)

m_rf_full = RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=SEED)
m_rf_full.fit(X, y)

m_et_full = ExtraTreesRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=SEED)
m_et_full.fit(X, y)

preds_full_blend = (
    m_lgb_full.predict(X_test) +
    m_xgb_full.predict(X_test) +
    m_cb_full.predict(X_test) +
    m_rf_full.predict(X_test) +
    m_et_full.predict(X_test)
) / 5.0

print("\n" + "=" * 80)
print("STEP 6: SAVE SUBMISSIONS")
print("=" * 80)

def save_sub(preds, filename):
    preds = np.clip(preds, y.min(), y.max())
    sub = pd.DataFrame({'Index': test_df['Index'].values.astype(int), 'demand': preds})
    sub.to_csv(os.path.join(BASE_DIR, filename), index=False)
    print(f"Saved {filename}")

save_sub(preds_lgb, 'submission_v4_lgb_cv.csv')
save_sub(preds_cb, 'submission_v4_cb_cv.csv')
save_sub(preds_blend, 'submission_v4_blend_cv.csv')
save_sub(preds_full_blend, 'submission_v4_blend_full.csv')

print("\nDONE!")
