#!/usr/bin/env python3
"""
Flipkart Gridlock 2.0 — Traffic Demand Prediction v3
KEY INSIGHT: Train has day48 (full 24h) + day49 (0:00-2:00).
             Test is day49 (2:15-13:45). Zero timestamp overlap.
             Geohash demand correlation between days = 0.85.
             
Strategy: Learn location-time patterns from day48's full coverage,
          use day49 early hours to calibrate, and predict daytime day49.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
SEED = 42
np.random.seed(SEED)

# ============================================================
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
print("STEP 1: LOADING DATA")
print("=" * 80)

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
print("\n" + "=" * 80)
print("STEP 2: BASE FEATURES")
print("=" * 80)

train['is_train'] = 1
test['is_train'] = 0
test['demand'] = np.nan
df = pd.concat([train, test], axis=0, ignore_index=True)

# Geohash
coords = df['geohash'].apply(geohash_decode)
df['latitude'] = coords.apply(lambda x: x[0])
df['longitude'] = coords.apply(lambda x: x[1])
for p in [2, 3, 4, 5]:
    df[f'geohash_p{p}'] = df['geohash'].str[:p]

df['lat_sin'] = np.sin(np.radians(df['latitude']))
df['lat_cos'] = np.cos(np.radians(df['latitude']))
df['lon_sin'] = np.sin(np.radians(df['longitude']))
df['lon_cos'] = np.cos(np.radians(df['longitude']))
lat_c, lon_c = df['latitude'].mean(), df['longitude'].mean()
df['dist_from_center'] = np.sqrt((df['latitude'] - lat_c)**2 + (df['longitude'] - lon_c)**2)

# Temporal
ts_parts = df['timestamp'].str.split(':', expand=True).astype(int)
df['hour'] = ts_parts[0]
df['minute'] = ts_parts[1]
df['total_minutes'] = df['hour'] * 60 + df['minute']
df['time_bucket_15'] = df['total_minutes'] // 15
df['time_bucket_30'] = df['total_minutes'] // 30

for col, period in [('hour', 24), ('minute', 60), ('time_bucket_15', 96), ('time_bucket_30', 48)]:
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)

df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
df['is_peak'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)

df['day_of_week'] = df['day'] % 7
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['part_of_day'] = pd.cut(df['hour'], bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]).astype(int)

# Road / Infrastructure
df['LargeVehicles_bin'] = (df['LargeVehicles'] == 'Allowed').astype(int)
df['Landmarks_bin'] = (df['Landmarks'] == 'Yes').astype(int)
df['road_capacity'] = df['NumberofLanes'] * (1 + 0.5 * df['LargeVehicles_bin'])
df['RoadType'] = df['RoadType'].fillna('Unknown')
df['Weather'] = df['Weather'].fillna('Unknown')

# Temperature
train_temp_median = df.loc[df['is_train']==1, 'Temperature'].median()
df['Temperature'] = df['Temperature'].fillna(train_temp_median)

print(f"  Base features done.")

# ============================================================
print("\n" + "=" * 80)
print("STEP 3: DAY-48-ONLY GROUP STATISTICS (KEY STRATEGY)")
print("=" * 80)
print("  Using ONLY day 48 data (full 24h) for group stats.")
print("  This prevents leakage from day 49 train → day 49 test overlap risk.")

day48_data = df[(df['is_train'] == 1) & (df['day'] == 48)]
global_mean = day48_data['demand'].mean()
global_std = day48_data['demand'].std()
global_median = day48_data['demand'].median()

# Interaction keys
for name, cols in [
    ('gh', ['geohash']),
    ('ghp5', ['geohash_p5']),
    ('ghp4', ['geohash_p4']),
    ('ghp3', ['geohash_p3']),
    ('gh_hour', ['geohash', 'hour']),
    ('gh_tb15', ['geohash', 'time_bucket_15']),
    ('gh_tb30', ['geohash', 'time_bucket_30']),
    ('gh_part', ['geohash', 'part_of_day']),
    ('ghp4_hour', ['geohash_p4', 'hour']),
    ('ghp4_part', ['geohash_p4', 'part_of_day']),
    ('ghp3_hour', ['geohash_p3', 'hour']),
    ('tb15', ['time_bucket_15']),
    ('tb30', ['time_bucket_30']),
    ('hour_grp', ['hour']),
    ('part_grp', ['part_of_day']),
    ('road', ['RoadType']),
    ('weather', ['Weather']),
    ('road_hour', ['RoadType', 'hour']),
    ('road_part', ['RoadType', 'part_of_day']),
    ('weather_part', ['Weather', 'part_of_day']),
    ('gh_road', ['geohash', 'RoadType']),
    ('gh_weather', ['geohash', 'Weather']),
    ('lanes', ['NumberofLanes']),
    ('road_lanes', ['RoadType', 'NumberofLanes']),
    ('road_lanes_hour', ['RoadType', 'NumberofLanes', 'hour']),
]:
    key_name = '_'.join([str(c) for c in cols])
    grp = day48_data.groupby(cols)['demand']
    stats = grp.agg(['mean', 'std', 'median', 'min', 'max', 'count']).reset_index()
    stats.columns = cols + [f'{name}_mean', f'{name}_std', f'{name}_median',
                            f'{name}_min', f'{name}_max', f'{name}_count']
    
    # Bayesian smoothing based on group size
    n = stats[f'{name}_count']
    # Adaptive alpha: bigger groups need less smoothing
    alpha = max(5, min(100, int(day48_data.shape[0] / max(len(stats), 1) * 0.5)))
    stats[f'{name}_smean'] = (n * stats[f'{name}_mean'] + alpha * global_mean) / (n + alpha)
    stats[f'{name}_sstd'] = (n * stats[f'{name}_std'].fillna(0) + alpha * global_std) / (n + alpha)
    stats[f'{name}_range'] = stats[f'{name}_max'] - stats[f'{name}_min']
    
    # Keep smoothed + count + range
    keep = cols + [f'{name}_smean', f'{name}_sstd', f'{name}_count',
                   f'{name}_min', f'{name}_max', f'{name}_range', f'{name}_median']
    stats = stats[keep]
    
    before = df.shape[1]
    df = df.merge(stats, on=cols, how='left')
    
    # Fill missing
    for c in stats.columns:
        if c not in cols and c in df.columns:
            fill_val = global_mean if 'mean' in c or 'median' in c else (global_std if 'std' in c else 0)
            df[c] = df[c].fillna(fill_val)
    
    print(f"  {name}: {len(stats)} groups, alpha={alpha}, +{df.shape[1]-before} cols")

print(f"\n  Total columns: {df.shape[1]}")

# ============================================================
print("\n" + "=" * 80)
print("STEP 4: DEMAND RATIOS AND CROSS-TIME FEATURES")
print("=" * 80)

# Ratio of group mean to global mean (multiplicative factor)
for stat_col in [c for c in df.columns if c.endswith('_smean')]:
    df[f'{stat_col}_ratio'] = df[stat_col] / (global_mean + 1e-8)

# Geohash demand percentile rank from day 48
gh_rank = day48_data.groupby('geohash')['demand'].mean().rank(pct=True)
df['gh_demand_pctrank'] = df['geohash'].map(gh_rank).fillna(0.5)

# Hour demand percentile rank from day 48
hr_rank = day48_data.groupby('hour')['demand'].mean().rank(pct=True)
df['hour_demand_pctrank'] = df['hour'].map(hr_rank).fillna(0.5)

print("  Ratio and rank features created.")

# ============================================================
print("\n" + "=" * 80)
print("STEP 5: FREQUENCY ENCODING")
print("=" * 80)

for col in ['geohash', 'geohash_p4', 'geohash_p3', 'RoadType', 'Weather']:
    freq = df[col].value_counts().to_dict()
    df[f'{col}_freq'] = df[col].map(freq)

# ============================================================
print("\n" + "=" * 80)
print("STEP 6: LABEL ENCODING + FINALIZE")
print("=" * 80)

cat_cols = ['geohash', 'geohash_p2', 'geohash_p3', 'geohash_p4', 'geohash_p5',
            'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = df[col].fillna('Unknown').astype(str)
    df[f'{col}_enc'] = le.fit_transform(df[col])

exclude = {'Index', 'demand', 'is_train', 'timestamp', 'day'}
exclude.update(set(cat_cols))

feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

for c in feature_cols:
    if df[c].isnull().any():
        med = df.loc[df['is_train']==1, c].median()
        df[c] = df[c].fillna(med if not np.isnan(med) else 0)

# Remove infinite values
for c in feature_cols:
    df[c] = df[c].replace([np.inf, -np.inf], 0)

print(f"  Features: {len(feature_cols)}")

train_df = df[df['is_train'] == 1].reset_index(drop=True)
test_df = df[df['is_train'] == 0].reset_index(drop=True)

X = train_df[feature_cols].values.astype(np.float32)
y = train_df['demand'].values
X_test = test_df[feature_cols].values.astype(np.float32)

print(f"  X: {X.shape}, X_test: {X_test.shape}")

# ============================================================
print("\n" + "=" * 80)
print("STEP 7: VALIDATION STRATEGY")
print("=" * 80)

# Time-based: train day48, validate day49 (most realistic)
day_col = train_df['day'].values
d48_idx = np.where(day_col == 48)[0]
d49_idx = np.where(day_col == 49)[0]

print(f"  Time split: train day48 ({len(d48_idx)}), val day49 ({len(d49_idx)})")

# Quick test
m_quick = lgb.LGBMRegressor(
    objective='regression', metric='rmse', n_estimators=3000,
    learning_rate=0.03, num_leaves=63, max_depth=7,
    feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5,
    min_child_samples=30, reg_alpha=0.1, reg_lambda=1.0,
    verbose=-1, random_state=SEED, n_jobs=-1
)
m_quick.fit(X[d48_idx], y[d48_idx], eval_set=[(X[d49_idx], y[d49_idx])],
           callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
pred_tb = m_quick.predict(X[d49_idx])
tb_r2 = r2_score(y[d49_idx], pred_tb)
print(f"  Time-based LGB R² (day48→day49): {tb_r2:.6f} ({max(0,tb_r2*100):.2f}/100)")

# ============================================================
print("\n" + "=" * 80)
print("STEP 8: OPTUNA TUNING (TIME-BASED OBJECTIVE)")
print("=" * 80)

def optuna_lgb(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse', 'n_estimators': 3000,
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('nl', 31, 127),
        'max_depth': trial.suggest_int('md', 5, 10),
        'min_child_samples': trial.suggest_int('mcs', 10, 100),
        'feature_fraction': trial.suggest_float('ff', 0.4, 0.9),
        'bagging_fraction': trial.suggest_float('bf', 0.4, 0.9),
        'bagging_freq': trial.suggest_int('bfq', 1, 7),
        'reg_alpha': trial.suggest_float('ra', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('rl', 1e-4, 10.0, log=True),
        'min_split_gain': trial.suggest_float('msg', 0.0, 1.0),
        'verbose': -1, 'random_state': SEED, 'n_jobs': -1,
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X[d48_idx], y[d48_idx], eval_set=[(X[d49_idx], y[d49_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return r2_score(y[d49_idx], m.predict(X[d49_idx]))

print("Running 80 Optuna trials (time-based)...")
study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(optuna_lgb, n_trials=80, show_progress_bar=False)
print(f"  Best time-based R²: {study.best_value:.6f}")

# Also tune XGBoost
def optuna_xgb(trial):
    params = {
        'objective': 'reg:squarederror', 'tree_method': 'hist', 'n_estimators': 3000,
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('md', 4, 10),
        'subsample': trial.suggest_float('ss', 0.4, 0.9),
        'colsample_bytree': trial.suggest_float('cs', 0.4, 0.9),
        'min_child_weight': trial.suggest_int('mcw', 5, 100),
        'reg_alpha': trial.suggest_float('ra', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('rl', 1e-4, 10.0, log=True),
        'random_state': SEED, 'n_jobs': -1, 'verbosity': 0,
    }
    m = xgb.XGBRegressor(**params)
    m.fit(X[d48_idx], y[d48_idx], eval_set=[(X[d49_idx], y[d49_idx])], verbose=False)
    return r2_score(y[d49_idx], m.predict(X[d49_idx]))

print("\nRunning 60 Optuna trials for XGBoost (time-based)...")
study_xgb = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=SEED+1))
study_xgb.optimize(optuna_xgb, n_trials=60, show_progress_bar=False)
print(f"  Best time-based XGB R²: {study_xgb.best_value:.6f}")

# Also tune CatBoost
def optuna_cb(trial):
    m = CatBoostRegressor(
        iterations=2000,
        learning_rate=trial.suggest_float('lr', 0.01, 0.15, log=True),
        depth=trial.suggest_int('depth', 4, 10),
        l2_leaf_reg=trial.suggest_float('l2', 0.1, 20.0, log=True),
        min_data_in_leaf=trial.suggest_int('mdl', 5, 100),
        loss_function='RMSE', random_seed=SEED, verbose=0,
        use_best_model=True, early_stopping_rounds=100,
    )
    m.fit(X[d48_idx], y[d48_idx], eval_set=(X[d49_idx], y[d49_idx]), verbose=0)
    return r2_score(y[d49_idx], m.predict(X[d49_idx]))

print("\nRunning 40 Optuna trials for CatBoost (time-based)...")
study_cb = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED+2))
study_cb.optimize(optuna_cb, n_trials=40, show_progress_bar=False)
print(f"  Best time-based CB R²: {study_cb.best_value:.6f}")

# ============================================================
print("\n" + "=" * 80)
print("STEP 9: TRAIN FINAL MODELS ON ALL DATA")
print("=" * 80)

# LightGBM with tuned params
lgb_best = study.best_params
lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'n_estimators': 3000,
    'learning_rate': lgb_best['lr'], 'num_leaves': lgb_best['nl'],
    'max_depth': lgb_best['md'], 'min_child_samples': lgb_best['mcs'],
    'feature_fraction': lgb_best['ff'], 'bagging_fraction': lgb_best['bf'],
    'bagging_freq': lgb_best['bfq'], 'reg_alpha': lgb_best['ra'],
    'reg_lambda': lgb_best['rl'], 'min_split_gain': lgb_best['msg'],
    'verbose': -1, 'random_state': SEED, 'n_jobs': -1,
}

# XGBoost with tuned params
xgb_best = study_xgb.best_params
xgb_params = {
    'objective': 'reg:squarederror', 'tree_method': 'hist', 'n_estimators': 3000,
    'learning_rate': xgb_best['lr'], 'max_depth': xgb_best['md'],
    'subsample': xgb_best['ss'], 'colsample_bytree': xgb_best['cs'],
    'min_child_weight': xgb_best['mcw'],
    'reg_alpha': xgb_best['ra'], 'reg_lambda': xgb_best['rl'],
    'random_state': SEED, 'n_jobs': -1, 'verbosity': 0,
}

# 5-Fold CV for OOF predictions + test predictions
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print("\n--- Tuned LightGBM (5-fold CV) ---")
oof_lgb = np.zeros(len(X))
preds_lgb = np.zeros(len(X_test))
for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
          callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    oof_lgb[va_idx] = m.predict(X[va_idx])
    preds_lgb += m.predict(X_test) / N_FOLDS
    print(f"  Fold {fold+1}: R²={r2_score(y[va_idx], oof_lgb[va_idx]):.6f}")
print(f"  LGB OOF R²={r2_score(y, oof_lgb):.6f}")

lgb_importance = pd.DataFrame({'feature': feature_cols, 'importance': m.feature_importances_}).sort_values('importance', ascending=False)

print("\n--- Tuned XGBoost (5-fold CV) ---")
oof_xgb = np.zeros(len(X))
preds_xgb = np.zeros(len(X_test))
for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
    oof_xgb[va_idx] = m.predict(X[va_idx])
    preds_xgb += m.predict(X_test) / N_FOLDS
    print(f"  Fold {fold+1}: R²={r2_score(y[va_idx], oof_xgb[va_idx]):.6f}")
print(f"  XGB OOF R²={r2_score(y, oof_xgb):.6f}")

print("\n--- Tuned CatBoost (5-fold CV) ---")
cb_best = study_cb.best_params
oof_cb = np.zeros(len(X))
preds_cb = np.zeros(len(X_test))
for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    m = CatBoostRegressor(
        iterations=2000, learning_rate=cb_best['lr'], depth=cb_best['depth'],
        l2_leaf_reg=cb_best['l2'], min_data_in_leaf=cb_best['mdl'],
        loss_function='RMSE', random_seed=SEED, verbose=0,
        use_best_model=True, early_stopping_rounds=150,
    )
    m.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
    oof_cb[va_idx] = m.predict(X[va_idx])
    preds_cb += m.predict(X_test) / N_FOLDS
    print(f"  Fold {fold+1}: R²={r2_score(y[va_idx], oof_cb[va_idx]):.6f}")
print(f"  CB OOF R²={r2_score(y, oof_cb):.6f}")

# Also train full-data models
print("\n--- Full data models ---")
m_full_lgb = lgb.LGBMRegressor(**lgb_params)
m_full_lgb.set_params(n_estimators=m_quick.best_iteration_ if hasattr(m_quick, 'best_iteration_') else 1200)
m_full_lgb.fit(X, y)
full_lgb = m_full_lgb.predict(X_test)

m_full_xgb = xgb.XGBRegressor(**xgb_params)
m_full_xgb.set_params(n_estimators=1200)
m_full_xgb.fit(X, y, verbose=False)
full_xgb = m_full_xgb.predict(X_test)

m_full_cb = CatBoostRegressor(
    iterations=2000, learning_rate=cb_best['lr'], depth=cb_best['depth'],
    l2_leaf_reg=cb_best['l2'], min_data_in_leaf=cb_best['mdl'],
    loss_function='RMSE', random_seed=SEED, verbose=0,
)
m_full_cb.fit(X, y)
full_cb = m_full_cb.predict(X_test)

print("  Full-data models trained.")

# ============================================================
print("\n" + "=" * 80)
print("STEP 10: ENSEMBLE (TIME-BASED WEIGHT OPTIMIZATION)")
print("=" * 80)

# Use day49 validation for weight optimization (most realistic)
# First get day49 predictions from day48-only models
m_tb_lgb = lgb.LGBMRegressor(**lgb_params)
m_tb_lgb.fit(X[d48_idx], y[d48_idx], eval_set=[(X[d49_idx], y[d49_idx])],
             callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
p_tb_lgb = m_tb_lgb.predict(X[d49_idx])

m_tb_xgb = xgb.XGBRegressor(**xgb_params)
m_tb_xgb.fit(X[d48_idx], y[d48_idx], eval_set=[(X[d49_idx], y[d49_idx])], verbose=False)
p_tb_xgb = m_tb_xgb.predict(X[d49_idx])

m_tb_cb = CatBoostRegressor(
    iterations=2000, learning_rate=cb_best['lr'], depth=cb_best['depth'],
    l2_leaf_reg=cb_best['l2'], min_data_in_leaf=cb_best['mdl'],
    loss_function='RMSE', random_seed=SEED, verbose=0,
    use_best_model=True, early_stopping_rounds=150,
)
m_tb_cb.fit(X[d48_idx], y[d48_idx], eval_set=(X[d49_idx], y[d49_idx]), verbose=0)
p_tb_cb = m_tb_cb.predict(X[d49_idx])

tb_stack = np.column_stack([p_tb_lgb, p_tb_xgb, p_tb_cb])

print(f"  Time-based individual R²:")
print(f"    LGB: {r2_score(y[d49_idx], p_tb_lgb):.6f}")
print(f"    XGB: {r2_score(y[d49_idx], p_tb_xgb):.6f}")
print(f"    CB:  {r2_score(y[d49_idx], p_tb_cb):.6f}")

def neg_r2_tb(w):
    w = np.abs(w)
    w = w / w.sum()
    return -r2_score(y[d49_idx], tb_stack @ w)

# Grid search
best_r2 = -999
best_w = np.array([1/3, 1/3, 1/3])
for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0: continue
        w = np.array([w1, w2, w3])
        r2 = r2_score(y[d49_idx], tb_stack @ w)
        if r2 > best_r2:
            best_r2 = r2
            best_w = w.copy()

result = minimize(neg_r2_tb, best_w, method='Nelder-Mead',
                  options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
opt_w = np.abs(result.x)
opt_w = opt_w / opt_w.sum()

opt_r2_tb = r2_score(y[d49_idx], tb_stack @ opt_w)
eq_r2_tb = r2_score(y[d49_idx], tb_stack.mean(axis=1))

print(f"\n  Time-based ensemble:")
print(f"    Optimized: LGB={opt_w[0]:.3f}, XGB={opt_w[1]:.3f}, CB={opt_w[2]:.3f} → R²={opt_r2_tb:.6f}")
print(f"    Equal weight → R²={eq_r2_tb:.6f}")

# Generate multiple submission variants
# 1. CV-blend with optimized weights
cv_blend = preds_lgb * opt_w[0] + preds_xgb * opt_w[1] + preds_cb * opt_w[2]

# 2. Full-data blend with optimized weights
full_blend = full_lgb * opt_w[0] + full_xgb * opt_w[1] + full_cb * opt_w[2]

# 3. Mix of CV and full-data
mix_blend = 0.4 * cv_blend + 0.6 * full_blend

# ============================================================
print("\n" + "=" * 80)
print("STEP 11: SAVE SUBMISSIONS")
print("=" * 80)

demand_min, demand_max = y.min(), y.max()

variants = {
    'submission.csv': cv_blend,
    'submission_v2.csv': full_blend,
    'submission_v3.csv': mix_blend,
}

for name, preds in variants.items():
    clipped = np.clip(preds, demand_min, demand_max)
    sub = pd.DataFrame({'Index': test_df['Index'].values.astype(int), 'demand': clipped})
    assert sub.shape == (41778, 2)
    assert not sub.isnull().any().any()
    sub.to_csv(os.path.join(BASE_DIR, name), index=False)
    print(f"  {name}: demand [{clipped.min():.6f}, {clipped.max():.6f}], mean={clipped.mean():.6f}")

# ============================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"""
  ┌────────────────────────────────┬──────────┬───────────┐
  │ Model                          │  R²      │ Score/100 │
  ├────────────────────────────────┼──────────┼───────────┤
  │ LGB OOF (5-fold)               │ {r2_score(y, oof_lgb):.6f} │  {max(0,r2_score(y, oof_lgb)*100):.2f}   │
  │ XGB OOF (5-fold)               │ {r2_score(y, oof_xgb):.6f} │  {max(0,r2_score(y, oof_xgb)*100):.2f}   │
  │ CB OOF (5-fold)                │ {r2_score(y, oof_cb):.6f} │  {max(0,r2_score(y, oof_cb)*100):.2f}   │
  ├────────────────────────────────┼──────────┼───────────┤
  │ LGB time-based (d48→d49)       │ {r2_score(y[d49_idx], p_tb_lgb):.6f} │  {max(0,r2_score(y[d49_idx], p_tb_lgb)*100):.2f}   │
  │ XGB time-based (d48→d49)       │ {r2_score(y[d49_idx], p_tb_xgb):.6f} │  {max(0,r2_score(y[d49_idx], p_tb_xgb)*100):.2f}   │
  │ CB time-based (d48→d49)        │ {r2_score(y[d49_idx], p_tb_cb):.6f} │  {max(0,r2_score(y[d49_idx], p_tb_cb)*100):.2f}   │
  │ Ensemble time-based            │ {opt_r2_tb:.6f} │  {max(0,opt_r2_tb*100):.2f}   │
  └────────────────────────────────┴──────────┴───────────┘

  Time-based R² is the realistic leaderboard estimate.
  3 submission files generated. Try all on HackerEarth!
""")

# Feature importance
top20 = lgb_importance.head(20)
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(range(len(top20)), top20['importance'].values, color='#4F46E5')
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance')
ax.set_title('Top 20 Features — v3 (Time-Based Tuning)')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'feature_importance.png'), dpi=150)
print(f"  feature_importance.png saved.")

print("\n" + "=" * 80)
print("DONE!")
print("=" * 80)
