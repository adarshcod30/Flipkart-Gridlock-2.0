#!/usr/bin/env python3
"""
Flipkart Gridlock 2.0 — Traffic Demand Prediction
End-to-end solution: Feature engineering → Model training → Ensemble → Submission
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
import optuna
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, time, json

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

# ============================================================
# GEOHASH DECODER (pure Python — no external library needed)
# ============================================================
_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_BASE32_MAP = {c: i for i, c in enumerate(_BASE32)}

def geohash_decode(gh):
    """Decode a geohash string to (latitude, longitude)."""
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    is_lon = True
    for c in gh:
        val = _BASE32_MAP.get(c, 0)
        for bit in [16, 8, 4, 2, 1]:
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if val & bit:
                    lon_range[0] = mid
                else:
                    lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if val & bit:
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid
            is_lon = not is_lon
    return (lat_range[0] + lat_range[1]) / 2, (lon_range[0] + lon_range[1]) / 2


print("=" * 80)
print("STEP 1: LOADING AND UNDERSTANDING THE DATA")
print("=" * 80)

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

print(f"\nTrain shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")

print("\n--- Train dtypes ---")
print(train.dtypes)
print("\n--- Null counts (train) ---")
print(train.isnull().sum())
print("\n--- Null counts (test) ---")
print(test.isnull().sum())

print("\n--- First 10 rows of train ---")
print(train.head(10).to_string())

print("\n--- Unique values ---")
for col in ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']:
    print(f"  {col}: {train[col].unique()}")

print(f"\n--- Sample geohash values: {train['geohash'].head(10).tolist()}")
print(f"--- Geohash lengths: {train['geohash'].str.len().unique()}")
print(f"--- Sample timestamp values: {train['timestamp'].head(20).tolist()}")
print(f"--- Sample day values: {sorted(train['day'].unique())}")

demand = train['demand']
print("\n--- Demand statistics ---")
print(f"  Mean:   {demand.mean():.6f}")
print(f"  Median: {demand.median():.6f}")
print(f"  Std:    {demand.std():.6f}")
print(f"  Min:    {demand.min():.6f}")
print(f"  Max:    {demand.max():.6f}")

outlier_thresh = demand.mean() + 3 * demand.std()
outliers = demand[demand > outlier_thresh]
print(f"  Outliers (>3σ): {len(outliers)} rows, max = {outliers.max():.6f}")

# ============================================================
# Combine for feature engineering
# ============================================================
train['is_train'] = 1
test['is_train'] = 0
test['demand'] = np.nan
df = pd.concat([train, test], axis=0, ignore_index=True)
print(f"\nCombined shape: {df.shape}")

# ============================================================
print("\n" + "=" * 80)
print("STEP 2: GEOHASH FEATURE ENGINEERING")
print("=" * 80)

# Decode geohash
print("Decoding geohash to lat/lon...")
coords = df['geohash'].apply(geohash_decode)
df['latitude'] = coords.apply(lambda x: x[0])
df['longitude'] = coords.apply(lambda x: x[1])

# Truncated geohash prefixes
for p in [2, 3, 4, 5]:
    df[f'geohash_p{p}'] = df['geohash'].str[:p]

# First 3 characters
df['gh_char1'] = df['geohash'].str[0]
df['gh_char2'] = df['geohash'].str[1]
df['gh_char3'] = df['geohash'].str[2]

# Sine/cosine of lat/lon
df['lat_rad'] = np.radians(df['latitude'])
df['lon_rad'] = np.radians(df['longitude'])
df['lat_sin'] = np.sin(df['lat_rad'])
df['lat_cos'] = np.cos(df['lat_rad'])
df['lon_sin'] = np.sin(df['lon_rad'])
df['lon_cos'] = np.cos(df['lon_rad'])

# Distance from centroid
lat_center = df['latitude'].mean()
lon_center = df['longitude'].mean()
df['dist_from_center'] = np.sqrt(
    (df['latitude'] - lat_center) ** 2 + (df['longitude'] - lon_center) ** 2
)

print(f"  Lat range: [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}]")
print(f"  Lon range: [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]")

# ============================================================
print("\n" + "=" * 80)
print("STEP 3: TEMPORAL FEATURE ENGINEERING")
print("=" * 80)

# Parse timestamp (format is H:M or HH:MM)
ts_parts = df['timestamp'].str.split(':', expand=True).astype(int)
df['hour'] = ts_parts[0]
df['minute'] = ts_parts[1]
df['total_minutes'] = df['hour'] * 60 + df['minute']
df['time_bucket_15'] = df['total_minutes'] // 15
df['time_bucket_30'] = df['total_minutes'] // 30
df['time_bucket_60'] = df['hour']  # same as hour

# Cyclical time encoding
for col, period in [('hour', 24), ('minute', 60), ('time_bucket_15', 96)]:
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)

# Time-of-day flags
df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
df['is_peak'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)

# Day features
df['day_of_week'] = df['day'] % 7
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

print(f"  Hour range: [{df['hour'].min()}, {df['hour'].max()}]")
print(f"  Day of week values: {sorted(df['day_of_week'].unique())}")
print(f"  Time bucket 15 range: [{df['time_bucket_15'].min()}, {df['time_bucket_15'].max()}]")

# ============================================================
print("\n" + "=" * 80)
print("STEP 4: INTERACTION AND DERIVED FEATURES")
print("=" * 80)

# Interaction strings
df['gh_tb15'] = df['geohash'] + '_' + df['time_bucket_15'].astype(str)
df['gh_tb30'] = df['geohash'] + '_' + df['time_bucket_30'].astype(str)
df['gh_hour'] = df['geohash'] + '_' + df['hour'].astype(str)
df['ghp4_tb15'] = df['geohash_p4'] + '_' + df['time_bucket_15'].astype(str)
df['dow_tb15'] = df['day_of_week'].astype(str) + '_' + df['time_bucket_15'].astype(str)
df['gh_dow'] = df['geohash'] + '_' + df['day_of_week'].astype(str)
df['roadtype_hour'] = df['RoadType'].fillna('Unknown') + '_' + df['hour'].astype(str)
df['weather_hour'] = df['Weather'].fillna('Unknown') + '_' + df['hour'].astype(str)
df['ghp4_hour'] = df['geohash_p4'] + '_' + df['hour'].astype(str)

# Binary encoding
df['LargeVehicles_bin'] = (df['LargeVehicles'] == 'Allowed').astype(int)
df['Landmarks_bin'] = (df['Landmarks'] == 'Yes').astype(int)

# Road capacity proxy
df['road_capacity'] = df['NumberofLanes'] * (1 + 0.5 * df['LargeVehicles_bin'])

# Infrastructure complexity
df['infra_complexity'] = df['Landmarks_bin'] + df['LargeVehicles_bin']

# Temperature features
df['Temperature'] = df['Temperature'].astype(float)
df['temp_squared'] = df['Temperature'] ** 2
df['temp_abs'] = df['Temperature'].abs()

print("  Interaction features created.")

# ============================================================
print("\n" + "=" * 80)
print("STEP 5: GROUP STATISTICS (TRAIN-ONLY)")
print("=" * 80)

train_mask = df['is_train'] == 1
train_demand = df.loc[train_mask, 'demand']
global_mean = train_demand.mean()
global_std = train_demand.std()

group_keys = {
    'geohash':      ['geohash'],
    'gh_tb15':      ['gh_tb15'],
    'gh_tb30':      ['gh_tb30'],
    'gh_hour':      ['gh_hour'],
    'ghp4_tb15':    ['ghp4_tb15'],
    'tb15':         ['time_bucket_15'],
    'tb30':         ['time_bucket_30'],
    'hour':         ['hour'],
    'dow':          ['day_of_week'],
    'dow_tb15':     ['dow_tb15'],
    'roadtype':     ['RoadType'],
    'weather':      ['Weather'],
    'roadtype_hour':['roadtype_hour'],
    'weather_hour': ['weather_hour'],
    'ghp4_hour':    ['ghp4_hour'],
    'gh_dow':       ['gh_dow'],
}

for grp_name, grp_cols in group_keys.items():
    print(f"  Computing stats for group: {grp_name} ({grp_cols})...")
    grp = df.loc[train_mask].groupby(grp_cols)['demand']
    stats = grp.agg(['mean', 'std', 'median', 'min', 'max', 'count']).reset_index()
    stats.columns = grp_cols + [
        f'{grp_name}_mean', f'{grp_name}_std', f'{grp_name}_median',
        f'{grp_name}_min', f'{grp_name}_max', f'{grp_name}_count'
    ]
    stats[f'{grp_name}_std'] = stats[f'{grp_name}_std'].fillna(0)
    df = df.merge(stats, on=grp_cols, how='left')
    # Fill missing for test rows where group key never appeared in train
    df[f'{grp_name}_mean'] = df[f'{grp_name}_mean'].fillna(global_mean)
    df[f'{grp_name}_std'] = df[f'{grp_name}_std'].fillna(0)
    df[f'{grp_name}_median'] = df[f'{grp_name}_median'].fillna(global_mean)
    df[f'{grp_name}_min'] = df[f'{grp_name}_min'].fillna(0)
    df[f'{grp_name}_max'] = df[f'{grp_name}_max'].fillna(1)
    df[f'{grp_name}_count'] = df[f'{grp_name}_count'].fillna(0)

print(f"\n  Total columns after group stats: {df.shape[1]}")

# ============================================================
print("\n" + "=" * 80)
print("STEP 6: ENCODE CATEGORICALS AND FINALIZE FEATURES")
print("=" * 80)

cat_cols = ['geohash', 'geohash_p2', 'geohash_p3', 'geohash_p4', 'geohash_p5',
            'gh_char1', 'gh_char2', 'gh_char3',
            'RoadType', 'Weather', 'LargeVehicles', 'Landmarks',
            'gh_tb15', 'gh_tb30', 'gh_hour', 'ghp4_tb15', 'dow_tb15',
            'gh_dow', 'roadtype_hour', 'weather_hour', 'ghp4_hour']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = df[col].fillna('Unknown').astype(str)
    le.fit(df[col])
    df[f'{col}_enc'] = le.transform(df[col])

# Build final feature list
exclude_cols = {'Index', 'demand', 'is_train', 'timestamp'}
exclude_cols.update(cat_cols)  # exclude raw string columns
exclude_cols.update({'lat_rad', 'lon_rad'})  # intermediate

feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

# Fill remaining NaN with median
for c in feature_cols:
    if df[c].isnull().any():
        median_val = df.loc[train_mask, c].median()
        df[c] = df[c].fillna(median_val)

print(f"\n  Total features: {len(feature_cols)}")
print(f"  Feature names: {feature_cols[:20]}... (showing first 20)")

# Split back
train_df = df[df['is_train'] == 1].reset_index(drop=True)
test_df = df[df['is_train'] == 0].reset_index(drop=True)

X = train_df[feature_cols].values
y = train_df['demand'].values
X_test = test_df[feature_cols].values
test_index = test_df['Index'].values

print(f"\n  X shape: {X.shape}, y shape: {y.shape}")
print(f"  X_test shape: {X_test.shape}")

# ============================================================
print("\n" + "=" * 80)
print("STEP 7: TRAIN THREE MODELS WITH 5-FOLD CV")
print("=" * 80)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# --- LightGBM Baseline ---
print("\n--- LightGBM Baseline ---")
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 3000,
    'learning_rate': 0.03,
    'num_leaves': 127,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'verbose': -1,
    'random_state': SEED,
    'n_jobs': -1,
}

oof_lgb = np.zeros(len(X))
preds_lgb = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"  Fold {fold+1}/{N_FOLDS}...", end=' ')
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])

    oof_lgb[va_idx] = model.predict(X_va)
    preds_lgb += model.predict(X_test) / N_FOLDS
    fold_r2 = r2_score(y_va, oof_lgb[va_idx])
    print(f"R² = {fold_r2:.6f}")

lgb_r2 = r2_score(y, oof_lgb)
print(f"  LightGBM OOF R² = {lgb_r2:.6f} (Score: {max(0, lgb_r2 * 100):.2f}/100)")

# Store feature importance from last fold
lgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# --- XGBoost ---
print("\n--- XGBoost ---")
xgb_params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'n_estimators': 3000,
    'learning_rate': 0.03,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': 0,
}

oof_xgb = np.zeros(len(X))
preds_xgb = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"  Fold {fold+1}/{N_FOLDS}...", end=' ')
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  verbose=False)

    oof_xgb[va_idx] = model_xgb.predict(X_va)
    preds_xgb += model_xgb.predict(X_test) / N_FOLDS
    fold_r2 = r2_score(y_va, oof_xgb[va_idx])
    print(f"R² = {fold_r2:.6f}")

xgb_r2 = r2_score(y, oof_xgb)
print(f"  XGBoost OOF R² = {xgb_r2:.6f} (Score: {max(0, xgb_r2 * 100):.2f}/100)")

# --- CatBoost ---
print("\n--- CatBoost ---")

oof_cb = np.zeros(len(X))
preds_cb = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"  Fold {fold+1}/{N_FOLDS}...", end=' ')
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model_cb = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        loss_function='RMSE',
        random_seed=SEED,
        verbose=0,
        use_best_model=True,
        early_stopping_rounds=150,
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)

    oof_cb[va_idx] = model_cb.predict(X_va)
    preds_cb += model_cb.predict(X_test) / N_FOLDS
    fold_r2 = r2_score(y_va, oof_cb[va_idx])
    print(f"R² = {fold_r2:.6f}")

cb_r2 = r2_score(y, oof_cb)
print(f"  CatBoost OOF R² = {cb_r2:.6f} (Score: {max(0, cb_r2 * 100):.2f}/100)")

# ============================================================
print("\n" + "=" * 80)
print("STEP 8: OPTUNA HYPERPARAMETER TUNING FOR LIGHTGBM")
print("=" * 80)

def optuna_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 3000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'verbose': -1,
        'random_state': SEED,
        'n_jobs': -1,
    }

    kf3 = KFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, va_idx in kf3.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pred = m.predict(X_va)
        scores.append(r2_score(y_va, pred))
    return np.mean(scores)

print("Running 50 Optuna trials (3-fold CV each)...")
study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(optuna_objective, n_trials=50, show_progress_bar=False)

best_params = study.best_params
print(f"\n  Best Optuna R²: {study.best_value:.6f}")
print(f"  Best params: {json.dumps(best_params, indent=2)}")

# Retrain with best params on 5-fold
print("\n--- Retraining LightGBM with tuned params ---")
tuned_lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 3000,
    'verbose': -1,
    'random_state': SEED,
    'n_jobs': -1,
}
tuned_lgb_params.update(best_params)

oof_lgb_tuned = np.zeros(len(X))
preds_lgb_tuned = np.zeros(len(X_test))
tuned_importances = np.zeros(len(feature_cols))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"  Fold {fold+1}/{N_FOLDS}...", end=' ')
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model_tuned = lgb.LGBMRegressor(**tuned_lgb_params)
    model_tuned.fit(X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])

    oof_lgb_tuned[va_idx] = model_tuned.predict(X_va)
    preds_lgb_tuned += model_tuned.predict(X_test) / N_FOLDS
    tuned_importances += model_tuned.feature_importances_ / N_FOLDS
    fold_r2 = r2_score(y_va, oof_lgb_tuned[va_idx])
    print(f"R² = {fold_r2:.6f}")

lgb_tuned_r2 = r2_score(y, oof_lgb_tuned)
print(f"  Tuned LightGBM OOF R² = {lgb_tuned_r2:.6f} (Score: {max(0, lgb_tuned_r2 * 100):.2f}/100)")

# Update feature importance with tuned model
lgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': tuned_importances
}).sort_values('importance', ascending=False)

# ============================================================
print("\n" + "=" * 80)
print("STEP 9: ENSEMBLE BLENDING WITH OPTIMIZED WEIGHTS")
print("=" * 80)

# Stack OOF predictions
oof_stack = np.column_stack([oof_lgb_tuned, oof_xgb, oof_cb])
test_stack = np.column_stack([preds_lgb_tuned, preds_xgb, preds_cb])

def neg_r2(weights):
    weights = np.abs(weights)
    weights = weights / weights.sum()
    blend = oof_stack @ weights
    return -r2_score(y, blend)

# Grid search for initial point
best_grid_r2 = -999
best_grid_w = None
for w1 in np.arange(0.1, 0.9, 0.1):
    for w2 in np.arange(0.05, 0.9 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue
        weights = np.array([w1, w2, w3])
        blend = oof_stack @ weights
        r2 = r2_score(y, blend)
        if r2 > best_grid_r2:
            best_grid_r2 = r2
            best_grid_w = weights.copy()

print(f"  Best grid search R²: {best_grid_r2:.6f}, weights: {best_grid_w}")

# Nelder-Mead optimization
result = minimize(neg_r2, best_grid_w, method='Nelder-Mead',
                  options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
opt_weights = np.abs(result.x)
opt_weights = opt_weights / opt_weights.sum()

oof_blend_opt = oof_stack @ opt_weights
opt_r2 = r2_score(y, oof_blend_opt)
print(f"  Optimized weights: LGB_tuned={opt_weights[0]:.4f}, XGB={opt_weights[1]:.4f}, CB={opt_weights[2]:.4f}")
print(f"  Optimized blend OOF R² = {opt_r2:.6f}")

# Equal weight comparison
oof_equal = oof_stack.mean(axis=1)
equal_r2 = r2_score(y, oof_equal)
print(f"  Equal-weight blend OOF R² = {equal_r2:.6f}")

# Choose best
if opt_r2 >= equal_r2:
    final_oof = oof_blend_opt
    final_preds = test_stack @ opt_weights
    final_r2 = opt_r2
    blend_type = 'optimized'
else:
    final_oof = oof_equal
    final_preds = test_stack.mean(axis=1)
    final_r2 = equal_r2
    blend_type = 'equal'

print(f"\n  Using {blend_type} blend: OOF R² = {final_r2:.6f} (Score: {max(0, final_r2 * 100):.2f}/100)")

# ============================================================
print("\n" + "=" * 80)
print("STEP 10: POST-PROCESS AND SAVE SUBMISSION")
print("=" * 80)

demand_min = y.min()
demand_max = y.max()
clipped = np.clip(final_preds, demand_min, demand_max)
n_clipped = np.sum((final_preds < demand_min) | (final_preds > demand_max))
print(f"  Clipped {n_clipped} predictions to [{demand_min:.6f}, {demand_max:.6f}]")
final_preds = clipped

submission = pd.DataFrame({
    'Index': test_df['Index'].values.astype(int),
    'demand': final_preds
})

assert submission.shape == (41778, 2), f"Wrong shape: {submission.shape}"
assert not submission.isnull().any().any(), "NaN found in submission!"
assert list(submission.columns) == ['Index', 'demand'], f"Wrong columns: {list(submission.columns)}"

submission.to_csv(os.path.join(BASE_DIR, 'submission.csv'), index=False)
print(f"  submission.csv saved with {len(submission)} rows.")

# ============================================================
print("\n" + "=" * 80)
print("STEP 11: FINAL SUMMARY")
print("=" * 80)

results = {
    'LightGBM (baseline)': lgb_r2,
    'XGBoost': xgb_r2,
    'CatBoost': cb_r2,
    'LightGBM (tuned)': lgb_tuned_r2,
    f'Ensemble ({blend_type})': final_r2,
}

print("\n  ┌──────────────────────────┬──────────┬────────────┐")
print("  │ Model                    │  OOF R²  │ Score/100  │")
print("  ├──────────────────────────┼──────────┼────────────┤")
for name, r2 in results.items():
    score = max(0, r2 * 100)
    print(f"  │ {name:<24} │ {r2:.6f} │  {score:6.2f}    │")
print("  └──────────────────────────┴──────────┴────────────┘")

# Top 20 features
print("\n  Top 20 Most Important Features (Tuned LightGBM):")
for i, row in lgb_importance.head(20).iterrows():
    print(f"    {lgb_importance.index.get_loc(i)+1:2d}. {row['feature']:<35} {row['importance']:.0f}")

# Feature importance chart
fig, ax = plt.subplots(figsize=(12, 8))
top20 = lgb_importance.head(20)
ax.barh(range(len(top20)), top20['importance'].values, color='#4F46E5')
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (split count)')
ax.set_title('Top 20 Feature Importances — Tuned LightGBM')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'feature_importance.png'), dpi=150)
print(f"\n  feature_importance.png saved.")

print("\n" + "=" * 80)
print("ALL DONE! submission.csv is ready for HackerEarth.")
print("=" * 80)
