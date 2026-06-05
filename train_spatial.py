#!/usr/bin/env python3
"""
Flipkart Gridlock 2.0 — Traffic Demand Prediction
Training pipeline for the spatial XGBoost model.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

def decode_geohash(geohash):
    __base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    __decodemap = {c: i for i, c in enumerate(__base32)}
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    is_lon = True
    for c in geohash:
        v = __decodemap[c]
        for i in range(4, -1, -1):
            bit = (v >> i) & 1
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if bit: lon_range[0] = mid
                else: lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if bit: lat_range[0] = mid
                else: lat_range[1] = mid
            is_lon = not is_lon
    return (lat_range[0] + lat_range[1]) / 2, (lon_range[0] + lon_range[1]) / 2

print("Loading training data...")
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

print("Extracting spatial and temporal features...")
train['lat'] = train['geohash'].apply(lambda x: decode_geohash(x)[0])
train['lon'] = train['geohash'].apply(lambda x: decode_geohash(x)[1])
train['hour'] = train['timestamp'].apply(lambda x: int(x.split(':')[0]))
train['minute'] = train['timestamp'].apply(lambda x: int(x.split(':')[1]))

features = ['lat', 'lon', 'hour', 'minute', 'day']
X_train = train[features]
y_train = train['demand']

print("Training spatial XGBoost Regressor...")
model = xgb.XGBRegressor(
    n_estimators=450,
    max_depth=12,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    tree_method='hist',
    random_state=42
)

model.fit(X_train, y_train)

model_path = os.path.join(BASE_DIR, 'spatial_model.json')
model.save_model(model_path)
print(f"Model successfully saved to {model_path}")
