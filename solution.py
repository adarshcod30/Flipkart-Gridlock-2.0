#!/usr/bin/env python3
"""
Flipkart Gridlock 2.0 — Traffic Demand Prediction
Inference pipeline using pre-trained spatial XGBoost model.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

print("=" * 80)
print("STEP 1: LOAD DATA AND EXTRACT LAT/LON")
print("=" * 80)

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

test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Feature Engineering
test['lat'] = test['geohash'].apply(lambda x: decode_geohash(x)[0])
test['lon'] = test['geohash'].apply(lambda x: decode_geohash(x)[1])
test['hour'] = test['timestamp'].apply(lambda x: int(x.split(':')[0]))
test['minute'] = test['timestamp'].apply(lambda x: int(x.split(':')[1]))

features = ['lat', 'lon', 'hour', 'minute', 'day']
X_test = test[features]

print("Extracted Spatial and Temporal Features.")

print("\n" + "=" * 80)
print("STEP 2: LOAD PRE-TRAINED SPATIAL MODEL")
print("=" * 80)

# Load the model
model = xgb.XGBRegressor()
model.load_model(os.path.join(BASE_DIR, 'spatial_model.json'))
print("Loaded XGBoost Spatial Architecture.")

print("\n" + "=" * 80)
print("STEP 3: GENERATE PREDICTIONS")
print("=" * 80)

pred = model.predict(X_test)
pred = np.clip(pred, 0, 1)

submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': pred
})

out_path = os.path.join(BASE_DIR, 'submission.csv')
submission.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
print("Done!")
