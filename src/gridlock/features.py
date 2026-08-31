"""Feature engineering: geohash decoding, temporal indices, the day-48
historical demand profile, and auxiliary road/weather features."""

import numpy as np
import pandas as pd

from .config import HISTORY_DAY

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODE_MAP = {c: i for i, c in enumerate(_BASE32)}

PROFILE_COLUMNS = [f"d48_t{i}" for i in range(96)]
CORE_FEATURES = ["lat", "lon", "hour", "minute", "time_idx"]
AUX_FEATURES = [
    "road_type_code",
    "weather_code",
    "number_of_lanes",
    "large_vehicles_allowed",
    "has_landmarks",
    "temperature",
    "temperature_missing",
]
ALL_FEATURES = CORE_FEATURES + AUX_FEATURES + PROFILE_COLUMNS + ["d48_same_time"]


def decode_geohash(geohash):
    """Decode a base32 geohash string into a (lat, lon) center-point."""
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    is_lon = True
    for char in geohash:
        value = _DECODE_MAP[char]
        for bit_index in range(4, -1, -1):
            bit = (value >> bit_index) & 1
            bounds = lon_range if is_lon else lat_range
            mid = (bounds[0] + bounds[1]) / 2
            if bit:
                bounds[0] = mid
            else:
                bounds[1] = mid
            is_lon = not is_lon
    return (lat_range[0] + lat_range[1]) / 2, (lon_range[0] + lon_range[1]) / 2


def add_spatiotemporal_features(df):
    df = df.copy()
    coords = df["geohash"].apply(decode_geohash)
    df["lat"] = coords.apply(lambda c: c[0])
    df["lon"] = coords.apply(lambda c: c[1])
    df["hour"] = df["timestamp"].str.split(":").str[0].astype(int)
    df["minute"] = df["timestamp"].str.split(":").str[1].astype(int)
    df["time_idx"] = df["hour"] * 4 + df["minute"] // 15
    return df


def fit_categorical_codes(*frames, column):
    """Build a stable category->code mapping from the union of frames,
    so train and test rows always encode the same category to the same
    integer, and codes can be inspected later."""
    values = pd.concat([f[column].fillna("Unknown") for f in frames])
    categories = pd.Categorical(values).categories
    return {category: code for code, category in enumerate(categories)}


def add_auxiliary_features(df, road_type_map, weather_map, temperature_median):
    df = df.copy()
    df["road_type_code"] = df["RoadType"].fillna("Unknown").map(road_type_map)
    df["weather_code"] = df["Weather"].fillna("Unknown").map(weather_map)
    df["number_of_lanes"] = df["NumberofLanes"]
    df["large_vehicles_allowed"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["has_landmarks"] = (df["Landmarks"] == "Yes").astype(int)
    df["temperature_missing"] = df["Temperature"].isna().astype(int)
    df["temperature"] = df["Temperature"].fillna(temperature_median)
    return df


def fit_auxiliary_encoders(train_df, test_df):
    """Fit the categorical maps and imputation values once, from train (and
    test feature columns, never test labels) so train.py and predict.py can
    never drift into inconsistent encodings the way the original
    train_spatial.py / solution.ipynb pair did."""
    road_type_map = fit_categorical_codes(train_df, test_df, column="RoadType")
    weather_map = fit_categorical_codes(train_df, test_df, column="Weather")
    temperature_median = train_df["Temperature"].median()
    return road_type_map, weather_map, temperature_median


def build_day48_profile(train_df):
    """Pivot day-48 demand into a (geohash x 96 time slots) lookup table.
    Missing slots for a known geohash are filled by linear interpolation
    across the day; geohashes absent from day 48 fall back to the global
    day-48 profile at prediction time."""
    day48 = train_df[train_df["day"] == HISTORY_DAY]
    profile = day48.pivot_table(index="geohash", columns="time_idx", values="demand", aggfunc="mean")
    profile = profile.reindex(columns=range(96))
    profile = profile.interpolate(method="linear", axis=1, limit_direction="both")
    profile.columns = PROFILE_COLUMNS
    global_profile = profile.mean(axis=0)
    return profile, global_profile


def attach_day48_profile(df, profile, global_profile):
    df = df.merge(profile, on="geohash", how="left")
    for column in PROFILE_COLUMNS:
        df[column] = df[column].fillna(global_profile[column])
    row_time_idx = df["time_idx"].to_numpy()
    profile_matrix = df[PROFILE_COLUMNS].to_numpy()
    df = df.copy()
    df["d48_same_time"] = profile_matrix[np.arange(len(df)), row_time_idx]
    return df


def build_feature_pipeline(train_df, test_df):
    """Fit every encoder/lookup table on train (+ test feature columns) once
    and return them alongside the day-48 profile, so both the training and
    prediction entry points transform rows identically."""
    train_df = add_spatiotemporal_features(train_df)
    test_df = add_spatiotemporal_features(test_df)
    road_type_map, weather_map, temperature_median = fit_auxiliary_encoders(train_df, test_df)
    profile, global_profile = build_day48_profile(train_df)
    encoders = {
        "road_type_map": road_type_map,
        "weather_map": weather_map,
        "temperature_median": temperature_median,
        "profile": profile,
        "global_profile": global_profile,
    }
    return train_df, test_df, encoders


def transform(df, encoders):
    df = attach_day48_profile(df, encoders["profile"], encoders["global_profile"])
    df = add_auxiliary_features(
        df,
        encoders["road_type_map"],
        encoders["weather_map"],
        encoders["temperature_median"],
    )
    return df
