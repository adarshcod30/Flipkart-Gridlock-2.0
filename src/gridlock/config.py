"""Paths, constants, and model hyperparameters for the Gridlock pipeline."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_PATH = ARTIFACTS_DIR / "spatial_model.json"
SUBMISSION_PATH = ARTIFACTS_DIR / "submission.csv"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PLOT = ARTIFACTS_DIR / "feature_importance.png"

# The competition provides two days of data: 48 (fully recorded) and 49
# (the day being forecast). Day 48 is used purely as a historical prior —
# training rows are restricted to day 49 so the profile feature never
# leaks a row's own label into its own inputs.
HISTORY_DAY = 48
TARGET_DAY = 49

RANDOM_STATE = 42
N_SPLITS = 5

XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.5,
    "min_child_weight": 5,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
}
