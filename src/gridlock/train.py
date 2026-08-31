"""Train the spatial XGBoost model and report cross-validated R².

Training rows are restricted to day 49 (the day being forecast). Day 48
only ever contributes the historical-profile feature, so no row ever
trains on a signal derived from its own label — the same property the
test set (all day 49, no label) relies on at inference time.
"""

import json
import time

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from . import config
from .features import ALL_FEATURES, build_feature_pipeline, transform


def load_training_rows():
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    train_df, test_df, encoders = build_feature_pipeline(train_df, test_df)
    target_rows = transform(train_df[train_df["day"] == config.TARGET_DAY].copy(), encoders)
    return target_rows, encoders


def cross_validate(X, y):
    kfold = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        model = XGBRegressor(**config.XGB_PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        score = r2_score(y.iloc[val_idx], preds)
        scores.append(score)
        print(f"  fold {fold}: R2 = {score:.4f}")
    return scores


def main():
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)

    print("Loading data and building features...")
    rows, _ = load_training_rows()
    X, y = rows[ALL_FEATURES], rows["demand"]
    print(f"Training rows: {len(X)} | features: {len(ALL_FEATURES)}")

    print(f"Running {config.N_SPLITS}-fold cross-validation...")
    start = time.time()
    scores = cross_validate(X, y)
    elapsed = time.time() - start

    mean_r2 = sum(scores) / len(scores)
    std_r2 = (sum((s - mean_r2) ** 2 for s in scores) / len(scores)) ** 0.5
    print(f"CV R2: {mean_r2:.4f} +/- {std_r2:.4f} ({elapsed:.1f}s)")

    print("Fitting final model on all target-day rows...")
    model = XGBRegressor(**config.XGB_PARAMS)
    model.fit(X, y)
    model.save_model(config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

    metrics = {
        "cv_r2_mean": mean_r2,
        "cv_r2_std": std_r2,
        "cv_folds": scores,
        "n_splits": config.N_SPLITS,
        "n_train_rows": len(X),
        "n_features": len(ALL_FEATURES),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {config.METRICS_PATH}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        importances = pd.Series(model.feature_importances_, index=ALL_FEATURES)
        top20 = importances.sort_values(ascending=False).head(20)
        plt.figure(figsize=(8, 6))
        top20[::-1].plot(kind="barh")
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig(config.FEATURE_IMPORTANCE_PLOT)
        print(f"Feature importance plot saved to {config.FEATURE_IMPORTANCE_PLOT}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
