"""Load the trained model and generate submission.csv for the test set."""

import pandas as pd
from xgboost import XGBRegressor

from . import config
from .features import ALL_FEATURES, build_feature_pipeline, transform


def main():
    print("Loading data and rebuilding features (train is needed for the day-48 profile)...")
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    _, test_df, encoders = build_feature_pipeline(train_df, test_df)
    test_df = transform(test_df, encoders)

    print(f"Loading model from {config.MODEL_PATH}...")
    model = XGBRegressor()
    model.load_model(config.MODEL_PATH)

    print("Running inference...")
    predictions = model.predict(test_df[ALL_FEATURES])
    predictions = predictions.clip(0, 1)

    submission = pd.DataFrame({"Index": test_df["Index"], "demand": predictions})
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Saved {len(submission)} predictions to {config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
