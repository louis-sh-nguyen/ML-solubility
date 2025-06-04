# src/evaluate.py

import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

from src.utils import load_config


def evaluate():
    # 1. Determine which config to load based on MODEL_CHOICE
    model_choice = os.environ.get("MODEL_CHOICE", "pytorch")  # "sklearn", "xgboost", or "pytorch"
    base_dir = Path(__file__).parent

    if model_choice == "sklearn":
        config_path = base_dir.parent / "config" / "config_sklearn.yaml"
    elif model_choice == "xgboost":
        config_path = base_dir.parent / "config" / "config_xgboost.yaml"
    else:
        config_path = base_dir.parent / "config" / "config_pytorch.yaml"

    config = load_config(str(config_path))

    # 2. Load processed test data
    processed_dir = base_dir.parent / "data" / "processed"
    X_test_path = processed_dir / "X_test.npy"
    y_test_path = processed_dir / "y_test.npy"

    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError("Processed test data not found. Run train.py first.")

    X_test = np.load(str(X_test_path), allow_pickle=True)
    y_test = np.load(str(y_test_path), allow_pickle=True)

    # 3. Load the trained model
    artifacts_dir = base_dir.parent / "artifacts"
    if model_choice == "sklearn":
        model_path = artifacts_dir / "rf_regressor.joblib"
    elif model_choice == "xgboost":
        model_path = artifacts_dir / "xgb_regressor.joblib"
    else:
        model_path = artifacts_dir / "torch_regressor.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")

    model = joblib.load(str(model_path))

    # 4. Make predictions and compute metrics
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    # 5. Print results
    print(f"[{model_choice.upper()}] Test set results:")
    print(f"  • MSE: {mse:.4f}")
    print(f"  • R² : {r2:.4f}")


if __name__ == "__main__":
    evaluate()
