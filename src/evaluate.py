# src/evaluate.py

import os
import numpy as np
import joblib
import torch
from sklearn.metrics import accuracy_score, classification_report
from src.utils import load_config


def evaluate():
    # 1. Determine which config to load based on environment variable
    model_choice = os.environ.get("MODEL_CHOICE", "pytorch")  # "sklearn", "xgboost", or "pytorch"

    if model_choice == "sklearn":
        config_path = os.path.join(os.path.dirname(__file__), "../config/config_sklearn.yaml")
    elif model_choice == "xgboost":
        config_path = os.path.join(os.path.dirname(__file__), "../config/config_xgboost.yaml")
    else:
        config_path = os.path.join(os.path.dirname(__file__), "../config/config_pytorch.yaml")

    config = load_config(config_path)

    # 2. Load processed data
    processed_dir = os.path.join(os.path.dirname(__file__), "../data/processed")
    X_test = np.load(os.path.join(processed_dir, "test.npy"))
    y_test = np.load(os.path.join(processed_dir, "test_labels.npy"))

    # 3. Load model and scaler artefacts
    artifacts_dir = os.path.join(os.path.dirname(__file__), "../artifacts")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    # 4. Prepare test features
    X_test_scaled = scaler.transform(X_test)

    if model_choice == "sklearn":
        # 5a. Load RandomForest model
        model_path = os.path.join(artifacts_dir, "rf_model.joblib")
        model = joblib.load(model_path)

        # 6a. Predict and evaluate
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        print(f"[RandomForest] Test Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, preds))

    elif model_choice == "xgboost":
        # 5b. Load XGBoost model
        model_path = os.path.join(artifacts_dir, "xgb_model.joblib")
        model = joblib.load(model_path)

        # 6b. Predict and evaluate
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        print(f"[XGBoost] Test Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, preds))

    else:
        # 5c. Load PyTorch wrapper model
        model_path = os.path.join(artifacts_dir, "torch_model.joblib")
        model = joblib.load(model_path)

        # 6c. Predict and evaluate
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        print(f"[PyTorch] Test Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, preds))


if __name__ == "__main__":
    evaluate()
