# src/train.py

import os
import joblib
import mlflow
import numpy as np
import csv
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.models.xgb_model import build_xgb_model
from src.models.torch_wrapper import TorchRegressor
from src.utils import load_config


def append_metrics_to_csv(csv_path: Path, row: dict):
    """
    Append a single row of metrics to `metrics_history.csv`.
    If the file does not exist, write header first.
    """
    header = ["timestamp", "model_choice", "val_mse", "val_r2", "test_mse", "test_r2"]
    file_exists = csv_path.exists()

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train():
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

    # 2. Load & merge all CSVs into a single DataFrame
    full_df = load_raw_data(config)

    # 3. Preprocess: split into train/val/test, apply ColumnTransformer, save processed arrays & preprocessor
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_data(full_df, config)

    # 4. Set up MLflow experiment
    experiment_name = config["training"].get("mlflow_experiment", f"reg_{model_choice}_exp")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("model_type", model_choice)
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("val_size", X_val.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])
        mlflow.log_param("random_seed", config["training"]["random_seed"])

        # ── RandomForestRegressor ─────────────────────────────────────────
        if model_choice == "sklearn":
            n_estimators = config["training"]["n_estimators"]
            max_depth = config["training"]["max_depth"]
            min_samples_split = config["training"]["min_samples_split"]
            bootstrap = config["training"]["bootstrap"]

            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_samples_split", min_samples_split)
            mlflow.log_param("bootstrap", bootstrap)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                bootstrap=bootstrap,
                random_state=config["training"]["random_seed"],
            )
            model.fit(X_train, y_train)

            # Validation
            val_preds = model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_preds)
            val_r2 = r2_score(y_val, val_preds)
            mlflow.log_metric("val_mse", val_mse)
            mlflow.log_metric("val_r2", val_r2)
            print(f"[RF] Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}")

            # Test
            test_preds = model.predict(X_test)
            test_mse = mean_squared_error(y_test, test_preds)
            test_r2 = r2_score(y_test, test_preds)
            mlflow.log_metric("test_mse", test_mse)
            mlflow.log_metric("test_r2", test_r2)
            print(f"[RF] Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}")

            # Save model
            artifacts_dir = base_dir.parent / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            model_path = artifacts_dir / "rf_regressor.joblib"
            joblib.dump(model, str(model_path))
            mlflow.log_artifact(str(model_path), artifact_path="model")

        # ── XGBRegressor ────────────────────────────────────────────────────
        elif model_choice == "xgboost":
            xgb_params = {
                "objective": "reg:squarederror",
                "learning_rate": config["training"]["learning_rate"],
                "max_depth": config["training"]["max_depth"],
                "n_estimators": config["training"]["n_estimators"],
                "subsample": config["training"]["subsample"],
                "colsample_bytree": config["training"]["colsample_bytree"],
                "random_state": config["training"]["random_seed"],
            }

            mlflow.log_params(xgb_params)
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Validation
            val_preds = model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_preds)
            val_r2 = r2_score(y_val, val_preds)
            mlflow.log_metric("val_mse", val_mse)
            mlflow.log_metric("val_r2", val_r2)
            print(f"[XGB] Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}")

            # Test
            test_preds = model.predict(X_test)
            test_mse = mean_squared_error(y_test, test_preds)
            test_r2 = r2_score(y_test, test_preds)
            mlflow.log_metric("test_mse", test_mse)
            mlflow.log_metric("test_r2", test_r2)
            print(f"[XGB] Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}")

            # Save model
            artifacts_dir = base_dir.parent / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            model_path = artifacts_dir / "xgb_regressor.joblib"
            joblib.dump(model, str(model_path))
            mlflow.log_artifact(str(model_path), artifact_path="model")

        # ── PyTorch Regressor ───────────────────────────────────────────────
        else:
            input_dim = X_train.shape[1]
            batch_size = config["training"]["batch_size"]
            num_epochs = config["training"]["num_epochs"]
            learning_rate = config["training"]["learning_rate"]
            hidden_dim = config["training"]["hidden_dim"]
            random_seed = config["training"]["random_seed"]

            mlflow.log_param("input_dim", input_dim)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("hidden_dim", hidden_dim)

            model = TorchRegressor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                random_seed=random_seed,
            )
            model.fit(X_train, y_train)

            # Validation
            val_preds = model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_preds)
            val_r2 = r2_score(y_val, val_preds)
            mlflow.log_metric("val_mse", val_mse)
            mlflow.log_metric("val_r2", val_r2)
            print(f"[PyTorch] Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}")

            # Test
            test_preds = model.predict(X_test.astype("float32"))
            test_mse = mean_squared_error(y_test, test_preds)
            test_r2 = r2_score(y_test, test_preds)
            mlflow.log_metric("test_mse", test_mse)
            mlflow.log_metric("test_r2", test_r2)
            print(f"[PyTorch] Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}")

            # Save model
            artifacts_dir = base_dir.parent / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            model_path = artifacts_dir / "torch_regressor.joblib"
            joblib.dump(model, str(model_path))
            mlflow.log_artifact(str(model_path), artifact_path="model")

        # ── Append metrics to `metrics_history.csv` ────────────────────────
        artifacts_dir = base_dir.parent / "artifacts"
        metrics_csv = artifacts_dir / "metrics_history.csv"

        # Build a metrics row dictionary
        metrics_row = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_choice": model_choice,
            "val_mse": float(val_mse),
            "val_r2": float(val_r2),
            "test_mse": float(test_mse),
            "test_r2": float(test_r2),
        }

        append_metrics_to_csv(metrics_csv, metrics_row)

        # Log the CSV as an MLflow artifact:
        mlflow.log_artifact(str(metrics_csv), artifact_path="model")


if __name__ == "__main__":
    train()
