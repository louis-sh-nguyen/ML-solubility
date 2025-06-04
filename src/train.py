# src/train.py

import os
import joblib
import mlflow
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.models.xgb_model import build_xgb_model
from src.models.torch_wrapper import TorchRegressor
from src.utils import load_config


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
        # Use shape[0] instead of len()
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
            
            # Check for NaN values before calculating metrics
            if np.isnan(val_preds).any():
                print(f"WARNING: NaN values detected in validation predictions: {np.sum(np.isnan(val_preds))}")
                # Option 1: Replace NaNs with zeros or mean values
                val_preds = np.nan_to_num(val_preds, nan=np.nanmean(val_preds) if not np.isnan(val_preds).all() else 0)
            
            if np.isnan(y_val).any():
                print(f"WARNING: NaN values detected in validation target: {np.sum(np.isnan(y_val))}")
                # This is more concerning as it suggests issues with your ground truth data
            
            # Now calculate metrics with cleaned data
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
            
            # Check for NaN values before calculating metrics
            if np.isnan(val_preds).any():
                print(f"WARNING: NaN values detected in validation predictions: {np.sum(np.isnan(val_preds))}")
                # Option 1: Replace NaNs with zeros or mean values
                val_preds = np.nan_to_num(val_preds, nan=np.nanmean(val_preds) if not np.isnan(val_preds).all() else 0)
            
            if np.isnan(y_val).any():
                print(f"WARNING: NaN values detected in validation target: {np.sum(np.isnan(y_val))}")
                # This is more concerning as it suggests issues with your ground truth data
            
            val_mse = mean_squared_error(y_val, val_preds)
            val_r2 = r2_score(y_val, val_preds)
            mlflow.log_metric("val_mse", val_mse)
            mlflow.log_metric("val_r2", val_r2)
            print(f"[PyTorch] Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}")

            # Test
            test_preds = model.predict(X_test)
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

        # ── (Optional) Save preprocessor under artifacts ─────────────────────
        # The preprocessor was already saved under data/processed/preprocessor.joblib.
        # To copy it into artifacts, uncomment below:
        #
        # proc_src = base_dir.parent / "data" / "processed" / "preprocessor.joblib"
        # proc_dst = base_dir.parent / "artifacts" / "preprocessor.joblib"
        # joblib.copy(proc_src, proc_dst)
        # mlflow.log_artifact(str(proc_dst), artifact_path="model")


if __name__ == "__main__":
    train()
