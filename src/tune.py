# src/tune.py

import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import torch

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.models.torch_wrapper import TorchRegressor
from src.utils import load_config


def tune():
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

    # 2. Load & preprocess data
    full_df = load_raw_data(config)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_data(full_df, config)

    # For tuning, use only the training split (X_train, y_train) with CV
    # 3. Set up GridSearchCV for each model type
    if model_choice == "sklearn":
        # 3a. RandomForestRegressor GridSearchCV
        param_grid = config["tuning"]["param_grid"]
        base_estimator = RandomForestRegressor(random_state=config["training"]["random_seed"])
        gs = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            scoring="r2",
            cv=3,
            n_jobs=-1,
            verbose=2,
            refit=True
        )
        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        print("Best parameters (RF):", gs.best_params_)
        print("Best CV R² (RF):", gs.best_score_)

    elif model_choice == "xgboost":
        # 3b. XGBRegressor GridSearchCV
        # We create a custom wrapper to fix objective and random_seed
        class XGBWrapper(xgb.XGBRegressor):
            def __init__(self, learning_rate, max_depth, n_estimators, subsample, colsample_bytree):
                super().__init__(
                    objective="reg:squarederror",
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=config["training"]["random_seed"],
                )

        param_grid = config["tuning"]["param_grid"]
        gs = GridSearchCV(
            estimator=XGBWrapper(
                learning_rate=config["training"]["learning_rate"],
                max_depth=config["training"]["max_depth"],
                n_estimators=config["training"]["n_estimators"],
                subsample=config["training"]["subsample"],
                colsample_bytree=config["training"]["colsample_bytree"],
            ),
            param_grid=param_grid,
            scoring="r2",
            cv=3,
            n_jobs=-1,
            verbose=2,
            refit=True
        )
        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        print("Best parameters (XGB):", gs.best_params_)
        print("Best CV R² (XGB):", gs.best_score_)

    else:
        # 3c. PyTorch regressor via TorchRegressor + GridSearchCV
        input_dim = X_train.shape[1]
        random_seed = config["training"]["random_seed"]

        # Define a subclass that fixes input_dim and random_seed
        class WrappedTorchRegressor(TorchRegressor):
            def __init__(self, hidden_dim, learning_rate, num_epochs, batch_size):
                super().__init__(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    random_seed=random_seed,
                )

        pg = config["tuning"]["param_grid"]
        param_grid = {
            "hidden_dim": pg["hidden_dim"],
            "learning_rate": pg["learning_rate"],
            "num_epochs": pg["num_epochs"],
            "batch_size": pg["batch_size"],
        }

        gs = GridSearchCV(
            estimator=WrappedTorchRegressor(
                hidden_dim=config["training"]["hidden_dim"],
                learning_rate=config["training"]["learning_rate"],
                num_epochs=config["training"]["num_epochs"],
                batch_size=config["training"]["batch_size"],
            ),
            param_grid=param_grid,
            scoring="r2",
            cv=3,
            n_jobs=-1,
            verbose=2,
            refit=True
        )
        gs.fit(X_train.astype("float32"), y_train.astype("float32"))

        best_model = gs.best_estimator_
        print("Best parameters (PyTorch):", gs.best_params_)
        print("Best CV R² (PyTorch):", gs.best_score_)

    # 4. Evaluate the best model on the hold-out test set
    test_preds = best_model.predict(X_test if model_choice != "pytorch" else X_test.astype("float32"))
    test_mse = mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    print(f"[{model_choice.upper()}] Test MSE: {test_mse:.4f}, Test R²: {test_r2:.4f}")

    # 5. Save the best model artifact
    artifacts_dir = base_dir.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    if model_choice == "sklearn":
        model_path = artifacts_dir / "rf_best_regressor.joblib"
    elif model_choice == "xgboost":
        model_path = artifacts_dir / "xgb_best_regressor.joblib"
    else:
        model_path = artifacts_dir / "torch_best_regressor.joblib"

    joblib.dump(best_model, str(model_path))
    print(f"Saved best model to: {model_path}")


if __name__ == "__main__":
    tune()
