# src/train.py

import os
import numpy as np
import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.data.load_data import load_raw_data, save_processed_data
from src.data.preprocess import preprocess_data
from src.models.pytorch_model import IrisNet
from src.models.xgb_model import build_xgb_model
from src.models.torch_wrapper import IrisTorchClassifier
from src.utils import load_config, get_device


def train():
    # 1. Determine which config to load based on environment variable or default
    #    (Here we simply pick one; in practice you might pass an argument)
    model_choice = os.environ.get("MODEL_CHOICE", "pytorch")  # options: "sklearn", "pytorch", "xgboost"

    if model_choice == "sklearn":
        config_path = os.path.join(os.path.dirname(__file__), "../config/config_sklearn.yaml")
    elif model_choice == "xgboost":
        config_path = os.path.join(os.path.dirname(__file__), "../config/config_xgboost.yaml")
    else:
        config_path = os.path.join(os.path.dirname(__file__), "../config/config_pytorch.yaml")

    config = load_config(config_path)

    # 2. Load raw data (Iris + any new CSVs)
    df_full = load_raw_data(config)

    # 3. Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df_full, config)

    # 4. Save processed data
    save_processed_data(X_train, y_train, X_test, y_test, config)

    # 5. Set up MLflow experiment
    mlflow_experiment = config["training"].get("mlflow_experiment", f"iris_{model_choice}_experiment")
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run():
        # Log data split info
        mlflow.log_param("model_type", model_choice)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("random_seed", config["training"]["random_seed"])

        if model_choice == "sklearn":
            # 6a. Train a RandomForestClassifier
            n_estimators = config["training"]["n_estimators"]
            max_depth = config["training"]["max_depth"]
            min_samples_split = config["training"]["min_samples_split"]
            criterion = config["training"]["criterion"]
            bootstrap = config["training"]["bootstrap"]

            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_samples_split", min_samples_split)
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("bootstrap", bootstrap)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
                bootstrap=bootstrap,
                random_state=config["training"]["random_seed"],
            )
            model.fit(X_train, y_train)

            test_acc = model.score(X_test, y_test)
            mlflow.log_metric("test_accuracy", test_acc)
            print(f"[RandomForest] Test Accuracy: {test_acc:.4f}")

            # 7a. Save model
            artifacts_dir = os.path.join(os.path.dirname(__file__), "../artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            model_path = os.path.join(artifacts_dir, "rf_model.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")

        elif model_choice == "xgboost":
            # 6b. Train an XGBoost classifier
            xgb_params = {
                "objective": config["training"]["objective"],
                "num_class": config["training"]["num_class"],
                "learning_rate": config["training"]["learning_rate"],
                "max_depth": config["training"]["max_depth"],
                "n_estimators": config["training"]["n_estimators"],
                "subsample": config["training"]["subsample"],
                "colsample_bytree": config["training"]["colsample_bytree"],
                "use_label_encoder": config["training"]["use_label_encoder"],
                "eval_metric": config["training"]["eval_metric"],
                "random_state": config["training"]["random_seed"],
            }

            mlflow.log_params(xgb_params)
            model = build_xgb_model(xgb_params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            test_acc = model.score(X_test, y_test)
            mlflow.log_metric("test_accuracy", test_acc)
            print(f"[XGBoost] Test Accuracy: {test_acc:.4f}")

            # 7b. Save model
            artifacts_dir = os.path.join(os.path.dirname(__file__), "../artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            model_path = os.path.join(artifacts_dir, "xgb_model.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")

        else:
            # 6c. Train a PyTorch neural network via the wrapper
            batch_size = config["training"]["batch_size"]
            num_epochs = config["training"]["num_epochs"]
            learning_rate = config["training"]["learning_rate"]
            hidden_dim = config["training"]["hidden_dim"]
            random_seed = config["training"]["random_seed"]

            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("hidden_dim", hidden_dim)

            # Wrap and train
            model = IrisTorchClassifier(
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                random_seed=random_seed,
            )
            model.fit(X_train, y_train)

            test_acc = model.score(X_test, y_test)
            mlflow.log_metric("test_accuracy", test_acc)
            print(f"[PyTorch] Test Accuracy: {test_acc:.4f}")

            # 7c. Save model
            artifacts_dir = os.path.join(os.path.dirname(__file__), "../artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            model_path = os.path.join(artifacts_dir, "torch_model.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")

        # 8. Save and log the fitted scaler
        artifacts_dir = os.path.join(os.path.dirname(__file__), "../artifacts")
        scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="model")


if __name__ == "__main__":
    train()
