# src/train.py

import os
import joblib
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.models.xgb_model import build_xgb_model
from src.models.torch_wrapper import IrisTorchClassifier
from src.utils import load_config


def train():
    # 1. Determine which config to load based on MODEL_CHOICE
    model_choice = os.environ.get("MODEL_CHOICE", "pytorch")  # "sklearn", "xgboost", or "pytorch"
    if model_choice == "sklearn":
        config_path = os.path.join(os.path.dirname(__file__), "../config/config_sklearn.yaml")
    elif model_choice == "xgboost":
        config_path = os.path.join(os.path.dirname(__file__), "../config/config_xgboost.yaml")
    else:
        config_path = os.path.join(os.path.dirname(__file__), "../config/config_pytorch.yaml")

    config = load_config(config_path)

    # 2. Load & merge all CSVs into a single DataFrame
    full_df = load_raw_data(config)

    # 3. Preprocess: split into train/val/test, apply ColumnTransformer, save processed arrays & preprocessor
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_data(full_df, config)

    # 4. Set up MLflow experiment
    mlflow_experiment = config["training"].get(
        "mlflow_experiment", f"iris_{model_choice}_experiment"
    )
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run():
        # Log basic info
        mlflow.log_param("model_type", model_choice)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("random_seed", config["training"]["random_seed"])

        # 5a. Train & evaluate RandomForest
        if model_choice == "sklearn":
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

            # Validation accuracy
            val_acc = model.score(X_val, y_val)
            mlflow.log_metric("val_accuracy", val_acc)
            print(f"[RandomForest] Validation Accuracy: {val_acc:.4f}")

            # Test accuracy
            test_acc = model.score(X_test, y_test)
            mlflow.log_metric("test_accuracy", test_acc)
            print(f"[RandomForest] Test Accuracy: {test_acc:.4f}")

            # Save model
            artifacts_dir = os.path.join(os.path.dirname(__file__), "../artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            model_path = os.path.join(artifacts_dir, "rf_model.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")

        # 5b. Train & evaluate XGBoost
        elif model_choice == "xgboost":
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
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # Validation accuracy
            val_acc = model.score(X_val, y_val)
            mlflow.log_metric("val_accuracy", val_acc)
            print(f"[XGBoost] Validation Accuracy: {val_acc:.4f}")

            # Test accuracy
            test_acc = model.score(X_test, y_test)
            mlflow.log_metric("test_accuracy", test_acc)
            print(f"[XGBoost] Test Accuracy: {test_acc:.4f}")

            # Save model
            artifacts_dir = os.path.join(os.path.dirname(__file__), "../artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            model_path = os.path.join(artifacts_dir, "xgb_model.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")

        # 5c. Train & evaluate PyTorch via sklearn wrapper
        else:
            batch_size = config["training"]["batch_size"]
            num_epochs = config["training"]["num_epochs"]
            learning_rate = config["training"]["learning_rate"]
            hidden_dim = config["training"]["hidden_dim"]
            random_seed = config["training"]["random_seed"]

            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("hidden_dim", hidden_dim)

            model = IrisTorchClassifier(
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                random_seed=random_seed,
            )
            model.fit(X_train, y_train)

            # Validation accuracy
            val_acc = model.score(X_val, y_val)
            mlflow.log_metric("val_accuracy", val_acc)
            print(f"[PyTorch] Validation Accuracy: {val_acc:.4f}")

            # Test accuracy
            test_acc = model.score(X_test, y_test)
            mlflow.log_metric("test_accuracy", test_acc)
            print(f"[PyTorch] Test Accuracy: {test_acc:.4f}")

            # Save model
            artifacts_dir = os.path.join(os.path.dirname(__file__), "../artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            model_path = os.path.join(artifacts_dir, "torch_model.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")

        # 6. (Optional) Re-save preprocessor for serving, if you want a consistent path
        #    preprocess_data already saved preprocessor under data/processed/preprocessor.joblib,
        #    so you can skip this step unless you want a copy under artifacts.
        #
        # preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
        # joblib.dump(preprocessor, preprocessor_path)
        # mlflow.log_artifact(preprocessor_path, artifact_path="model")


if __name__ == "__main__":
    train()
