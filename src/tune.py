# src/tune.py

import os
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from skorch import NeuralNetClassifier
import torch
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.models.iris_net import IrisNet  # PyTorch nn.Module
from src.models.xgb_model import build_xgb_model
from src.models.torch_wrapper import IrisTorchClassifier
from src.utils import load_config


def tune():
    # 1. Determine model choice from environment
    model_choice = os.environ.get("MODEL_CHOICE", "pytorch")  # "sklearn", "xgboost", or "pytorch"

    if model_choice == "sklearn":
        config_path = os.path.join(os.path.dirname(__file__), "../config/config_sklearn.yaml")
    elif model_choice == "xgboost":
        config_path = os.path.join(os.path.dirname(__file__), "../config/config_xgboost.yaml")
    else:
        config_path = os.path.join(os.path.dirname(__file__), "../config/config_pytorch.yaml")

    config = load_config(config_path)

    # 2. Load and preprocess data
    df = load_raw_data(config)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, config)

    if model_choice == "sklearn":
        # 3a. RandomForest with GridSearchCV
        param_grid = config["tuning"]["param_grid"]
        base_estimator = RandomForestClassifier(random_state=config["training"]["random_seed"])
        gs = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
            verbose=2,
            refit=True
        )
        gs.fit(X_train, y_train)
        print("Best params:", gs.best_params_)
        print("Best CV accuracy:", gs.best_score_)
        best_model = gs.best_estimator_

    elif model_choice == "xgboost":
        # 3b. XGBoost with GridSearchCV
        param_grid = config["tuning"]["param_grid"]
        def xgb_wrapper(**params):
            # Merge fixed params with grid-search params
            fixed = {
                "objective": config["training"]["objective"],
                "num_class": config["training"]["num_class"],
                "use_label_encoder": config["training"]["use_label_encoder"],
                "eval_metric": config["training"]["eval_metric"],
                "random_state": config["training"]["random_seed"],
            }
            fixed.update(params)
            return xgb.XGBClassifier(**fixed)

        gs = GridSearchCV(
            estimator=xgb_wrapper(),
            param_grid=param_grid,
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
            verbose=2,
            refit=True
        )
        gs.fit(X_train, y_train)
        print("Best params:", gs.best_params_)
        print("Best CV accuracy:", gs.best_score_)
        best_model = gs.best_estimator_

    else:
        # 3c. PyTorch via Skorch + GridSearchCV
        # Wrap IrisNet in Skorch’s NeuralNetClassifier
        device = "cuda" if torch.cuda.is_available() else "cpu"
        skorch_net = NeuralNetClassifier(
            IrisNet,
            module__hidden_dim=config["training"]["hidden_dim"],
            max_epochs=config["training"]["num_epochs"],
            lr=config["training"]["learning_rate"],
            batch_size=config["training"]["batch_size"],
            device=device,
            # necessary to reinstantiate model each time
            train_split=lambda ds, _: (ds, None),
        )

        param_grid = {
            "module__hidden_dim": config["tuning"]["param_grid"]["hidden_dim"],
            "lr": config["tuning"]["param_grid"]["learning_rate"],
            "max_epochs": config["tuning"]["param_grid"]["num_epochs"],
            "batch_size": config["tuning"]["param_grid"]["batch_size"],
        }

        gs = GridSearchCV(
            estimator=skorch_net,
            param_grid=param_grid,
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
            verbose=2,
            refit=True
        )
        # Skorch expects float32 inputs
        gs.fit(X_train.astype("float32"), y_train.astype("int64"))
        print("Best params:", gs.best_params_)
        print("Best CV accuracy:", gs.best_score_)
        best_model = gs.best_estimator_

    # 4. Evaluate on hold-out test set
    test_acc = best_model.score(
        X_test.astype("float32") if model_choice == "pytorch" else X_test,
        y_test,
    )
    print(f"[{model_choice}] Test accuracy: {test_acc:.4f}")

    # 5. Save the best model artifact
    artifacts_dir = os.path.join(os.path.dirname(__file__), "../artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    if model_choice == "sklearn":
        model_path = os.path.join(artifacts_dir, "rf_best_model.joblib")
        joblib.dump(best_model, model_path)
    elif model_choice == "xgboost":
        model_path = os.path.join(artifacts_dir, "xgb_best_model.joblib")
        joblib.dump(best_model, model_path)
    else:
        model_path = os.path.join(artifacts_dir, "torch_best_model.joblib")
        joblib.dump(best_model, model_path)

    print(f"Saved best model to: {model_path}")


if __name__ == "__main__":
    tune()
