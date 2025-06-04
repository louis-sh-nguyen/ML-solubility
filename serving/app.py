# serving/app.py

import os
import uvicorn
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from serving.schema import IrisFeatures
from src.utils import load_config

app = FastAPI(title="Iris Classification API")

# 1. Determine which config to load
model_choice = os.environ.get("MODEL_CHOICE", "pytorch")  # "sklearn", "xgboost", or "pytorch"

if model_choice == "sklearn":
    config_path = os.path.join(os.path.dirname(__file__), "../config/config_sklearn.yaml")
elif model_choice == "xgboost":
    config_path = os.path.join(os.path.dirname(__file__), "../config/config_xgboost.yaml")
else:
    config_path = os.path.join(os.path.dirname(__file__), "../config/config_pytorch.yaml")

config = load_config(config_path)

# 2. Load scaler and model artefacts
artifacts_dir = os.path.join(os.path.dirname(__file__), "../artifacts")
scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
model = None

try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    raise RuntimeError(f"Scaler not found at {scaler_path}. Run training first.")

if model_choice == "sklearn":
    model_path = os.path.join(artifacts_dir, "rf_model.joblib")
elif model_choice == "xgboost":
    model_path = os.path.join(artifacts_dir, "xgb_model.joblib")
else:
    model_path = os.path.join(artifacts_dir, "torch_model.pt")

try:
    import torch  # Import torch for PyTorch model loading
    model = torch.load(model_path)
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {model_path}. Run training first.")


@app.post("/predict")
def predict(features: IrisFeatures):
    """
    Accepts IrisFeatures JSON, returns predicted class (0, 1, or 2).
    """
    # 3. Convert input to numpy array
    X = np.array([[features.sepal_length,
                   features.sepal_width,
                   features.petal_length,
                   features.petal_width]])

    # 4. Scale features
    try:
        X_scaled = scaler.transform(X)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=500, detail=f"Error scaling input: {e}")

    # 5. Predict
    try:
        pred_label = int(model.predict(X_scaled)[0])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input for prediction: {e}")
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Type error during prediction: {e}")
    except joblib.exceptions.JoblibException as e:
        raise HTTPException(status_code=500, detail=f"Model-related error during prediction: {e}")

    return {"predicted_class": pred_label}


if __name__ == "__main__":
    # Run with: MODEL_CHOICE=<sklearn|xgboost|pytorch> python serving/app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
