# dashboard/streamlit_app.py

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Disable PIL decompression bomb check
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
st.set_page_config(
    page_title="Solubility EDA & Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Solubility: EDA & Model Monitoring Dashboard")

# ─── 1. Load processed data ───────────────────────────────────────────────────

@st.cache_data
def load_processed_data():
    """
    Loads X_train, y_train, X_val, y_val, X_test, y_test from data/processed.
    Returns a dict of numpy arrays.
    """
    proc_dir = Path(__file__).parent.parent / "data" / "processed"
    data = {}
    for split in ["train", "val", "test"]:
        data[f"X_{split}"] = np.load(proc_dir / f"X_{split}.npy", allow_pickle=True)
        data[f"y_{split}"] = np.load(proc_dir / f"y_{split}.npy", allow_pickle=True)
    # Load the preprocessor to inspect feature names if needed:
    data["preprocessor"] = joblib.load(proc_dir / "preprocessor.joblib")
    return data

data_dict = load_processed_data()

# Convert to dense arrays if they are sparse matrices
for key in ["X_train", "X_val", "X_test"]:
    # First check if we need to extract the sparse matrix
    if isinstance(data_dict[key], np.ndarray) and data_dict[key].size == 1:
        try:
            # Extract the sparse matrix from the array
            data_dict[key] = data_dict[key].item()
        except (ValueError, AttributeError):
            pass
    
    # Now convert to dense if it's a sparse matrix
    if hasattr(data_dict[key], "toarray"):
        data_dict[key] = data_dict[key].toarray()

X_train, y_train = data_dict["X_train"], data_dict["y_train"]
X_val,   y_val   = data_dict["X_val"],   data_dict["y_val"]
X_test,  y_test  = data_dict["X_test"],  data_dict["y_test"]

        
print(f"Loaded data shapes: "
      f"X_train: {X_train.shape}, y_train: {y_train.shape}, "
      f"X_val: {X_val.shape}, y_val: {y_val.shape}, "
      f"X_test: {X_test.shape}, y_test: {y_test.shape}")

st.sidebar.header("Data Overview")
st.sidebar.write(f"Training examples: {X_train.shape[0]}")
st.sidebar.write(f"Validation examples: {X_val.shape[0]}")
st.sidebar.write(f"Test examples: {X_test.shape[0]}")
st.sidebar.write(f"Number of features: {X_train.shape[1]}")

# Display raw shapes
with st.expander("Show data shapes"):
    st.write("X_train shape:", X_train.shape)
    st.write("y_train shape:", y_train.shape)
    st.write("X_val shape:", X_val.shape)
    st.write("y_val shape:", y_val.shape)
    st.write("X_test shape:", X_test.shape)
    st.write("y_test shape:", y_test.shape)

# ─── 2. Exploratory Data Analysis ─────────────────────────────────────────────

st.header("1. Exploratory Data Analysis (EDA)")

# If you know feature names, you can retrieve them from preprocessor:
preprocessor = data_dict["preprocessor"]
# For simplicity, let’s generate placeholder feature names:
num_features = preprocessor.transformers_[0][2]  # numeric column names
cat_features = preprocessor.transformers_[1][2]  # categorical column names
feature_names = list(num_features) + list(preprocessor.transformers_[1][1].
                                           named_steps["onehot"].
                                           get_feature_names_out(cat_features))

# Convert X_train back to DataFrame for plotting
df_train = pd.DataFrame(X_train, columns=feature_names)
df_train["wa"] = y_train

# 2a. Show histograms of each feature in pages of up to 12 at a time
if st.checkbox("Show feature histograms"):
    st.subheader("Feature distributions (Train set)")
    max_per_page = 12
    total_feats = len(feature_names)
    pages = (total_feats + max_per_page - 1) // max_per_page

    for page in range(pages):
        start_idx = page * max_per_page
        end_idx = min(start_idx + max_per_page, total_feats)
        subset_feats = feature_names[start_idx:end_idx]

        n_sub = len(subset_feats)
        n_cols = 3
        n_rows = (n_sub + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()

        for idx, feat in enumerate(subset_feats):
            axes[idx].hist(df_train[feat], bins=30, color="skyblue", edgecolor="black")
            axes[idx].set_title(feat)

        # Remove any unused subplots
        for j in range(n_sub, len(axes)):
            fig.delaxes(axes[j])

        st.pyplot(fig)
        plt.close(fig)

# 2b. Scatter plot: predicted vs. true for test set (if model exists)
st.subheader("Model Predictions vs. True (if a trained model is found)")
artifacts_dir = Path(__file__).parent.parent / "artifacts"
# model_choice = os.environ.get("MODEL_CHOICE", "sklearn")
# Allow user to select which model to display
model_choice = st.selectbox(
    "Choose model for prediction plot:",
    options=["sklearn", "xgboost", "pytorch"],
    index=["sklearn", "xgboost", "pytorch"].index(os.environ.get("MODEL_CHOICE", "sklearn"))
)

model_path_map = {
    "sklearn": "rf_regressor.joblib",
    "xgboost": "xgb_regressor.joblib",
    "pytorch": "torch_regressor.joblib"
}
model_filename = model_path_map.get(model_choice, "rf_regressor.joblib")
model_path = artifacts_dir / model_filename

if model_path.exists():
    model = joblib.load(model_path)
    y_test_pred = model.predict(X_test.astype("float32") 
                                if model_choice == "pytorch" else X_test)

    df_preds = pd.DataFrame({
        "True wa": y_test,
        "Predicted wa": y_test_pred
    })
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(
        df_preds["True wa"], df_preds["Predicted wa"],
        color="teal", alpha=0.6, edgecolor="w", s=50
    )
    ax2.plot(
        [df_preds["True wa"].min(), df_preds["True wa"].max()],
        [df_preds["True wa"].min(), df_preds["True wa"].max()],
        color="red", linestyle="--"
    )
    ax2.set_xlabel("True wa")
    ax2.set_ylabel("Predicted wa")
    ax2.set_title("Predicted vs. True (Test Set)")
    st.pyplot(fig2)
else:
    st.warning(f"No model file found at `{model_path}`. Train a model first.")

# ─── 3. Monitoring: Plot metrics over time ─────────────────────────────────────

st.header("2. Model Monitoring Over Time")

st.write(
    "Below, we read the MLflow `metrics.csv` (if available) or "
    "an artifacts‐generated CSV to show how validation/test MSE & R² have changed."
)

# Suppose training logs wrote out a CSV of metrics in `artifacts/metrics_history.csv`
metrics_file = artifacts_dir / "metrics_history.csv"
if metrics_file.exists():
    df_metrics = pd.read_csv(metrics_file, parse_dates=["timestamp"])
    st.subheader("Historical Metrics")
    st.dataframe(df_metrics.tail(10))

    # Line chart for validation & test metrics
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(df_metrics["timestamp"], df_metrics["val_mse"], label="Val MSE")
    ax3.plot(df_metrics["timestamp"], df_metrics["val_r2"], label="Val R²")
    ax3.plot(df_metrics["timestamp"], df_metrics["test_mse"], label="Test MSE")
    ax3.plot(df_metrics["timestamp"], df_metrics["test_r2"], label="Test R²")
    ax3.set_ylabel("Metric value")
    ax3.set_xlabel("Timestamp")
    ax3.legend()
    st.pyplot(fig3)
else:
    st.info(f"No `metrics_history.csv` found at `{metrics_file}`. You can generate one by logging metrics to CSV in train.py.")

st.markdown("---")
st.write("© 2025 Louis Nguyen. All rights reserved.")
