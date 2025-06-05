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

# 2a. Show histograms by feature type
if st.checkbox("Show feature histograms"):
    st.subheader("Feature distributions (Train set)")
    
    # Separate numeric and categorical features
    numeric_features = list(num_features)
    encoded_features = [f for f in feature_names if f not in numeric_features]
    
    # Add feature type selector
    feature_type = st.radio(
        "Select feature type to display:",
        ["Numeric Features", "One-Hot Encoded Features"]
    )
    
    features_to_plot = numeric_features if feature_type == "Numeric Features" else encoded_features
    
    # For one-hot encoded features
    if feature_type == "One-Hot Encoded Features":
        search_term = st.text_input("Filter features by name:")
        if search_term:
            features_to_plot = [f for f in encoded_features if search_term.lower() in f.lower()]
        
        # Show occurrence percentages instead of histograms
        st.subheader("One-hot encoded feature occurrence rates")
        
        # Calculate occurrence rates for each feature
        occurrence_rates = {feat: df_train[feat].mean() * 100 for feat in features_to_plot}
        
        # Sort by occurrence rate
        sorted_features = sorted(occurrence_rates.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N features to avoid overcrowding
        top_n = st.slider("Number of top features to show:", 10, 100, 30)
        top_features = sorted_features[:top_n]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, max(5, top_n//4)))
        feature_names = [f[0] for f in top_features]
        occurrence_values = [f[1] for f in top_features]
        
        ax.barh(feature_names, occurrence_values, color="teal")
        ax.set_xlabel("Occurrence Rate (%)")
        ax.set_title(f"Top {top_n} One-Hot Encoded Features by Occurrence Rate")
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Offer a PCA visualization option
        if st.checkbox("Show PCA visualization of one-hot encoded features"):
            from sklearn.decomposition import PCA
            
            # Create PCA model
            n_components = st.slider("Number of PCA components:", 2, 10, 2)
            pca = PCA(n_components=n_components)
            
            # Apply PCA to one-hot encoded features only
            onehot_data = df_train[encoded_features]
            pca_result = pca.fit_transform(onehot_data)
            
            # Show explained variance
            explained_variance = pca.explained_variance_ratio_
            st.write(f"Explained variance by components: {[f'{var:.2%}' for var in explained_variance]}")
            
            # Plot first two components
            fig, ax = plt.subplots(figsize=(10, 8))
            sc = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                        c=df_train["wa"], cmap="viridis", alpha=0.6)
            plt.colorbar(sc, label="Target (wa)")
            ax.set_xlabel(f"PC1 ({explained_variance[0]:.2%} variance)")
            ax.set_ylabel(f"PC2 ({explained_variance[1]:.2%} variance)")
            ax.set_title("PCA of One-Hot Encoded Features")
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
