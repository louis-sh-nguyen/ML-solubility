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

# Create tabs for better organization
tab1, tab2 = st.tabs(["Exploratory Data Analysis (EDA)", "Model Monitoring"])

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

# Get feature names
preprocessor = data_dict["preprocessor"]
num_features = preprocessor.transformers_[0][2]  # numeric column names
cat_features = preprocessor.transformers_[1][2]  # categorical column names
feature_names = list(num_features) + list(preprocessor.transformers_[1][1].
                                           named_steps["onehot"].
                                           get_feature_names_out(cat_features))

# Convert X_train back to DataFrame for plotting
df_train = pd.DataFrame(X_train, columns=feature_names)
df_train["wa"] = y_train

# Define the artifacts directory for later
artifacts_dir = Path(__file__).parent.parent / "artifacts"

# ─── 2. Exploratory Data Analysis Tab ────────────────────────────────────────

with tab1:
    st.header("Exploratory Data Analysis (EDA)")
        # Display raw shapes
    with st.expander("Data shapes"):
        st.write("X_train shape:", X_train.shape)
        st.write("y_train shape:", y_train.shape)
        st.write("X_val shape:", X_val.shape)
        st.write("y_val shape:", y_val.shape)
        st.write("X_test shape:", X_test.shape)
        st.write("y_test shape:", y_test.shape)
        
    # 2a. Show histograms by feature type
    with st.expander("Feature histograms"):
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
        
        if feature_type == "Numeric Features":
            # Filter numeric features by name
            search_term = st.text_input("Filter features by name:")
            if search_term:
                features_to_plot = [f for f in numeric_features if search_term.lower() in f.lower()]
            
            # Show histograms for numeric features
            st.subheader("Numeric feature distributions")
            
            # Create a grid of histograms
            num_cols = st.number_input("Number of columns in histogram grid:", 1, 5, 3)
            num_rows = (len(features_to_plot) + num_cols - 1) // num_cols
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, max(5, num_rows * 3)))
            axes = axes.flatten() if num_rows > 1 else [axes]
            
            for i, feature in enumerate(features_to_plot):
                ax = axes[i]
                ax.hist(df_train[feature], bins=30, color="skyblue", edgecolor="black")
                ax.set_title(feature)
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
            
            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
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
    
    # Add correlation heatmap for numeric features
    with st.expander("Show correlation heatmap for numeric features"):
        st.subheader("Correlation Heatmap")

        # Calculate correlation matrix for numeric features only
        corr_matrix = df_train[numeric_features + ["wa"]].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = plt.cm.coolwarm
        
        # Plot heatmap with annotations
        heatmap = ax.pcolormesh(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(heatmap)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(corr_matrix.columns)) + 0.5)
        ax.set_yticks(np.arange(len(corr_matrix.index)) + 0.5)
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr_matrix.index)
        
        # Add correlation values as text annotations
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                ax.text(j + 0.5, i + 0.5, f'{corr_matrix.iloc[i, j]:.2f}',
                        ha='center', va='center', 
                        color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Add Train vs Test Distribution Comparison
    with st.expander("Compare Train vs Test Distributions"):
        st.subheader("Train vs Test Distribution Comparison")
        
        # Convert X_test to DataFrame
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test["wa"] = y_test
        
        # Feature selection
        feature_to_compare = st.selectbox(
            "Select feature to compare:", 
            options=numeric_features + ["wa"],
            index=0
        )
        
        # Create distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        ax.hist(df_train[feature_to_compare], bins=30, alpha=0.6, 
                color='blue', label='Train')
        ax.hist(df_test[feature_to_compare], bins=30, alpha=0.6, 
                color='red', label='Test')
        
        # Add labels and title
        ax.set_xlabel(feature_to_compare)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feature_to_compare} - Train vs Test')
        ax.legend()
        
        # Show KS test results for distribution comparison
        from scipy.stats import ks_2samp
        ks_stat, p_val = ks_2samp(df_train[feature_to_compare], df_test[feature_to_compare])
        st.write(f"Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={p_val:.4f}")
        
        if p_val < 0.05:
            st.warning("The distributions are significantly different (p < 0.05)")
        else:
            st.success("The distributions are not significantly different (p >= 0.05)")
        
        st.pyplot(fig)
        plt.close(fig)
        

# ─── 3. Model Monitoring Tab ─────────────────────────────────────────────────

with tab2:
    st.header("Model Monitoring")
    # 2b. Scatter plot: predicted vs. true for test set (if model exists)
    st.subheader("Model Predictions vs. True")

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
        
        # Add Feature Importance Bar Chart (if supported by model)
        if st.checkbox("Show Feature Importance"):
            st.subheader("Feature Importance")
            
            # Check if model has feature_importances_ attribute
            if hasattr(model, 'feature_importances_'):
                # Get feature importances
                importances = model.feature_importances_
                
                # Create DataFrame for plotting
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Plot top N features
                top_n = st.slider("Number of top features to show:", 10, 50, 20, 
                                key="importance_slider")
                top_features = importance_df.head(top_n)
                
                fig_imp, ax_imp = plt.subplots(figsize=(10, max(6, top_n//3)))
                bars = ax_imp.barh(top_features['Feature'], top_features['Importance'], 
                            color='teal')
                
                # Add values on bars
                for bar in bars:
                    width = bar.get_width()
                    ax_imp.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                            f'{width:.4f}', ha='left', va='center')
                
                ax_imp.set_xlabel('Importance')
                ax_imp.set_title(f'Top {top_n} Features by Importance')
                plt.tight_layout()
                st.pyplot(fig_imp)
                plt.close(fig_imp)
                
                # Offer download of importance data
                csv = importance_df.to_csv(index=False)
                st.download_button(
                    "Download feature importance data as CSV",
                    data=csv,
                    file_name="feature_importance.csv",
                    mime="text/csv"
                )
            elif hasattr(model, 'coef_'):
                # For linear models
                importances = np.abs(model.coef_)
                
                # Create DataFrame for plotting
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient (abs)': importances
                }).sort_values('Coefficient (abs)', ascending=False)
                
                # Plot top N features
                top_n = st.slider("Number of top features to show:", 10, 50, 20,
                                key="importance_slider")
                top_features = importance_df.head(top_n)
                
                fig_imp, ax_imp = plt.subplots(figsize=(10, max(6, top_n//3)))
                bars = ax_imp.barh(top_features['Feature'], top_features['Coefficient (abs)'], 
                            color='teal')
                
                ax_imp.set_xlabel('Absolute Coefficient Value')
                ax_imp.set_title(f'Top {top_n} Features by Coefficient Magnitude')
                plt.tight_layout()
                st.pyplot(fig_imp)
                plt.close(fig_imp)
            else:
                st.info("This model type doesn't provide built-in feature importance information.")
    else:
        st.warning(f"No model file found at `{model_path}`. Train a model first.")

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
