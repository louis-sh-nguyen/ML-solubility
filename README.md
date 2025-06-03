# ML Solubility Prediction Project

This project aims to predict the solubility of solutes (specifically CO2 initially) in amorphous polymers using machine learning techniques. The project follows a structured approach, starting with model exploration and culminating in an MLOps pipeline.

Data Source: [Predicting the solubility of gases, vapors, and supercritical fluids in amorphous polymers from electron density using convolutional neural networks](https://doi.org/10.1039/D3PY01028G)

## Project Stages

### Stage 1: Model Exploration and Familiarization (Notebooks)

*   **Goal:** Gain hands-on experience with common regression models for solubility prediction.
*   **Activities:**
    *   Explore the provided dataset (`experimental_dataset.csv`, `list_of_polymers.csv`, `list_of_solvents.csv`).
    *   Perform feature engineering and preprocessing.
    *   Implement and evaluate various models within Jupyter notebooks (`notebooks/polymer-prediction.ipynb`):
        *   **Scikit-learn Models:**
            *   Linear Regression (Baseline)
            *   Ridge Regression
            *   Lasso Regression
            *   ElasticNet
            *   K-Nearest Neighbors (KNN) Regressor
            *   Decision Tree Regressor
            *   Random Forest Regressor
            *   Gradient Boosting Regressor
            *   Support Vector Regressor (SVR)
            *   XGBoost Regressor
        *   **PyTorch:**
            *   Implement a basic feed-forward Neural Network.
    *   Compare model performance using appropriate metrics (e.g., RMSE) and cross-validation.

### Stage 2: Advanced Model Development (PyTorch)

*   **Goal:** Develop and train a more sophisticated deep learning model using PyTorch, potentially leveraging architectures suitable for chemical/physical property prediction.
*   **Activities:**
    *   Design a custom PyTorch model architecture (e.g., potentially incorporating graph neural networks if molecular structures are used later, or more complex feed-forward networks).
    *   Implement training loops, validation, and hyperparameter tuning specific to the PyTorch model.
    *   Integrate the PyTorch model training and evaluation into the project structure.

### Stage 3: MLOps Pipeline Implementation

*   **Goal:** Build a complete, reusable machine learning pipeline demonstrating MLOps best practices. The pipeline should be flexible enough to accommodate different model types explored in Stage 1 and Stage 2.
*   **Activities:**
    *   **Data Processing:** Create robust scripts for data loading, validation, cleaning, and preprocessing.
    *   **Training:** Develop a modular training script that can accept different model configurations (e.g., specifying scikit-learn model types or PyTorch model classes).
    *   **Evaluation:** Implement standardized evaluation procedures.
    *   **Experiment Tracking:** Integrate tools like MLflow or Weights & Biases to log parameters, metrics, and artifacts.
    *   **Model Registry:** Store trained models and manage versions.
    *   **Automation/Orchestration:** (Optional/Future) Explore tools like Kubeflow Pipelines, Airflow, or GitHub Actions for automating the pipeline execution.
    *   **Serving:** (Optional/Future) Implement a basic model serving mechanism (e.g., using FastAPI or Flask).

## Setup
1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd ML-solubility
    ```
2.  **Create and activate the Conda environment:**
    Ensure you have Anaconda or Miniconda installed. Then, use the provided `environment.yml` file to create the environment. This file contains all the necessary dependencies.
    ```bash
    conda env create -f environment.yml
    conda activate ml-solubility
    ```
## Project tree
```
├── config/
│   ├── config_sklearn.yaml
│   ├── config_pytorch.yaml
│   └── config_xgboost.yaml
│
├── data/
│   ├── raw/
│   ├── new/
│   └── processed/
│       ├── train.npy
│       ├── train_labels.npy
│       ├── test.npy
│       └── test_labels.npy
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pytorch_model.py         # or iris_net.py (for PyTorch)
│   │   ├── xgb_model.py             # if using XGBoost
│   │   └── torch_wrapper.py         # sklearn‐compatible PyTorch wrapper
│   │
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   └── utils.py
│
├── pipelines/
│   └── training_pipeline.py
│
├── serving/
│   ├── app.py
│   └── schema.py
│
├── scripts/
│   └── run_training.sh
│
├── artifacts/                       # saved model & scaler files
│   ├── iris_model.pt
│   └── scaler.pkl
│
├── requirements.txt
└── README.md
```

## Usage

### Installation as Development Package

The recommended way to use this code is to install it as a development package:

1. From the project root directory, install the package in editable mode:
   ```bash
   pip install -e .
   ```

2. This creates an `.egg-info` directory that helps Python find your package. You don't need to reinstall the package when making code changes - they will be reflected immediately.

3. Now you can import modules from anywhere:
   ```python
   from src.data.load_data import load_raw_data
   from src.models.torch_wrapper import TorchRegressor
   ```

### Running the Models

You can train models using the following command:

```bash
# Run from the project root directory
python -m src.train
```

Specify which model to use by setting the `MODEL_CHOICE` environment variable:

```bash
# Windows
set MODEL_CHOICE=sklearn
python -m src.train

# Mac/Linux
MODEL_CHOICE=sklearn python -m src.train
```

Available model choices:
- `sklearn` (RandomForest)
- `xgboost` (XGBoost)
- `pytorch` (PyTorch neural network, default)