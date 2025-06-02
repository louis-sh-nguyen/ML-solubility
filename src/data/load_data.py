# src/data/load_data.py

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def load_raw_data(config: dict):
    """
    Loads raw data. For this example, we simulate “raw” by using sklearn’s Iris dataset.
    If new CSV files appear in data/new/, they’ll be concatenated.
    """
    # 1. Load base Iris dataset
    iris = load_iris(as_frame=True)
    df_base = iris.frame.copy()
    df_base['target'] = iris.target

    # 2. Check for any new CSVs in data/new/
    new_dir = os.path.join(os.path.dirname(__file__), "../../data/new")
    all_dfs = [df_base]

    if os.path.isdir(new_dir):
        for fname in os.listdir(new_dir):
            if fname.endswith(".csv"):
                path = os.path.join(new_dir, fname)
                df_new = pd.read_csv(path)
                # Expect same columns: features + "target" column
                all_dfs.append(df_new)

    df_full = pd.concat(all_dfs, ignore_index=True)
    return df_full


def save_processed_data(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray, config: dict):
    """
    Saves numpy arrays to processed/ directory.
    """
    processed_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, "train.npy"), X_train)
    np.save(os.path.join(processed_dir, "train_labels.npy"), y_train)
    np.save(os.path.join(processed_dir, "test.npy"), X_test)
    np.save(os.path.join(processed_dir, "test_labels.npy"), y_test)
