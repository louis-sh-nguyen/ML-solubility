# src/data/load_data.py
import numpy as np
import os
from pathlib import Path
import pandas as pd


def load_raw_data(config: dict):
    """
    1. Read experimental_dataset.csv, list_of_polymers.csv, list_of_solvents.csv,
       polymer_mass.csv, solvent_mass.csv, and solvent_macro_features.csv.
    2. Merge them into a single DataFrame (full_df).
    3. Return full_df.
    
    Expects:
      DATA_DIR = Path('..') / 'data'
      config may supply RANDOM_SEED or not, but is not used here.
    """
    # Base directory for CSVs (one level up from src/)
    DATA_DIR = Path(__file__).parent.parent / 'data'
    
    # 1. Read all CSVs
    df_exp     = pd.read_csv(DATA_DIR / 'experimental_dataset.csv')
    df_poly    = pd.read_csv(DATA_DIR / 'list_of_polymers.csv')
    df_solv    = pd.read_csv(DATA_DIR / 'list_of_solvents.csv')
    df_poly_mw = pd.read_csv(DATA_DIR / 'polymer_mass.csv')
    df_solv_mw = pd.read_csv(DATA_DIR / 'solvent_mass.csv')
    df_solv_mac= pd.read_csv(DATA_DIR / 'solvent_macro_features.csv')
    
    # 2. Merge into one DataFrame
    full_df = (
        df_exp
        .merge(df_poly,    on='polymer_id', how='left')
        .merge(df_solv,    on='solvent_id', how='left')
        .merge(df_poly_mw, on='polymer_id', how='left')
        .merge(df_solv_mw, on='solvent_id', how='left')
        .merge(df_solv_mac,on='solvent_id', how='left')
    )
    
    # Optional: print shapes for sanity
    print("experimental_dataset:", df_exp.shape)
    print("full_df after merging:", full_df.shape)
    
    return full_df



# def save_processed_data(X_train: np.ndarray, y_train: np.ndarray,
#                         X_test: np.ndarray, y_test: np.ndarray, config: dict):
#     """
#     Saves numpy arrays to processed/ directory.
#     """
#     processed_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
#     os.makedirs(processed_dir, exist_ok=True)
#     np.save(os.path.join(processed_dir, "train.npy"), X_train)
#     np.save(os.path.join(processed_dir, "train_labels.npy"), y_train)
#     np.save(os.path.join(processed_dir, "test.npy"), X_test)
#     np.save(os.path.join(processed_dir, "test_labels.npy"), y_test)
