# src/data/preprocess.py

import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(full_df, config: dict):
    """
    1. Split full_df into train/validation/test (15% hold-out for test; then 15% of remaining for validation).
    2. Identify numeric vs. categorical columns (exclude IDs and target).
    3. Build a ColumnTransformer that scales numeric features and one-hot encodes categoricals.
    4. Fit the transformer on train_df; transform train/val/test.
    5. Export processed NumPy arrays to data/processed/.
       - X_train.npy, y_train.npy
       - X_val.npy,   y_val.npy
       - X_test.npy,  y_test.npy
    6. Return (X_train, y_train), (X_val, y_val), (X_test, y_test).
    """

    # a) Constants & paths
    RANDOM_SEED = config['training']['random_seed']
    TARGET_COL  = config['data'].get('target_col', 'target')  
    # (make sure your config defines data.target_col, or change 'target' to your actual target column name)
    
    # b) Split into train+val vs. test
    train_val_df, test_df = train_test_split(
        full_df,
        test_size=0.15,
        random_state=RANDOM_SEED
    )
    # c) Split train_val into train vs. val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.15,
        random_state=RANDOM_SEED
    )
    print(f"Split sizes → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    
    # d) Separate features and targets
    X_train_df, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    X_val_df,   y_val   = val_df.drop(columns=[TARGET_COL]),   val_df[TARGET_COL]
    X_test_df,  y_test  = test_df.drop(columns=[TARGET_COL]),  test_df[TARGET_COL]
    
    # e) Identify numeric vs. categorical columns
    #    Exclude IDs (polymer_id, solvent_id) and the target itself
    num_cols = [
        c for c in full_df.columns 
        if full_df[c].dtype != 'object' 
           and c not in ['polymer_id', 'solvent_id', TARGET_COL]
    ]
    cat_cols = [
        c for c in full_df.columns 
        if full_df[c].dtype == 'object'
    ]
    
    # f) Build ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])
    
    # g) Fit on train, transform train/val/test
    X_train = preprocessor.fit_transform(X_train_df)
    X_val   = preprocessor.transform(X_val_df)
    X_test  = preprocessor.transform(X_test_df)
    
    # Convert y to NumPy arrays
    y_train = y_train.values
    y_val   = y_val.values
    y_test  = y_test.values
    
    # h) Save processed arrays under data/processed/
    processed_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(processed_dir / 'X_train.npy', X_train)
    np.save(processed_dir / 'y_train.npy', y_train)
    np.save(processed_dir / 'X_val.npy',   X_val)
    np.save(processed_dir / 'y_val.npy',   y_val)
    np.save(processed_dir / 'X_test.npy',  X_test)
    np.save(processed_dir / 'y_test.npy',  y_test)
    
    # Optionally: save the fitted preprocessor itself for later use
    import joblib
    joblib.dump(preprocessor, processed_dir / 'preprocessor.joblib')
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
