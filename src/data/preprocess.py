# src/data/preprocess.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, config: dict):
    """
    - Splits DataFrame into features/labels.
    - Applies standard scaling.
    - Splits into train/test according to config.
    - Returns numpy arrays and fitted scaler.
    """
    # Separate features and target
    X = df.drop(columns=['target']).values
    y = df['target'].values

    # Train/test split
    test_size = config['training']['test_size']
    random_seed = config['training']['random_seed']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
