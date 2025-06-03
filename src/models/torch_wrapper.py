# src/models/torch_wrapper.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
import numpy as np
import scipy.sparse as sp


class TorchRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn‐compatible wrapper around a simple PyTorch feedforward regressor.
    Accepts sparse or dense input; outputs a single continuous value.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 16,
                 learning_rate: float = 0.001,
                 num_epochs: int = 20,
                 batch_size: int = 16,
                 random_seed: int = 42):
        """
        Parameters:
        -----------
        input_dim : int
            Number of input features.
        hidden_dim : int
            Size of the hidden layer.
        learning_rate : float
        num_epochs : int
        batch_size : int
        random_seed : int
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()

    def _build_model(self):
        """
        Construct a simple feedforward network:
            input_dim → hidden_dim → ReLU → output_dim=1
        """
        torch.manual_seed(self.random_seed)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _ensure_dense(self, X):
        """
        Convert sparse or object-dtype arrays to a dense numpy array.
        """
        # If it's a scipy sparse matrix, convert to dense
        if sp.issparse(X):
            return X.toarray()

        # If it's an object-dtype numpy array (e.g. loaded .npy wrapping a sparse), extract and convert
        if isinstance(X, np.ndarray) and X.dtype == object:
            # assume the array contains one sparse matrix or list of rows
            element = X.item() if X.ndim == 0 else None
            if sp.issparse(element):
                return element.toarray()
            try:
                # If each row is a sparse vector
                dense_rows = []
                for row in X:
                    if sp.issparse(row):
                        dense_rows.append(row.toarray().ravel())
                    else:
                        dense_rows.append(np.asarray(row))
                return np.vstack(dense_rows)
            except Exception:
                pass

        # Otherwise assume it's already a dense numpy array
        return np.asarray(X)

    def fit(self, X, y):
        """
        Train the PyTorch model.

        X : array-like of shape (n_samples, input_dim)
            May be dense numpy array, scipy sparse matrix, or object-dtype array containing a sparse.
        y : array-like of shape (n_samples,)
        """
        # Convert to dense numpy if needed
        X = self._ensure_dense(X)
        y = np.asarray(y)

        # Rebuild model to reset weights on each fit
        self._build_model()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.num_epochs):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, X):
        """
        Generate predictions for X.

        X : array-like of shape (n_samples, input_dim)
            May be dense numpy array, scipy sparse matrix, or object-dtype array containing a sparse.
        Returns: numpy array of shape (n_samples,)
        """
        # Convert to dense numpy if needed
        X = self._ensure_dense(X)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().reshape(-1)
        return preds

    def score(self, X, y):
        """
        Returns R² score to be compatible with sklearn.

        X : array-like of shape (n_samples, input_dim)
        y : array-like of shape (n_samples,)
        """
        preds = self.predict(X)
        return r2_score(y, preds)
