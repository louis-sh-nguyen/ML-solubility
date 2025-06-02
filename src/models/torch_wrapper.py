# src/models/torch_wrapper.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin


class IrisTorchClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for a simple PyTorch neural network on the Iris dataset.
    """

    def __init__(self,
                 hidden_dim=16,
                 learning_rate=0.001,
                 num_epochs=20,
                 batch_size=16,
                 random_seed=42):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()

    def _build_model(self):
        torch.manual_seed(self.random_seed)
        # Assume 4 input features (Iris) and 3 output classes
        self.model = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3)
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, X, y):
        """
        X: numpy array of shape (n_samples, 4)
        y: numpy array of shape (n_samples,)
        """
        self._build_model()  # Rebuild to reset weights each fit call
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.num_epochs):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    def score(self, X, y):
        """
        Returns accuracy on (X, y).
        """
        preds = self.predict(X)
        return (preds == y).mean()
