# src/models/pytorch_model.py

import torch
import torch.nn as nn


class IrisNet(nn.Module):
    """
    Simple feedforward neural network for iris classification.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 3):
        super(IrisNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
