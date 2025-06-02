# src/utils.py

import os
import yaml
import torch


def load_config(path: str) -> dict:
    """
    Reads a YAML configuration file and returns a dictionary.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device():
    """
    Returns 'cuda' if a GPU is available, else 'cpu'.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
