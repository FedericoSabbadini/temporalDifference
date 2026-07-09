"""Reproducibility utilities."""

import numpy as np


def set_seed(seed):
    """Set random seed for numpy (and torch if available)."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
