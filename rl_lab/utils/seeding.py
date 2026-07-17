"""Reproducibility: one seed call for numpy and torch.

Origin: NEW. Combined with env.reset(seed=...) in the runner, this is what makes the
experiments repeatable, as the assignment requires.
"""

import numpy as np
import torch

def set_seed(seed):
    """Seed numpy and, if installed, torch (CPU and all GPUs)."""
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # tabular-only mode: torch is not needed for CartPole/Acrobot
