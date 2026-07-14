"""Reproducibility utilities.

Origin: NEW. Standard pattern (seed numpy + torch).

The assignment requires "chiaramente riproducibile": fixing the random seed
ensures that running the same experiment twice produces identical results.
We seed both numpy (used by tabular algorithms) and torch (used by DQN).
"""

import numpy as np


def set_seed(seed):
    """Set random seed for numpy (and torch if available).

    Called before each experiment run. Combined with env.reset(seed=...),
    this makes every experiment fully reproducible.
    """
    np.random.seed(seed)  # seed numpy's global RNG (used by tabular ε-greedy, etc.)
    try:
        import torch
        torch.manual_seed(seed)  # seed PyTorch CPU RNG (network init, dropout)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # seed ALL GPU RNGs (for multi-GPU setups)
    except ImportError:
        pass  # torch not installed - tabular-only mode (CartPole/Acrobot)
