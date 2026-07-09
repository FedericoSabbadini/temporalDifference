"""Multi-seed experiment runner.

Runs an algorithm N times with different seeds and collects results.
Every reported result in the project must be the mean over 3 seeds
with standard-deviation bands — this module enforces that.

Generalizes the run_bias_experiment pattern from the course exercises.
"""

import numpy as np
from rl_lab.utils.seeding import set_seed


def run_multi_seed(make_env_fn, train_fn, seeds=(42, 123, 456),
                   **train_kwargs):
    """Run a training function with multiple seeds, collect episode returns.

    Args:
        make_env_fn: callable() → Gymnasium env. Called once per seed.
        train_fn: algorithm function with signature train_fn(env=..., **kwargs).
                  Must return a tuple where index [1] is the episode returns list.
        seeds: sequence of random seeds.
        **train_kwargs: forwarded to train_fn.

    Returns:
        all_returns: np.array of shape (n_seeds, n_episodes).
        all_results: list of full result tuples from each seed.
    """
    all_returns = []
    all_results = []

    for seed in seeds:
        set_seed(seed)
        env = make_env_fn()
        env.reset(seed=seed)

        result = train_fn(env=env, **train_kwargs)
        all_returns.append(result[1])
        all_results.append(result)
        env.close()

    return np.array(all_returns), all_results
