"""Multi-seed experiment runner.

Origin: INSPIRED by the run_bias_experiment pattern from the course exercises
(Esercitazione_RL). The course did 100-200 runs for bias analysis.
Generalized here to work with any algorithm and return arrays for plotting.

The assignment requires: "Ogni risultato deve essere media su 3 seed diversi,
accompagnato da deviazione standard (o bande)." This module enforces that
by running any algorithm across multiple seeds and collecting results.
"""

import numpy as np
from rl_lab.utils.seeding import set_seed


def run_multi_seed(make_env_fn, train_fn, seeds=(42, 123, 456),
                   **train_kwargs):
    """Run a training function with multiple seeds, collect episode returns.

    This is the core of the experimental methodology:
    1. For each seed: create a fresh env, set the seed, train, collect returns
    2. Stack all returns into a (n_seeds, n_episodes) array
    3. The plotting functions compute mean ± std from this array

    Args:
        make_env_fn: callable() - Gymnasium env. Called once per seed
                     (fresh env each time to avoid cross-contamination).
        train_fn: algorithm function. Must accept env= as first kwarg and
                  return a tuple where index [1] is the episode returns list.
                  Works with: q_learning, sarsa (returns Q, returns, lengths)
                  and mc_prediction, td0_prediction (returns V, returns).
        seeds: sequence of random seeds (default: 3 seeds per assignment).
        **train_kwargs: forwarded to train_fn (episodes, alpha, gamma, etc.).

    Returns:
        all_returns: np.array of shape (n_seeds, n_episodes).
        all_results: list of full result tuples from each seed.
    """
    all_returns = []  # will become a (n_seeds, n_episodes) array containing all the REWARDS for each seed+episode
    all_results = []  # full result tuples for each seed (Q-tables, V, etc.)

    for seed in seeds:
        set_seed(seed)  # seed numpy and torch for this run
        env = make_env_fn()  # fresh environment (no state from previous seed)
        env.reset(seed=seed)  # seed the env's internal RNG (Gymnasium API)

        # Run the algorithm; all algorithms return episode_returns at index [1]:
        # q_learning - (Q, returns, lengths)
        # mc_prediction / td0_prediction - (V, returns, value_history)
        result = train_fn(env=env, **train_kwargs)
        all_returns.append(result[1])  # index [1] = episode_returns list
        all_results.append(result)
        env.close()  # free resources (important for Atari envs)

    # Stack into (n_seeds, n_episodes) array for easy mean/std computation
    return np.array(all_returns), all_results
