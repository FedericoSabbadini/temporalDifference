"""Multi-seed experiment runner.

Origin: ADAPTED by the run_bias_experiment pattern from the course exercises,
generalized to any algorithm.

This is where the assignment's "media su 3 seed + deviazione standard" is enforced:
every experiment goes through here, so no result can accidentally be single-seed.
"""

import numpy as np
from rl_lab.utils.seeding import set_seed


def run_multi_seed(make_env_fn, train_fn, seeds=(42, 123, 456), **train_kwargs):
    """Run `train_fn` once per seed and stack the episode returns.

    Args:
        make_env_fn: callable() -> env. Called once per seed, so no state leaks
            between runs.
        train_fn: any algorithm in this package. All of them return the episode
            returns at index [1] of their result tuple, which is the contract this
            runner relies on.
        seeds: the 3 seeds required by the assignment.
        **train_kwargs: forwarded to train_fn.

    Returns:
        (all_returns of shape (n_seeds, n_episodes), all_results) — the full result
        tuples are kept because some experiments need value_history from them.
    """
    all_returns = []
    all_results = []

    for seed in seeds:
        set_seed(seed)
        env = make_env_fn()
        env.reset(seed=seed)         # the env RNG, not just numpy/torch
        env.action_space.seed(seed)  # and its own, separate RNG (see run_pong.py)

        result = train_fn(env=env, **train_kwargs)
        all_returns.append(result[1])
        all_results.append(result)
        env.close()

    return np.array(all_returns), all_results
