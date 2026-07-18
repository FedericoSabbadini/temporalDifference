"""Multi-seed experiment runner.

Origin: ADAPTED by the run_bias_experiment pattern from the course exercises,
generalized to any algorithm.

This is where the assignment's "media su 3 seed + deviazione standard" is enforced:
every experiment goes through here, so no result can accidentally be single-seed.
`run_multi_seed` serves the tabular algorithms; `run_dqn_multi_seed` is its step-based
counterpart for the DQN, which cannot reuse the same code (see its docstring).
"""

import os
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
    all_returns = [] # array of shape (n_seeds, n_episodes), the episode returns for each seed.
    all_results = [] # list of the full result tuples for each seed, because some experiments need value_history from them.

    for seed in seeds:
        set_seed(seed)
        env = make_env_fn()
        env.reset(seed=seed)          # set the env's RNG for reproducibility.
        env.action_space.seed(seed)   # set the action space's RNG for reproducibility (e.g. for stochastic policies as random one).

        result = train_fn(env=env, **train_kwargs)
        all_returns.append(result[1]) # the episode returns are at index 1 of the result tuple, by contract.
        all_results.append(result)
        env.close()

    return np.array(all_returns), all_results


def run_dqn_multi_seed(make_env_fn, label, seeds=(42, 123, 456), device="cpu",
                       total_steps=200_000, ckpt_dir="results/pong", **dqn_kwargs):
    """Train one DQN configuration on all seeds, save checkpoints, return the curves.

    The DQN counterpart of `run_multi_seed`. It cannot reuse that function because the
    DQN loop is step-based: each seed ends up with a DIFFERENT number of episodes (an
    agent that learns rallies longer plays fewer games in the same step budget), so the
    rows are padded with NaN to form a rectangular array. The plotting takes a plain
    mean, so a curve is drawn only up to the shortest seed and every plotted point is a
    genuine multi-seed mean.

    Args:
        make_env_fn: callable() -> env. Called once per seed, so no state leaks.
        label: name of the configuration; also the checkpoint subfolder.
        seeds: the 3 seeds required by the assignment.
        device: "cuda" or "cpu", forwarded to train_dqn.
        total_steps: environment-step budget per seed.
        ckpt_dir: base directory for the per-seed model checkpoints.
        **dqn_kwargs: ablation flags and hyperparameters forwarded to train_dqn.

    Returns:
        padded returns of shape (n_seeds, max_n_episodes), NaN-padded on the right.
    """
    # Lazy import so the tabular runs, which also import this module, do not pull in the
    # deep stack (torch networks, replay buffer) they never use.
    import torch
    from rl_lab.algorithms.deep.dqn import train_dqn

    print(f"\n--- {label} ---")
    all_returns = []

    for seed in seeds:
        print(f"  Seed {seed}...")
        set_seed(seed)
        env = make_env_fn()
        env.reset(seed=seed)
        env.action_space.seed(seed)

        returns, lengths, model = train_dqn(
            env=env,
            total_steps=total_steps,
            device=device,
            log_interval=20,
            **dqn_kwargs,
        )
        all_returns.append(returns)
        env.close()

        seed_ckpt_dir = f"{ckpt_dir}/{label.replace(' ', '_')}"
        os.makedirs(seed_ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{seed_ckpt_dir}/model_seed{seed}.pt")

    max_len = max(len(r) for r in all_returns)
    padded = np.full((len(seeds), max_len), np.nan)
    for i, r in enumerate(all_returns):
        padded[i, :len(r)] = r

    return padded
