#!/usr/bin/env python3
"""CartPole experiments — Env 1 (common to all groups).

Generates all required analyses:
  1. Q-Learning vs SARSA learning curves
  2. Sensitivity to ε, α, γ (both algorithms)
  3. MC prediction vs TD(0) prediction
  4. Failure case (instability / non-convergence)

Every result: mean over 3 seeds + std bands.
Figures saved to figures/cartpole/.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np

from rl_lab.configs import CARTPOLE, SEEDS
from rl_lab.envs.discretizer import DiscretizedEnv
from rl_lab.algorithms.tabular.q_learning import q_learning
from rl_lab.algorithms.tabular.sarsa import sarsa
from rl_lab.algorithms.tabular.mc_prediction import mc_prediction
from rl_lab.algorithms.tabular.td0 import td0_prediction
from rl_lab.policies import greedy_action
from rl_lab.experiments.runner import run_multi_seed
from rl_lab.utils.plotting import plot_learning_curves, plot_sensitivity
from rl_lab.utils.seeding import set_seed

FIG_DIR = "figures/cartpole"


def make_cartpole():
    env = gym.make(CARTPOLE["env_name"])
    return DiscretizedEnv(env, CARTPOLE["n_bins"],
                          CARTPOLE["clip_low"], CARTPOLE["clip_high"])


# -----------------------------------------------------------------------
# 1. Q-Learning vs SARSA
# -----------------------------------------------------------------------
def exp_control_comparison():
    print("\n=== 1. Q-Learning vs SARSA ===")
    ql_ret, _ = run_multi_seed(make_cartpole, q_learning, SEEDS,
                               episodes=5000, alpha=0.1, gamma=0.99,
                               epsilon=0.1)
    sa_ret, _ = run_multi_seed(make_cartpole, sarsa, SEEDS,
                               episodes=5000, alpha=0.1, gamma=0.99,
                               epsilon=0.1)
    plot_learning_curves(
        {"Q-Learning": ql_ret, "SARSA": sa_ret},
        title="CartPole — Q-Learning vs SARSA",
        save_path=f"{FIG_DIR}/control_comparison.png",
    )
    np.savez(f"results/cartpole/control_comparison.npz",
             ql=ql_ret, sarsa=sa_ret)


# -----------------------------------------------------------------------
# 2. Sensitivity analyses
# -----------------------------------------------------------------------
def _sweep(algo_fn, algo_name, param_name, param_values, defaults):
    print(f"\n=== 2. {algo_name} sensitivity to {param_name} ===")
    results = {}
    for val in param_values:
        kwargs = dict(defaults)
        kwargs[param_name] = val
        ret, _ = run_multi_seed(make_cartpole, algo_fn, SEEDS, **kwargs)
        results[val] = ret
    plot_sensitivity(
        results, param_name, algo_name,
        save_path=f"{FIG_DIR}/sensitivity_{algo_name}_{param_name}.png",
    )


def exp_sensitivity():
    defaults = dict(episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1)

    for algo_fn, name in [(q_learning, "QLearning"), (sarsa, "SARSA")]:
        _sweep(algo_fn, name, "epsilon", [0.01, 0.05, 0.1, 0.3], defaults)
        _sweep(algo_fn, name, "alpha",   [0.01, 0.05, 0.1, 0.5], defaults)
        _sweep(algo_fn, name, "gamma",   [0.9, 0.95, 0.99, 1.0], defaults)


# -----------------------------------------------------------------------
# 3. MC prediction vs TD(0)
# -----------------------------------------------------------------------
def exp_prediction_comparison():
    print("\n=== 3. MC prediction vs TD(0) ===")

    # Train a Q-Learning agent first → extract a greedy policy to evaluate
    set_seed(42)
    env = make_cartpole()
    Q, _, _ = q_learning(env, episodes=5000, alpha=0.1, gamma=0.99,
                         epsilon=0.1)
    env.close()
    print("  Learned Q-table, extracting greedy policy...")

    def learned_policy(state):
        return greedy_action(Q, state)

    # Also evaluate a random policy for comparison
    n_act = 2
    def random_policy(state):
        return np.random.randint(n_act)

    for policy_fn, label in [(learned_policy, "learned"),
                             (random_policy, "random")]:
        mc_ret, _ = run_multi_seed(make_cartpole, mc_prediction, SEEDS,
                                   policy_fn=policy_fn, episodes=3000,
                                   gamma=0.99)
        td_ret, _ = run_multi_seed(make_cartpole, td0_prediction, SEEDS,
                                   policy_fn=policy_fn, episodes=3000,
                                   alpha=0.1, gamma=0.99)
        plot_learning_curves(
            {"MC Prediction": mc_ret, "TD(0) Prediction": td_ret},
            title=f"CartPole — MC vs TD(0) ({label} policy)",
            save_path=f"{FIG_DIR}/prediction_{label}_policy.png",
            ylabel="Episode return (under evaluated policy)",
        )


# -----------------------------------------------------------------------
# 4. Failure case
# -----------------------------------------------------------------------
def exp_failure_case():
    print("\n=== 4. Failure case ===")

    # High learning rate → instability
    results = {}
    for alpha in [0.1, 0.5, 1.0]:
        ret, _ = run_multi_seed(make_cartpole, q_learning, SEEDS,
                                episodes=3000, alpha=alpha, gamma=0.99,
                                epsilon=0.1)
        results[alpha] = ret
    plot_sensitivity(
        results, "α", "Q-Learning",
        title="CartPole — Failure case: high learning rate → instability",
        save_path=f"{FIG_DIR}/failure_high_alpha.png",
    )

    # No exploration → poor convergence
    results2 = {}
    for eps in [0.0, 0.01, 0.1]:
        ret, _ = run_multi_seed(make_cartpole, q_learning, SEEDS,
                                episodes=3000, alpha=0.1, gamma=0.99,
                                epsilon=eps)
        results2[eps] = ret
    plot_sensitivity(
        results2, "ε", "Q-Learning",
        title="CartPole — Failure case: no exploration (ε=0)",
        save_path=f"{FIG_DIR}/failure_no_exploration.png",
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("results/cartpole", exist_ok=True)

    exp_control_comparison()
    exp_sensitivity()
    exp_prediction_comparison()
    exp_failure_case()

    print(f"\nDone. Figures saved in {FIG_DIR}/")
