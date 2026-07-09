#!/usr/bin/env python3
"""Acrobot experiments — Env 2 (group-specific).

Applies the same tabular algorithms used on CartPole to Acrobot-v1.
Key discussion points for the report:
  - 6-D continuous state → discretization curse (6^6 = 46656 states)
  - Sparse reward (-1 per step) → hard exploration
  - Coarse bins lose critical angular information → suboptimal behaviour
  - This motivates function approximation (Env 3 / DQN)

Figures saved to figures/acrobot/.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np

from rl_lab.configs import ACROBOT, SEEDS
from rl_lab.envs.discretizer import DiscretizedEnv
from rl_lab.algorithms.tabular.q_learning import q_learning
from rl_lab.algorithms.tabular.sarsa import sarsa
from rl_lab.algorithms.tabular.mc_prediction import mc_prediction
from rl_lab.algorithms.tabular.td0 import td0_prediction
from rl_lab.policies import greedy_action
from rl_lab.experiments.runner import run_multi_seed
from rl_lab.utils.plotting import plot_learning_curves, plot_sensitivity
from rl_lab.utils.seeding import set_seed

FIG_DIR = "figures/acrobot"

# Acrobot reward: -1 per step, episode truncated at 500 steps.
# Best possible return ≈ -80 (swing up quickly). Random ≈ -500.
EPISODES = 10000  # more episodes needed due to large state space
MAX_STEPS = 500


def make_acrobot():
    env = gym.make(ACROBOT["env_name"])
    return DiscretizedEnv(env, ACROBOT["n_bins"],
                          ACROBOT["clip_low"], ACROBOT["clip_high"])


# -----------------------------------------------------------------------
# 1. Q-Learning vs SARSA
# -----------------------------------------------------------------------
def exp_control_comparison():
    print("\n=== 1. Q-Learning vs SARSA (Acrobot) ===")
    ql_ret, _ = run_multi_seed(make_acrobot, q_learning, SEEDS,
                               episodes=EPISODES, alpha=0.1, gamma=0.99,
                               epsilon=0.1, max_steps=MAX_STEPS)
    sa_ret, _ = run_multi_seed(make_acrobot, sarsa, SEEDS,
                               episodes=EPISODES, alpha=0.1, gamma=0.99,
                               epsilon=0.1, max_steps=MAX_STEPS)
    plot_learning_curves(
        {"Q-Learning": ql_ret, "SARSA": sa_ret},
        title="Acrobot — Q-Learning vs SARSA",
        save_path=f"{FIG_DIR}/control_comparison.png",
    )
    np.savez("results/acrobot/control_comparison.npz",
             ql=ql_ret, sarsa=sa_ret)


# -----------------------------------------------------------------------
# 2. Impact of bin resolution (discretization analysis)
# -----------------------------------------------------------------------
def exp_bin_resolution():
    """Show how bin count affects learning — curse of dimensionality."""
    print("\n=== 2. Discretization resolution analysis ===")
    results = {}

    for n_bins in [4, 6, 8, 10]:
        bins = [n_bins] * 6
        total_states = n_bins ** 6
        print(f"  n_bins={n_bins} → {total_states} states")

        def _make(b=bins):
            env = gym.make(ACROBOT["env_name"])
            return DiscretizedEnv(env, b,
                                  ACROBOT["clip_low"], ACROBOT["clip_high"])

        ret, _ = run_multi_seed(_make, q_learning, SEEDS,
                                episodes=EPISODES, alpha=0.1, gamma=0.99,
                                epsilon=0.2, max_steps=MAX_STEPS)
        results[f"{n_bins} ({total_states} states)"] = ret

    plot_learning_curves(
        results,
        title="Acrobot — Q-Learning: effect of bin resolution",
        save_path=f"{FIG_DIR}/bin_resolution.png",
    )


# -----------------------------------------------------------------------
# 3. Sensitivity to epsilon (exploration difficulty)
# -----------------------------------------------------------------------
def exp_sensitivity_epsilon():
    print("\n=== 3. Sensitivity to ε (Acrobot) ===")
    results = {}
    for eps in [0.05, 0.1, 0.2, 0.4]:
        ret, _ = run_multi_seed(make_acrobot, q_learning, SEEDS,
                                episodes=EPISODES, alpha=0.1, gamma=0.99,
                                epsilon=eps, max_steps=MAX_STEPS)
        results[eps] = ret
    plot_sensitivity(
        results, "ε", "Q-Learning",
        title="Acrobot — exploration difficulty (ε sensitivity)",
        save_path=f"{FIG_DIR}/sensitivity_epsilon.png",
    )


# -----------------------------------------------------------------------
# 4. MC vs TD(0) prediction
# -----------------------------------------------------------------------
def exp_prediction():
    print("\n=== 4. MC vs TD(0) prediction (Acrobot) ===")

    set_seed(42)
    env = make_acrobot()
    Q, _, _ = q_learning(env, episodes=EPISODES, alpha=0.1, gamma=0.99,
                         epsilon=0.1, max_steps=MAX_STEPS)
    env.close()

    def learned_policy(state):
        return greedy_action(Q, state)

    mc_ret, _ = run_multi_seed(make_acrobot, mc_prediction, SEEDS,
                               policy_fn=learned_policy, episodes=3000,
                               gamma=0.99, max_steps=MAX_STEPS)
    td_ret, _ = run_multi_seed(make_acrobot, td0_prediction, SEEDS,
                               policy_fn=learned_policy, episodes=3000,
                               alpha=0.1, gamma=0.99, max_steps=MAX_STEPS)
    plot_learning_curves(
        {"MC Prediction": mc_ret, "TD(0) Prediction": td_ret},
        title="Acrobot — MC vs TD(0) (learned policy)",
        save_path=f"{FIG_DIR}/prediction_comparison.png",
        ylabel="Episode return (under evaluated policy)",
    )


# -----------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("results/acrobot", exist_ok=True)

    exp_control_comparison()
    exp_bin_resolution()
    exp_sensitivity_epsilon()
    exp_prediction()

    print(f"\nDone. Figures saved in {FIG_DIR}/")
