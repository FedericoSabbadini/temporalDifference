#!/usr/bin/env python3
"""CartPole experiments - Env 1 (common to all groups).

Origin: NEW. Orchestrator that runs all required CartPole analyses.
The experiment design follows the assignment requirements point by point.

Generates all required analyses:
  1. Q-Learning vs SARSA learning curves (control comparison)
  2. Sensitivity to ε, α, γ (both algorithms)
  3. MC prediction vs TD(0) prediction (prediction comparison)
  4. Failure case (instability / non-convergence)

Every result: mean over 3 seeds + std bands.
Figures saved to figures/cartpole/.
"""

import sys, os
# Add project root to Python path so `from rl_lab import ...` works
# when running this script directly (python scripts/run_cartpole.py)
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
    """Create a discretized CartPole environment."""
    env = gym.make(CARTPOLE["env_name"])  # continuous 4-D observation, 2 actions
    # Wrap with discretizer: continuous obs - tuple of bin indices (hashable for Q-table)
    return DiscretizedEnv(env, CARTPOLE["n_bins"],
                          CARTPOLE["clip_low"], CARTPOLE["clip_high"])


# -----------------------------------------------------------------------
# 1. Q-Learning vs SARSA (control comparison)
# Assignment: "SARSA vs Q-learning: differenze empiriche e teoriche"
# -----------------------------------------------------------------------
def exp_control_comparison():
    """Compare Q-Learning (off-policy) vs SARSA (on-policy) on CartPole.

    Both are TD control algorithms that learn a policy from interaction.
    Q-Learning uses max (greedy target), SARSA uses the actual next action.
    On CartPole the difference is subtle - no "cliff" to make it dramatic.
    """
    print("\n=== 1. Q-Learning vs SARSA ===")
    # Run each algorithm with the SAME hyperparameters for a fair comparison
    ql_ret, _ = run_multi_seed(make_cartpole, q_learning, SEEDS,
                               episodes=5000, alpha=0.1, gamma=0.99,
                               epsilon=0.1)
    sa_ret, _ = run_multi_seed(make_cartpole, sarsa, SEEDS,
                               episodes=5000, alpha=0.1, gamma=0.99,
                               epsilon=0.1)
    # Plot both on the same axes for direct visual comparison
    plot_learning_curves(
        {"Q-Learning": ql_ret, "SARSA": sa_ret},
        title="CartPole - Q-Learning vs SARSA",
        save_path=f"{FIG_DIR}/control_comparison.png",
    )
    # Save raw data so plots can be regenerated without re-training
    np.savez("results/cartpole/control_comparison.npz",
             ql=ql_ret, sarsa=sa_ret)


# -----------------------------------------------------------------------
# 2. Sensitivity analyses
# Assignment: "analisi sensibilità rispetto a ε, α, γ"
# -----------------------------------------------------------------------
def _sweep(algo_fn, algo_name, param_name, param_values, defaults):
    """Run an algorithm with different values of one hyperparameter.

    Keeps all other parameters at their default values.
    Produces a plot showing how the learning curve changes.
    """
    print(f"\n=== 2. {algo_name} sensitivity to {param_name} ===")
    results = {}  # {param_value: (n_seeds, n_episodes) array}
    for val in param_values:
        kwargs = dict(defaults)  # copy defaults so we don't mutate the original
        kwargs[param_name] = val  # override only the parameter being swept
        ret, _ = run_multi_seed(make_cartpole, algo_fn, SEEDS, **kwargs)
        results[val] = ret
    plot_sensitivity(
        results, param_name, algo_name,
        save_path=f"{FIG_DIR}/sensitivity_{algo_name}_{param_name}.png",
    )


def exp_sensitivity():
    """Sweep ε, α, γ for both Q-Learning and SARSA.

    This produces 6 plots (3 params × 2 algorithms).
    The default values (α=0.1, γ=0.99, ε=0.1) are standard from S&B.
    """
    defaults = dict(episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1)

    for algo_fn, name in [(q_learning, "QLearning"), (sarsa, "SARSA")]:
        _sweep(algo_fn, name, "epsilon", [0.01, 0.05, 0.1, 0.3], defaults)
        _sweep(algo_fn, name, "alpha",   [0.01, 0.05, 0.1, 0.5], defaults)
        _sweep(algo_fn, name, "gamma",   [0.9, 0.95, 0.99, 1.0], defaults)


# -----------------------------------------------------------------------
# 3. MC prediction vs TD(0)
# Assignment: "Monte Carlo vs TD(0): bias, varianza, stabilità"
# -----------------------------------------------------------------------
def exp_prediction_comparison():
    """Compare MC and TD(0) as prediction algorithms.

    MC and TD(0) are PREDICTION algorithms: they estimate V(s) for a
    FIXED policy (they don't learn or improve the policy).

    Approach:
    1. Train Q-Learning to learn a good Q-table
    2. Extract the greedy policy: π(s) = argmax_a Q(s,a)
    3. Evaluate this fixed policy with both MC and TD(0)
    4. Also evaluate a random policy for comparison

    TWO views are produced:
    - Episode RETURN: shows how the evaluated policy performs. Note that MC and
      TD(0) follow the SAME policy, so this curve is identical by construction - 
      it does NOT reveal the bias/variance difference between the estimators.
    - Value ESTIMATE V(reference state) over episodes: THIS is where MC and TD(0)
      differ. MC's estimate is noisy (unbiased, high variance); TD(0)'s is smooth
      (biased early, low variance). This is the meaningful comparison and matches
      the classic random-walk experiment from Sutton & Barto (§6.2).
    """
    print("\n=== 3. MC prediction vs TD(0) ===")

    # Step 1: Train Q-Learning to get a good Q-table (control - learn a policy)
    set_seed(42)  # fixed seed for reproducibility of the learned policy
    env = make_cartpole()
    Q, _, _ = q_learning(env, episodes=5000, alpha=0.1, gamma=0.99,
                         epsilon=0.1)
    env.close()
    print("  Learned Q-table, extracting greedy policy...")

    # Extract the greedy policy from the learned Q-table: π(s) = argmax_a Q(s,a)
    # This is now a FIXED policy - MC and TD(0) will evaluate it without changing it
    def learned_policy(state):
        return greedy_action(Q, state)

    # Random policy for comparison - helps show that the learned policy is actually good
    n_act = 2  # CartPole has 2 actions (left, right)
    def random_policy(state):
        return np.random.randint(n_act)  # uniform random: each action with prob 0.5

    # Reference state whose value we track over episodes: the DISCRETIZED RESET
    # state. This is guaranteed reachable and is visited at the start of every
    # episode, so both estimators update it often. (A geometric "centre bin"
    # would be unreachable for envs with trig-encoded angles like Acrobot, where
    # cos θ and sin θ cannot both be ~0.)
    _e = make_cartpole(); anchor, _ = _e.reset(seed=0); _e.close()

    # Step 2: Evaluate each policy with both MC and TD(0)
    for policy_fn, label in [(learned_policy, "learned"),
                             (random_policy, "random")]:
        # MC prediction: waits until episode end, then uses the full return G
        mc_ret, mc_res = run_multi_seed(make_cartpole, mc_prediction, SEEDS,
                                        policy_fn=policy_fn, episodes=3000,
                                        gamma=0.99, track_state=anchor)
        # TD(0) prediction: updates at every step using bootstrap estimate V(S')
        td_ret, td_res = run_multi_seed(make_cartpole, td0_prediction, SEEDS,
                                        policy_fn=policy_fn, episodes=3000,
                                        alpha=0.1, gamma=0.99, track_state=anchor)

        # View 1: episode return (identical by construction - kept for honesty)
        plot_learning_curves(
            {"MC Prediction": mc_ret, "TD(0) Prediction": td_ret},
            title=f"CartPole - MC vs TD(0): episode return ({label} policy)",
            save_path=f"{FIG_DIR}/prediction_{label}_return.png",
            ylabel="Episode return (under evaluated policy)",
        )

        # View 2: value estimate of the reference state - the meaningful one.
        # value_history is result index [2]; stack across seeds - (n_seeds, episodes)
        mc_vhist = np.array([r[2] for r in mc_res])
        td_vhist = np.array([r[2] for r in td_res])
        plot_learning_curves(
            {"MC estimate": mc_vhist, "TD(0) estimate": td_vhist},
            title=f"CartPole - MC vs TD(0): value estimate V(s_ref) ({label} policy)",
            save_path=f"{FIG_DIR}/prediction_{label}_value.png",
            ylabel="V(reference state) estimate",
            window=1,  # no smoothing: we WANT to see MC's raw variance vs TD's
        )
        np.savez(f"results/cartpole/prediction_{label}.npz",
                 mc_value=mc_vhist, td_value=td_vhist,
                 mc_return=mc_ret, td_return=td_ret)


# -----------------------------------------------------------------------
# 4. Failure case
# Assignment: "Identificare almeno un failure case: instabilità,
#              mancata convergenza, comportamento subottimale"
# -----------------------------------------------------------------------
def exp_failure_case():
    """Demonstrate two common failure modes of tabular RL.

    1. HIGH α (learning rate): update too aggressive - Q-values oscillate
       without converging. This is the instability failure case.
       (S&B §6.7 discusses sensitivity to α)

    2. NO EXPLORATION (ε=0): the agent follows the greedy policy from the
       start, never discovers better actions - convergence to suboptimal.
       This is the exploration/exploitation dilemma (S&B §2.1).
    """
    print("\n=== 4. Failure case ===")

    # Failure 1: high learning rate - instability
    # α=1.0 means Q(s,a) is fully replaced by the new target every step
    # - Q-values oscillate wildly and never converge
    results = {}
    for alpha in [0.1, 0.5, 1.0]:
        ret, _ = run_multi_seed(make_cartpole, q_learning, SEEDS,
                                episodes=3000, alpha=alpha, gamma=0.99,
                                epsilon=0.1)
        results[alpha] = ret
    plot_sensitivity(
        results, "α", "Q-Learning",
        title="CartPole - Failure case: high learning rate - instability",
        save_path=f"{FIG_DIR}/failure_high_alpha.png",
    )

    # Failure 2: no exploration - poor convergence
    # ε=0.0 means the agent NEVER explores - it gets stuck with whatever
    # actions it tried first, even if better actions exist (S&B §2.1)
    results2 = {}
    for eps in [0.0, 0.01, 0.1]:
        ret, _ = run_multi_seed(make_cartpole, q_learning, SEEDS,
                                episodes=3000, alpha=0.1, gamma=0.99,
                                epsilon=eps)
        results2[eps] = ret
    plot_sensitivity(
        results2, "ε", "Q-Learning",
        title="CartPole - Failure case: no exploration (ε=0)",
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
