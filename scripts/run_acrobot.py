#!/usr/bin/env python3
"""Acrobot experiments — Env 2 (group-specific, assigned by professor).

Origin: NEW. Experiment design is original, addressing the Env 2 requirements:
"Applicare gli algoritmi implementati, discutere le difficoltà specifiche."

Key analysis points for the report:
  - 6-D continuous state → discretization curse (6^6 = 46656 states)
  - Sparse reward (-1 per step) → hard exploration
  - Coarse bins lose critical angular information → suboptimal behaviour
  - This motivates function approximation (Env 3 / DQN)

The bin_resolution experiment is the most original analysis in the project:
it concretely demonstrates the curse of dimensionality.

Figures saved to figures/acrobot/.
"""

import sys, os
# Add project root to Python path so `from rl_lab import ...` works
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
# 10,000 episodes needed (vs 5,000 on CartPole) due to 6D state space.
EPISODES = 10000
MAX_STEPS = 500


def make_acrobot():
    """Create a discretized Acrobot environment (6 bins per dimension)."""
    env = gym.make(ACROBOT["env_name"])  # continuous 6-D observation, 3 actions
    # 6 bins/dim → 6^6 = 46,656 states (much larger than CartPole's 10^4)
    return DiscretizedEnv(env, ACROBOT["n_bins"],
                          ACROBOT["clip_low"], ACROBOT["clip_high"])


# -----------------------------------------------------------------------
# 1. Q-Learning vs SARSA
# -----------------------------------------------------------------------
def exp_control_comparison():
    """Same comparison as CartPole, but on a harder environment.

    Expected: both algorithms struggle more due to sparse reward and
    large state space. Convergence is slower and final return is lower
    (relative to the optimum) compared to CartPole.
    """
    print("\n=== 1. Q-Learning vs SARSA (Acrobot) ===")
    # max_steps=500 matches Acrobot's truncation limit (episode ends after 500 steps)
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
# Assignment: "eventuale necessità di discretizzazione o approximation"
# -----------------------------------------------------------------------
def exp_bin_resolution():
    """Show how bin count affects learning — discretization resolution study.

    A-priori hypothesis: more bins → worse, because the state space grows
    exponentially (10 bins → 10^6 states, "hopelessly sparse" at 10k episodes).

    ACTUAL RESULT (see report Sec. Results / figures/acrobot/bin_resolution.png):
    the OPPOSITE. Within 10k episodes, 10 bins is the BEST (~-300) and 4 bins
    the WORST (~-420). Acrobot's dynamics need angular precision, so a too-coarse
    grid (4 bins) cannot distinguish states, and that effect dominates over
    sparsity at this episode budget. The curse of dimensionality is real in
    principle but is not yet the bottleneck here.

    This discrepancy is kept and analyzed honestly rather than "corrected":
    hypotheses must be verified against data.
    """
    print("\n=== 2. Discretization resolution analysis ===")
    results = {}

    for n_bins in [4, 6, 8, 10]:
        bins = [n_bins] * 6  # same resolution on all 6 dimensions
        total_states = n_bins ** 6  # exponential growth: this IS the curse
        print(f"  n_bins={n_bins} → {total_states} states")

        # b=bins captures the current `bins` value (closure default arg trick)
        # Without it, all closures would use the last loop's value of `bins`
        def _make(b=bins):
            env = gym.make(ACROBOT["env_name"])
            return DiscretizedEnv(env, b,
                                  ACROBOT["clip_low"], ACROBOT["clip_high"])

        # Higher ε (0.2 vs 0.1) because Acrobot's sparse reward needs more exploration
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
# Assignment: "difficoltà di esplorazione"
# -----------------------------------------------------------------------
def exp_sensitivity_epsilon():
    """Show how ε affects learning on Acrobot (sparse reward).

    A-priori hypothesis: sparse reward → needs MORE exploration.

    ACTUAL RESULT (see figures/acrobot/sensitivity_epsilon.png): low ε
    (0.05-0.1) performs BEST (~-320), high ε=0.4 performs WORST (~-440).
    With CONSTANT ε (no annealing), a high value keeps disrupting the learned
    greedy policy and prevents exploiting the swing-up. High exploration would
    help only early on (annealed schedule), not held constant. Another case
    where the data corrects the initial intuition.
    """
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
    """MC vs TD(0) prediction on the learned Acrobot policy.

    Same approach as CartPole (see run_cartpole.py for the full rationale):
    train Q-Learning, extract the greedy policy, evaluate it with MC and TD(0).
    Two views are produced — the episode return (identical by construction,
    same policy) and the value estimate V(reference state) over episodes, which
    is where MC (noisy) and TD(0) (smooth) actually differ.
    """
    print("\n=== 4. MC vs TD(0) prediction (Acrobot) ===")

    # First learn a policy with Q-Learning (control), then evaluate it with MC/TD(0) (prediction)
    set_seed(42)
    env = make_acrobot()
    Q, _, _ = q_learning(env, episodes=EPISODES, alpha=0.1, gamma=0.99,
                         epsilon=0.1, max_steps=MAX_STEPS)
    env.close()

    # Extract the greedy policy: π(s) = argmax_a Q(s,a) — this is now FIXED
    def learned_policy(state):
        return greedy_action(Q, state)

    # Reference state = the DISCRETIZED RESET state (always reachable & visited
    # every episode). A geometric "centre bin" is unreachable on Acrobot, whose
    # observation encodes angles as [cos θ, sin θ, ...]: cos θ and sin θ cannot
    # both be ~0, so the all-centre bin is never visited and V stays 0.
    _e = make_acrobot(); anchor, _ = _e.reset(seed=0); _e.close()
    # achor is a tuple of 6 integers, one per dimension, e.g. (3, 3, 3, 3, 3, 3), retrieved from the discretized reset state.
    # Instead of using a geometric centre bin (which is unreachable), we use the discretized reset state as the reference state for value estimation.

    # MC prediction: higher variance on Acrobot because episodes are up to 500 steps long
    # → the return G = R1 + γR2 + ... + γ^499 R500 depends on 500 random transitions
    mc_ret, mc_res = run_multi_seed(make_acrobot, mc_prediction, SEEDS,
                                    policy_fn=learned_policy, episodes=3000,
                                    gamma=0.99, max_steps=MAX_STEPS,
                                    track_state=anchor)
    # TD(0) prediction: lower variance because it only uses ONE step of randomness
    td_ret, td_res = run_multi_seed(make_acrobot, td0_prediction, SEEDS,
                                    policy_fn=learned_policy, episodes=3000,
                                    alpha=0.1, gamma=0.99, max_steps=MAX_STEPS,
                                    track_state=anchor)

    # View 1: episode return (identical by construction — kept for honesty)
    plot_learning_curves(
        {"MC Prediction": mc_ret, "TD(0) Prediction": td_ret},
        title="Acrobot — MC vs TD(0): episode return (learned policy)",
        save_path=f"{FIG_DIR}/prediction_return.png",
        ylabel="Episode return (under evaluated policy)",
    )

    # View 2: value estimate of the reference state — the meaningful comparison
    mc_vhist = np.array([r[2] for r in mc_res])
    td_vhist = np.array([r[2] for r in td_res])
    plot_learning_curves(
        {"MC estimate": mc_vhist, "TD(0) estimate": td_vhist},
        title="Acrobot — MC vs TD(0): value estimate V(s_ref) (learned policy)",
        save_path=f"{FIG_DIR}/prediction_value.png",
        ylabel="V(reference state) estimate",
        window=1,  # no smoothing: show MC's raw variance vs TD's smoothness
    )
    np.savez("results/acrobot/prediction.npz",
             mc_value=mc_vhist, td_value=td_vhist,
             mc_return=mc_ret, td_return=td_ret)


# -----------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("results/acrobot", exist_ok=True)

    exp_control_comparison()
    exp_bin_resolution()
    exp_sensitivity_epsilon()
    exp_prediction()

    print(f"\nDone. Figures saved in {FIG_DIR}/")
