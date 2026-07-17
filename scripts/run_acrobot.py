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

# Reward is -1 per step, truncated at 500. Random ~ -500, a good swing-up ~ -100.
# 20k episodes (double CartPole's) because the state space is far larger.
EPISODES = 20000
MAX_STEPS = 500


def make_acrobot():
    env = gym.make(ACROBOT["env_name"])
    return DiscretizedEnv(env, ACROBOT["n_bins"],
                          ACROBOT["clip_low"], ACROBOT["clip_high"])


def exp_control_comparison():
    """Q-Learning vs SARSA on the hard environment."""
    print("\n=== 1. Q-Learning vs SARSA (Acrobot) ===")
    ql_ret, _ = run_multi_seed(make_acrobot, q_learning, SEEDS,
                               episodes=EPISODES, alpha=0.1, gamma=0.99,
                               epsilon=0.1, max_steps=MAX_STEPS)
    sa_ret, _ = run_multi_seed(make_acrobot, sarsa, SEEDS,
                               episodes=EPISODES, alpha=0.1, gamma=0.99,
                               epsilon=0.1, max_steps=MAX_STEPS)
    plot_learning_curves(
        {"Q-Learning": ql_ret, "SARSA": sa_ret},
        title="Acrobot - Q-Learning vs SARSA",
        save_path=f"{FIG_DIR}/control_comparison.png",
    )
    np.savez("results/acrobot/control_comparison.npz", ql=ql_ret, sarsa=sa_ret)


def exp_bin_resolution():
    """How many bins per dimension? 4 / 6 / 8 / 10, i.e. 4k to 1M states.

    My hypothesis was that MORE bins would be WORSE: the state space grows as bins^6,
    so at 10 bins (10^6 states) the table should be hopelessly sparse.

    The data says the opposite, and consistently so over 20k episodes: 10 bins is the
    best configuration and 4 bins the worst, in a clean monotone ordering. Acrobot's
    dynamics need angular precision; a 4-bin grid cannot separate states that require
    different torques, and that aliasing costs more than sparsity saves. The curse of
    dimensionality is real, but at this budget it is not yet the binding constraint.
    I kept the result and analyzed it rather than tuning it away.
    """
    print("\n=== 2. Discretization resolution ===")
    results = {}

    for n_bins in [4, 6, 8, 10]:
        bins = [n_bins] * 6
        total_states = n_bins ** 6
        print(f"  n_bins={n_bins} - {total_states} states")

        def _make(b=bins):  # b=bins binds the current value, not the loop variable
            env = gym.make(ACROBOT["env_name"])
            return DiscretizedEnv(env, b, ACROBOT["clip_low"], ACROBOT["clip_high"])

        ret, _ = run_multi_seed(_make, q_learning, SEEDS,
                                episodes=EPISODES, alpha=0.1, gamma=0.99,
                                epsilon=0.2, max_steps=MAX_STEPS)
        results[f"{n_bins} ({total_states} states)"] = ret

    plot_learning_curves(
        results,
        title="Acrobot - Q-Learning: effect of bin resolution",
        save_path=f"{FIG_DIR}/bin_resolution.png",
    )


def exp_sensitivity_epsilon():
    """Sensitivity to epsilon, i.e. the exploration difficulty of a sparse reward.

    Hypothesis: a sparse reward should demand MORE exploration.

    Again the data disagrees: low epsilon (0.05-0.1) wins and epsilon=0.4 is clearly
    the worst. The catch is that epsilon here is CONSTANT: a high value keeps injecting
    random torques forever, which wrecks the swing-up the agent already found. High
    exploration would only pay off early, which is exactly what the annealing
    experiment below tests.
    """
    print("\n=== 3. Sensitivity to epsilon (Acrobot) ===")
    results = {}
    for eps in [0.05, 0.1, 0.2, 0.4]:
        ret, _ = run_multi_seed(make_acrobot, q_learning, SEEDS,
                                episodes=EPISODES, alpha=0.1, gamma=0.99,
                                epsilon=eps, max_steps=MAX_STEPS)
        results[eps] = ret
    plot_sensitivity(
        results, "epsilon", "Q-Learning",
        title="Acrobot - exploration difficulty (epsilon sensitivity)",
        save_path=f"{FIG_DIR}/sensitivity_epsilon.png",
    )


def exp_prediction():
    """MC vs TD(0) on the learned policy. Same design as CartPole (see run_cartpole)."""
    print("\n=== 4. MC vs TD(0) prediction (Acrobot) ===")

    set_seed(42)
    env = make_acrobot()
    Q, _, _ = q_learning(env, episodes=EPISODES, alpha=0.1, gamma=0.99,
                         epsilon=0.1, max_steps=MAX_STEPS)
    env.close()

    def learned_policy(state):
        return greedy_action(Q, state)

    # Reference state = discretized reset state. The "centre bin" anchor I first tried
    # is physically impossible here: the observation encodes angles as [cos, sin], and
    # cos and sin cannot both be ~0, so that state is never visited and V stays flat
    # at 0 forever. This bit me before I looked at the plot.
    _e = make_acrobot(); anchor, _ = _e.reset(seed=0); _e.close()

    mc_ret, mc_res = run_multi_seed(make_acrobot, mc_prediction, SEEDS,
                                    policy_fn=learned_policy, episodes=6000,
                                    alpha=0.1, gamma=0.99, max_steps=MAX_STEPS,
                                    track_state=anchor)
    td_ret, td_res = run_multi_seed(make_acrobot, td0_prediction, SEEDS,
                                    policy_fn=learned_policy, episodes=6000,
                                    alpha=0.1, gamma=0.99, max_steps=MAX_STEPS,
                                    track_state=anchor)

    plot_learning_curves(
        {"MC Prediction": mc_ret, "TD(0) Prediction": td_ret},
        title="Acrobot - MC vs TD(0): episode return (learned policy)",
        save_path=f"{FIG_DIR}/prediction_return.png",
        ylabel="Episode return (under evaluated policy)",
    )

    mc_vhist = np.array([r[2] for r in mc_res])
    td_vhist = np.array([r[2] for r in td_res])
    plot_learning_curves(
        {"MC estimate": mc_vhist, "TD(0) estimate": td_vhist},
        title="Acrobot - MC vs TD(0): value estimate V(s_ref) (learned policy)",
        save_path=f"{FIG_DIR}/prediction_value.png",
        ylabel="V(reference state) estimate",
        window=1,
    )
    np.savez("results/acrobot/prediction.npz",
             mc_value=mc_vhist, td_value=td_vhist,
             mc_return=mc_ret, td_return=td_ret)


def exp_epsilon_annealing():
    """Does an annealed epsilon beat the best constant one (0.1) on Acrobot?

    This is where annealing should pay off, in theory: explore hard early to stumble
    onto the goal at all, then exploit the swing-up. It is the direct follow-up to the
    epsilon-sensitivity result above.
    """
    print("\n=== 5. Epsilon annealing vs constant (Acrobot) ===")
    const, _ = run_multi_seed(make_acrobot, q_learning, SEEDS,
                              episodes=EPISODES, alpha=0.1, gamma=0.99,
                              epsilon=0.1, max_steps=MAX_STEPS)
    anneal, _ = run_multi_seed(make_acrobot, q_learning, SEEDS,
                               episodes=EPISODES, alpha=0.1, gamma=0.99,
                               epsilon=1.0, epsilon_end=0.01, max_steps=MAX_STEPS)
    plot_learning_curves(
        {"epsilon costante = 0.1": const,
         "epsilon annealing 1.0 to 0.01": anneal},
        title="Acrobot - epsilon costante vs annealing",
        save_path=f"{FIG_DIR}/epsilon_annealing.png",
    )
    np.savez("results/acrobot/epsilon_annealing.npz", const=const, anneal=anneal)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("results/acrobot", exist_ok=True)

    exp_control_comparison()
    exp_bin_resolution()
    exp_sensitivity_epsilon()
    exp_prediction()
    exp_epsilon_annealing()

    print(f"\nDone. Figures saved in {FIG_DIR}/")
