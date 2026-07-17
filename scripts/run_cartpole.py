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
EPISODES = 10000


def make_cartpole():
    env = gym.make(CARTPOLE["env_name"])
    return DiscretizedEnv(env, CARTPOLE["n_bins"],
                          CARTPOLE["clip_low"], CARTPOLE["clip_high"])


def exp_control_comparison():
    """Q-Learning (off-policy) vs SARSA (on-policy), identical hyperparameters."""
    print("\n=== 1. Q-Learning vs SARSA (CartPole) ===")
    ql_ret, _ = run_multi_seed(make_cartpole, q_learning, SEEDS,
                               episodes=EPISODES, alpha=0.1, gamma=0.99,
                               epsilon=0.1)
    sa_ret, _ = run_multi_seed(make_cartpole, sarsa, SEEDS,
                               episodes=EPISODES, alpha=0.1, gamma=0.99,
                               epsilon=0.1)
    plot_learning_curves(
        {"Q-Learning": ql_ret, "SARSA": sa_ret},
        title="CartPole - Q-Learning vs SARSA",
        save_path=f"{FIG_DIR}/control_comparison.png",
    )
    np.savez("results/cartpole/control_comparison.npz", ql=ql_ret, sarsa=sa_ret)


def _sweep(algo_fn, algo_name, param_name, param_values, defaults):
    """Sweep one hyperparameter, holding the others at their defaults."""
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
    """3 hyperparameters x 2 algorithms = 6 figures. Defaults are the S&B values."""
    defaults = dict(episodes=EPISODES, alpha=0.1, gamma=0.99, epsilon=0.1)

    for algo_fn, name in [(q_learning, "QLearning"), (sarsa, "SARSA")]:
        _sweep(algo_fn, name, "epsilon", [0.01, 0.05, 0.1, 0.3], defaults)
        _sweep(algo_fn, name, "alpha",   [0.01, 0.05, 0.1, 0.5], defaults)
        _sweep(algo_fn, name, "gamma",   [0.9, 0.95, 0.99, 1.0], defaults)


def exp_prediction_comparison():
    """MC vs TD(0): estimate V(s) of a FIXED policy (they never change it).

    The policy to evaluate has to come from somewhere, so I train Q-Learning first and
    evaluate its greedy policy; a random policy is evaluated too, as a reference.

    Two views, because only the second one is informative:
      - episode RETURN: identical by construction between MC and TD(0), since both run
        the same fixed policy. It says nothing about the estimators.
      - value ESTIMATE V(s_ref) per episode: this is where they differ, and it
        reproduces the classic bias/variance picture of S&B §6.2.
    """
    print("\n=== 3. MC prediction vs TD(0) ===")

    set_seed(42)
    env = make_cartpole()
    Q, _, _ = q_learning(env, episodes=EPISODES, alpha=0.1, gamma=0.99, epsilon=0.1)
    env.close()

    def learned_policy(state):
        return greedy_action(Q, state)

    def random_policy(state):
        return np.random.randint(2)

    # Reference state = the DISCRETIZED RESET state: guaranteed reachable and visited
    # at the start of every episode, so both estimators update it constantly. A
    # geometric "centre bin" would be unreachable on Acrobot (cos and sin cannot both
    # be ~0), and I want the same anchor rule in both environments.
    _e = make_cartpole(); anchor, _ = _e.reset(seed=0); _e.close()

    for policy_fn, label in [(learned_policy, "learned"), (random_policy, "random")]:
        mc_ret, mc_res = run_multi_seed(make_cartpole, mc_prediction, SEEDS,
                                        policy_fn=policy_fn, episodes=6000,
                                        alpha=0.1, gamma=0.99, track_state=anchor)
        td_ret, td_res = run_multi_seed(make_cartpole, td0_prediction, SEEDS,
                                        policy_fn=policy_fn, episodes=6000,
                                        alpha=0.1, gamma=0.99, track_state=anchor)

        plot_learning_curves(
            {"MC Prediction": mc_ret, "TD(0) Prediction": td_ret},
            title=f"CartPole - MC vs TD(0): episode return ({label} policy)",
            save_path=f"{FIG_DIR}/prediction_{label}_return.png",
            ylabel="Episode return (under evaluated policy)",
        )

        # value_history is index [2] of the result tuple
        mc_vhist = np.array([r[2] for r in mc_res])
        td_vhist = np.array([r[2] for r in td_res])
        plot_learning_curves(
            {"MC estimate": mc_vhist, "TD(0) estimate": td_vhist},
            title=f"CartPole - MC vs TD(0): value estimate V(s_ref) ({label} policy)",
            save_path=f"{FIG_DIR}/prediction_{label}_value.png",
            ylabel="V(reference state) estimate",
            window=1,  # no smoothing: MC's raw variance is the whole point
        )
        np.savez(f"results/cartpole/prediction_{label}.npz",
                 mc_value=mc_vhist, td_value=td_vhist,
                 mc_return=mc_ret, td_return=td_ret)


def exp_failure_case():
    """Three failure modes, each isolating one hyperparameter taken to its extreme.

    1. alpha=1.0: the update fully replaces the estimate every step, so Q never
       averages over the stochastic transitions and the agent stalls.
    2. epsilon=0: pure exploitation from an all-zero Q-table; the agent locks onto
       whatever it tried first (S&B §2.1).
    3. gamma=0: the target is just the immediate reward, which on CartPole is +1 for
       every non-terminal step whatever the action. All actions therefore look equally
       good, the greedy policy degenerates to random, and no credit is ever assigned.
    """
    print("\n=== 4. Failure cases ===")

    for values, param, fname, title in [
        ([0.1, 0.5, 1.0], "alpha", "failure_high_alpha",
         "CartPole - Failure case: high learning rate - instability"),
        ([0.0, 0.01, 0.1], "epsilon", "failure_no_exploration",
         "CartPole - Failure case: no exploration (eps=0)"),
        ([0.0, 0.9, 0.99], "gamma", "failure_no_discount",
         "CartPole - Failure case: no discounting (gamma=0 - myopia)"),
    ]:
        results = {}
        for val in values:
            kwargs = dict(episodes=6000, alpha=0.1, gamma=0.99, epsilon=0.1)
            kwargs[param] = val
            ret, _ = run_multi_seed(make_cartpole, q_learning, SEEDS, **kwargs)
            results[val] = ret
        plot_sensitivity(results, param, "Q-Learning", title=title,
                         save_path=f"{FIG_DIR}/{fname}.png")


def exp_epsilon_annealing():
    """Constant vs annealed epsilon.

    I run the same comparison on both environments on purpose. CartPole has a DENSE
    reward and is not exploration-limited, Acrobot is the opposite, so the contrast
    tells me whether annealing helps universally or only where exploration is the
    bottleneck.
    """
    print("\n=== 5. Epsilon annealing vs constant ===")
    const, _ = run_multi_seed(make_cartpole, q_learning, SEEDS,
                              episodes=EPISODES, alpha=0.1, gamma=0.99,
                              epsilon=0.1)
    anneal, _ = run_multi_seed(make_cartpole, q_learning, SEEDS,
                               episodes=EPISODES, alpha=0.1, gamma=0.99,
                               epsilon=1.0, epsilon_end=0.01)
    plot_learning_curves(
        {"epsilon costante = 0.1": const,
         "epsilon annealing 1.0 to 0.01": anneal},
        title="CartPole - epsilon costante vs annealing (control)",
        save_path=f"{FIG_DIR}/epsilon_annealing.png",
    )
    np.savez("results/cartpole/epsilon_annealing.npz", const=const, anneal=anneal)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("results/cartpole", exist_ok=True)

    exp_control_comparison()
    exp_sensitivity()
    exp_prediction_comparison()
    exp_failure_case()
    exp_epsilon_annealing()

    print(f"\nDone. Figures saved in {FIG_DIR}/")
