#!/usr/bin/env python3
"""Pong DQN experiments - Env 3 (Atari / Deep RL).

Origin: NEW. The ablation is exactly the one the assignment asks for (DQN with vs
without target network, with vs without experience replay), plus the worst case with
neither and the Double DQN bonus. Five configurations, 3 seeds each.

The bar is deliberately low: "non è richiesto ottenere performance elevate, è
sufficiente mostrare apprendimento rispetto a policy casuale". So I train for 200k
steps (the Nature paper uses 50M) and compare against a random baseline.

Figures -> figures/pong/.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from rl_lab.configs import PONG, SEEDS
from rl_lab.envs.atari_wrappers import make_atari_env
from rl_lab.algorithms.deep.dqn import train_dqn
from rl_lab.utils.plotting import plot_learning_curves
from rl_lab.utils.seeding import set_seed

FIG_DIR = "figures/pong"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If this prints cpu on the Alienware, the wrong interpreter is active (needs cu128).
print(f"[run_pong] python     : {sys.executable}")
print(f"[run_pong] torch      : {torch.__version__}  (cuda build: {torch.version.cuda})")
print(f"[run_pong] DEVICE     : {DEVICE}"
      + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
if DEVICE == "cpu":
    print("[run_pong] WARNING: running on CPU - training will be slow.")

TOTAL_STEPS = 200_000  # ~0.4% of the Nature budget; enough to clear the random baseline


def make_pong():
    return make_atari_env(PONG["env_name"])


def run_config(label, seeds=SEEDS, **dqn_kwargs):
    """Train one DQN configuration on all seeds, save checkpoints, return the curves.

    Not run_multi_seed: the DQN loop is step-based, so each seed ends up with a
    DIFFERENT number of episodes (an agent that learns rallies longer plays fewer
    games in the same 200k steps). I pad the short rows with NaN to get a rectangular
    array. Note the plotting takes a plain mean, so a curve is drawn only up to the
    shortest seed: every plotted point is a genuine 3-seed mean.
    """
    print(f"\n--- {label} ---")
    all_returns = []

    for seed in seeds:
        print(f"  Seed {seed}...")
        set_seed(seed)
        env = make_pong()
        env.reset(seed=seed)
        # The action space owns a SEPARATE RNG that neither set_seed nor reset(seed=)
        # touches. train_dqn draws every exploratory action from it, and with
        # learning_starts=10k and eps~1 early on, that is essentially all the experience
        # that fills the buffer. Without this line the run is not reproducible at all.
        env.action_space.seed(seed)

        returns, lengths, model = train_dqn(
            env=env,
            total_steps=TOTAL_STEPS,
            device=DEVICE,
            log_interval=20,
            **dqn_kwargs,
        )
        all_returns.append(returns)
        env.close()

        ckpt_dir = f"results/pong/{label.replace(' ', '_')}"
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{ckpt_dir}/model_seed{seed}.pt")

    max_len = max(len(r) for r in all_returns)
    padded = np.full((len(seeds), max_len), np.nan)
    for i, r in enumerate(all_returns):
        padded[i, :len(r)] = r

    return padded


def random_baseline(seeds=SEEDS, n_episodes=50):
    """Uniform-random policy: the reference any configuration must beat to have learned."""
    print("\n--- Random baseline ---")
    all_returns = []

    for seed in seeds:
        set_seed(seed)
        env = make_pong()
        env.reset(seed=seed)
        env.action_space.seed(seed)  # the baseline IS action_space.sample(): seed it
        ep_returns = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ret = 0.0
            while not done:
                obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                done = terminated or truncated
                ret += reward
            ep_returns.append(ret)

        all_returns.append(ep_returns)
        env.close()

    mean_ret = np.mean(all_returns)
    print(f"  Random policy mean return: {mean_ret:.1f}")
    return mean_ret


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("results/pong", exist_ok=True)

    # Identical for every configuration: only the ablation flags change, otherwise the
    # comparison would confound the flag with the hyperparameters.
    common = dict(
        gamma=0.99,
        lr=1e-4,
        buffer_capacity=100_000,      # Nature uses 1M -> ~28 GB, more than my RAM
        batch_size=32,
        target_update_freq=1000,      # Nature: 10k, scaled down with the budget
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_steps=100_000,  # decay over the first half of training
        learning_starts=10_000,
    )

    random_mean = random_baseline()

    full = run_config("Full DQN",
                      use_target_network=True, use_replay=True,
                      use_double_dqn=False, **common)

    no_target = run_config("No target net",
                           use_target_network=False, use_replay=True,
                           use_double_dqn=False, **common)

    # Without a buffer there is nothing to pre-fill, so learning can start at once.
    no_replay_common = dict(common)
    no_replay_common["learning_starts"] = 32

    no_replay = run_config("No replay",
                           use_target_network=True, use_replay=False,
                           use_double_dqn=False, **no_replay_common)

    no_both = run_config("No target + no replay",
                         use_target_network=False, use_replay=False,
                         use_double_dqn=False, **no_replay_common)

    double = run_config("Double DQN",
                        use_target_network=True, use_replay=True,
                        use_double_dqn=True, **common)

    # window=10, not the default 40: the DQN only produces ~150 episodes per seed.
    plot_learning_curves(
        {"Full DQN": full, "No replay": no_replay},
        title="Pong - DQN: with vs without experience replay",
        save_path=f"{FIG_DIR}/ablation_replay.png",
        window=10,
    )

    plot_learning_curves(
        {"Full DQN": full, "No target net": no_target},
        title="Pong - DQN: with vs without target network",
        save_path=f"{FIG_DIR}/ablation_target.png",
        window=10,
    )

    plot_learning_curves(
        {
            "Full DQN": full,
            "No target net": no_target,
            "No replay": no_replay,
            "No target + no replay": no_both,
            "Double DQN (bonus)": double,
        },
        title=f"Pong - DQN ablation study (random baseline ~ {random_mean:.0f})",
        save_path=f"{FIG_DIR}/full_comparison.png",
        window=10,
    )

    np.savez("results/pong/ablation.npz",
             full=full, no_target=no_target, no_replay=no_replay,
             no_both=no_both, double=double)

    print(f"\nDone. Figures saved in {FIG_DIR}/")
    print(f"Random baseline mean: {random_mean:.1f}")


if __name__ == "__main__":
    main()
