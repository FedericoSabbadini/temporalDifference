#!/usr/bin/env python3
"""Pong DQN experiments — Env 3 (Atari / Deep RL).

Implements the required DQN ablation study:
  1. Full DQN (experience replay + target network)
  2. No target network
  3. No experience replay
  4. No target + no replay

Plus the bonus:
  5. Double DQN

Each configuration is run for 3 seeds.
The goal is NOT high performance — just showing learning vs random.

Figures saved to figures/pong/.
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

# Training budget — short enough to be feasible, long enough to show learning
TOTAL_STEPS = 200_000  # ~50 min on RTX 5060; increase if time allows


def make_pong():
    return make_atari_env(PONG["env_name"])


# -----------------------------------------------------------------------
# Run a single DQN configuration over multiple seeds
# -----------------------------------------------------------------------
def run_config(label, seeds=SEEDS, **dqn_kwargs):
    """Train DQN with given kwargs over multiple seeds, return all returns."""
    print(f"\n--- {label} ---")
    all_returns = []

    for seed in seeds:
        print(f"  Seed {seed}...")
        set_seed(seed)
        env = make_pong()
        env.reset(seed=seed)

        returns, lengths, model = train_dqn(
            env=env,
            total_steps=TOTAL_STEPS,
            device=DEVICE,
            log_interval=20,
            **dqn_kwargs,
        )
        all_returns.append(returns)
        env.close()

        # Save model checkpoint
        ckpt_dir = f"results/pong/{label.replace(' ', '_')}"
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(),
                   f"{ckpt_dir}/model_seed{seed}.pt")

    # Pad to same length (episodes may differ across seeds)
    max_len = max(len(r) for r in all_returns)
    padded = np.full((len(seeds), max_len), np.nan)
    for i, r in enumerate(all_returns):
        padded[i, :len(r)] = r

    return padded


# -----------------------------------------------------------------------
# Random baseline
# -----------------------------------------------------------------------
def random_baseline(seeds=SEEDS, n_episodes=50):
    """Evaluate a random policy on Pong for comparison."""
    print("\n--- Random baseline ---")
    all_returns = []

    for seed in seeds:
        set_seed(seed)
        env = make_pong()
        env.reset(seed=seed)
        ep_returns = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ret = 0.0
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ret += reward
            ep_returns.append(ret)

        all_returns.append(ep_returns)
        env.close()

    mean_ret = np.mean(all_returns)
    print(f"  Random policy mean return: {mean_ret:.1f}")
    return mean_ret


# -----------------------------------------------------------------------
# Main experiments
# -----------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("results/pong", exist_ok=True)

    # Shared hyperparameters
    common = dict(
        gamma=0.99,
        lr=1e-4,
        buffer_capacity=100_000,
        batch_size=32,
        target_update_freq=1000,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_steps=100_000,
        learning_starts=10_000,
    )

    # --- Random baseline ---
    random_mean = random_baseline()

    # --- 1. Full DQN ---
    full = run_config("Full DQN",
                      use_target_network=True, use_replay=True,
                      use_double_dqn=False, **common)

    # --- 2. No target network ---
    no_target = run_config("No target net",
                           use_target_network=False, use_replay=True,
                           use_double_dqn=False, **common)

    # --- 3. No experience replay ---
    no_replay_common = dict(common)
    no_replay_common["learning_starts"] = 32  # can start earlier
    no_replay = run_config("No replay",
                           use_target_network=True, use_replay=False,
                           use_double_dqn=False, **no_replay_common)

    # --- 4. No target + no replay ---
    no_both = run_config("No target + no replay",
                         use_target_network=False, use_replay=False,
                         use_double_dqn=False, **no_replay_common)

    # --- 5. Double DQN (bonus) ---
    double = run_config("Double DQN",
                        use_target_network=True, use_replay=True,
                        use_double_dqn=True, **common)

    # --- Ablation plot: replay ---
    plot_learning_curves(
        {"Full DQN": full, "No replay": no_replay},
        title="Pong — DQN: with vs without experience replay",
        save_path=f"{FIG_DIR}/ablation_replay.png",
        window=10,
    )

    # --- Ablation plot: target network ---
    plot_learning_curves(
        {"Full DQN": full, "No target net": no_target},
        title="Pong — DQN: with vs without target network",
        save_path=f"{FIG_DIR}/ablation_target.png",
        window=10,
    )

    # --- Full comparison ---
    plot_learning_curves(
        {
            "Full DQN": full,
            "No target net": no_target,
            "No replay": no_replay,
            "No target + no replay": no_both,
            "Double DQN (bonus)": double,
        },
        title=f"Pong — DQN ablation study (random baseline ≈ {random_mean:.0f})",
        save_path=f"{FIG_DIR}/full_comparison.png",
        window=10,
    )

    # Save raw data
    np.savez("results/pong/ablation.npz",
             full=full, no_target=no_target, no_replay=no_replay,
             no_both=no_both, double=double)

    print(f"\nDone. Figures saved in {FIG_DIR}/")
    print(f"Random baseline mean: {random_mean:.1f}")


if __name__ == "__main__":
    main()
