#!/usr/bin/env python3
"""Pong DQN experiments — Env 3 (Atari / Deep RL).

Origin: NEW. Ablation study design follows the assignment requirements:
"confrontare DQN con vs senza target network / experience replay."
Double DQN (bonus) added as a 5th configuration.

Implements the required DQN ablation study:
  1. Full DQN (experience replay + target network)
  2. No target network (same object for online and target → moving target)
  3. No experience replay (online learning, correlated updates)
  4. No target + no replay (worst case: both problems at once)

Plus the bonus:
  5. Double DQN (decouples action selection from evaluation)

Each configuration is run for 3 seeds.
The goal is NOT high performance — just showing learning vs random.
Assignment: "Non è richiesto ottenere performance elevate. È sufficiente
mostrare apprendimento rispetto a policy casuale."

Figures saved to figures/pong/.
"""

import sys, os
# Add project root to Python path so `from rl_lab import ...` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from rl_lab.configs import PONG, SEEDS
from rl_lab.envs.atari_wrappers import make_atari_env
from rl_lab.algorithms.deep.dqn import train_dqn
from rl_lab.utils.plotting import plot_learning_curves
from rl_lab.utils.seeding import set_seed

FIG_DIR = "figures/pong"
# Use GPU if available (RTX 5060); falls back to CPU otherwise
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

<<<<<<< Updated upstream
# Startup banner — makes it obvious which interpreter/device is actually in use.
# If DEVICE prints "cpu", you launched with the wrong Python; use the cu128 venv:
#   .venv\Scripts\python.exe scripts\run_pong.py
print(f"[run_pong] python     : {sys.executable}")
print(f"[run_pong] torch      : {torch.__version__}  (cuda build: {torch.version.cuda})")
print(f"[run_pong] cuda avail : {torch.cuda.is_available()}")
print(f"[run_pong] DEVICE     : {DEVICE}"
      + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
if DEVICE == "cpu":
    print("[run_pong] WARNING: running on CPU. Launch with the cu128 venv:")
    print("[run_pong]   .venv\\Scripts\\python.exe scripts\\run_pong.py")

# Training budget — short enough to be feasible, long enough to show learning
=======
# Training budget — short enough to be feasible, long enough to show learning.
# Nature paper uses 50M steps; 200k is ~0.4% of that.
>>>>>>> Stashed changes
TOTAL_STEPS = 200_000  # ~50 min on RTX 5060; increase if time allows


def make_pong():
    """Create a preprocessed Pong environment (grayscale, 84×84, 4-frame stack)."""
    return make_atari_env(PONG["env_name"])


# -----------------------------------------------------------------------
# Run a single DQN configuration over multiple seeds
# -----------------------------------------------------------------------
def run_config(label, seeds=SEEDS, **dqn_kwargs):
    """Train DQN with given kwargs over multiple seeds, return all returns.

    Unlike run_multi_seed (which works for tabular algorithms), this handles
    DQN specifics: model checkpoint saving, and padding to equal length
    (different seeds may have different numbers of episodes).
    """
    print(f"\n--- {label} ---")
    all_returns = []

    for seed in seeds:
        print(f"  Seed {seed}...")
        set_seed(seed)
        env = make_pong()
        env.reset(seed=seed)

        # train_dqn returns (episode_returns, episode_lengths, trained_model)
        returns, lengths, model = train_dqn(
            env=env,
            total_steps=TOTAL_STEPS,
            device=DEVICE,
            log_interval=20,  # print progress every 20 episodes
            **dqn_kwargs,
        )
        all_returns.append(returns)
        env.close()

        # Save model weights for potential later analysis or visualization
        ckpt_dir = f"results/pong/{label.replace(' ', '_')}"
        os.makedirs(ckpt_dir, exist_ok=True)
        # state_dict() saves only the weights (not optimizer state) — smaller file
        torch.save(model.state_dict(),
                   f"{ckpt_dir}/model_seed{seed}.pt")

    # Pad to same length: different seeds may produce different episode counts
    # (the step budget is fixed at TOTAL_STEPS, but episode lengths vary)
    max_len = max(len(r) for r in all_returns)
    # np.nan for missing values → nanmean/nanstd in plotting will ignore them
    padded = np.full((len(seeds), max_len), np.nan)
    for i, r in enumerate(all_returns):
        padded[i, :len(r)] = r  # fill in actual data, leave rest as NaN

    return padded


# -----------------------------------------------------------------------
# Random baseline
# -----------------------------------------------------------------------
def random_baseline(seeds=SEEDS, n_episodes=50):
    """Evaluate a random policy on Pong for comparison.

    The random baseline gives a reference point: any DQN configuration
    that beats this has "learned something." On Pong, a random policy
    gets approximately -21 (loses every point).
    """
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
            ret = 0.0  # Pong reward: +1 per point scored, -1 per point conceded
            while not done:
                action = env.action_space.sample()  # uniformly random action
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ret += reward  # game score = sum of +1/-1 rewards
            ep_returns.append(ret)

        all_returns.append(ep_returns)
        env.close()

    # Expected: ≈ -21 (random agent loses almost every point in a game to 21)
    mean_ret = np.mean(all_returns)
    print(f"  Random policy mean return: {mean_ret:.1f}")
    return mean_ret


# -----------------------------------------------------------------------
# Main experiments
# -----------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("results/pong", exist_ok=True)

    # Shared hyperparameters — SAME across all configurations for a fair ablation.
    # Only the ablation flags (use_target_network, use_replay, use_double_dqn) differ.
    common = dict(
        gamma=0.99,                 # discount factor (standard for Atari)
        lr=1e-4,                    # Adam learning rate (Nature paper uses RMSprop 2.5e-4)
        buffer_capacity=100_000,    # 100k transitions; Nature uses 1M but that needs ~28 GB RAM
        batch_size=32,              # mini-batch size for gradient updates (same as Nature)
        target_update_freq=1000,    # hard update θ⁻ ← θ every 1000 steps (Nature: 10k)
        epsilon_start=1.0,          # start with 100% random actions (full exploration)
        epsilon_end=0.02,           # end with 2% random actions (mostly exploitation)
        epsilon_decay_steps=100_000,  # linear decay over 50% of training steps
        learning_starts=10_000,     # fill buffer with 10k transitions before first gradient step
    )

    # --- Random baseline ---
    random_mean = random_baseline()

    # --- 1. Full DQN (both innovations enabled) ---
    full = run_config("Full DQN",
                      use_target_network=True, use_replay=True,
                      use_double_dqn=False, **common)

    # --- 2. No target network (ablation: moving target problem) ---
    no_target = run_config("No target net",
                           use_target_network=False, use_replay=True,
                           use_double_dqn=False, **common)

    # --- 3. No experience replay (ablation: correlated updates) ---
    no_replay_common = dict(common)
    # Without replay, there's no large buffer to fill → can start learning immediately
    no_replay_common["learning_starts"] = 32
    no_replay = run_config("No replay",
                           use_target_network=True, use_replay=False,
                           use_double_dqn=False, **no_replay_common)

    # --- 4. No target + no replay (worst case: both problems) ---
    no_both = run_config("No target + no replay",
                         use_target_network=False, use_replay=False,
                         use_double_dqn=False, **no_replay_common)

    # --- 5. Double DQN (bonus: reduces maximization bias) ---
    double = run_config("Double DQN",
                        use_target_network=True, use_replay=True,
                        use_double_dqn=True, **common)

    # --- Generate ablation plots ---
    # Each plot isolates one factor to show its impact on learning.
    # window=10 instead of default 20 because DQN produces fewer episodes.

    # Ablation 1: experience replay ON vs OFF (with target network in both)
    # Expected: without replay, learning is slower/noisier due to correlated updates
    plot_learning_curves(
        {"Full DQN": full, "No replay": no_replay},
        title="Pong — DQN: with vs without experience replay",
        save_path=f"{FIG_DIR}/ablation_replay.png",
        window=10,
    )

    # Ablation 2: target network ON vs OFF (with replay in both)
    # Expected: without target net, the "moving target" causes oscillation
    plot_learning_curves(
        {"Full DQN": full, "No target net": no_target},
        title="Pong — DQN: with vs without target network",
        save_path=f"{FIG_DIR}/ablation_target.png",
        window=10,
    )

    # Full comparison: all 5 configurations on one plot
    # The random baseline (≈ -21) provides the "no learning" reference
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

    # Save raw data so plots can be regenerated without re-training
    np.savez("results/pong/ablation.npz",
             full=full, no_target=no_target, no_replay=no_replay,
             no_both=no_both, double=double)

    print(f"\nDone. Figures saved in {FIG_DIR}/")
    print(f"Random baseline mean: {random_mean:.1f}")


if __name__ == "__main__":
    main()
