"""Atari preprocessing wrappers (standard from the Nature DQN paper).

These are *environment* wrappers provided by Gymnasium, NOT high-level RL
library code — so they are allowed by the assignment.

Pipeline:
  1. NoopReset — random no-ops at episode start (up to 30)
  2. Frame-skip 4 — repeat each action for 4 frames, max over last 2
  3. Grayscale + resize to 84×84
  4. Stack 4 consecutive frames → observation shape (84, 84, 4)
"""

import gymnasium as gym


def make_atari_env(env_name="ALE/Pong-v5", render_mode=None):
    """Create an Atari env with standard DQN preprocessing."""
    env = gym.make(env_name, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,  # keep uint8 to save memory in replay buffer
    )
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env
