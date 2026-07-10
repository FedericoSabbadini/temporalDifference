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
import ale_py

# Register the ALE/* environments with Gymnasium. Modern ale-py (>=0.10) no
# longer auto-registers on import of gymnasium, so this is required or
# gym.make("ALE/Pong-v5") raises NamespaceNotFound.
gym.register_envs(ale_py)


def make_atari_env(env_name="ALE/Pong-v5", render_mode=None):
    """Create an Atari env with standard DQN preprocessing.

    Final observation shape is (4, 84, 84): 4 stacked grayscale frames,
    channel-first (PyTorch NCHW convention for the Nature CNN).
    """
    # frameskip=1 disables the -v5 env's built-in frame-skip so that
    # AtariPreprocessing is the single source of the 4-frame skip.
    env = gym.make(env_name, render_mode=render_mode, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,  # keep uint8 to save memory in replay buffer
    )
    # FrameStack was renamed to FrameStackObservation in Gymnasium 1.x.
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env
