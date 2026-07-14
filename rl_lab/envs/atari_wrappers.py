"""Atari preprocessing pipeline from the Nature DQN paper (Mnih et al. 2015).

Origin: ADAPTED from the professor's Gymnasium tutorial. Only built-in Gymnasium
wrappers are used, never Stable-Baselines3 ones, as the assignment requires.

Pipeline: random no-ops on reset (so the game does not always start identically),
frame-skip 4 with a pixel-wise max over the last two frames (Atari flickers sprites,
so the Pong ball can vanish on alternate frames), grayscale + 84x84 resize, and a
stack of the 4 most recent frames so the CNN can infer motion. Frames stay uint8 to
keep the replay buffer ~4x smaller; normalization happens in dqn.py.
"""

import gymnasium as gym
import ale_py

# Recent ale-py no longer auto-registers the ALE envs: without this,
# gym.make("ALE/Pong-v5") raises NamespaceNotFound.
gym.register_envs(ale_py)


def make_atari_env(env_name="ALE/Pong-v5", render_mode=None):
    """Pong with Nature-DQN preprocessing. Final observation: (4, 84, 84), uint8."""

    # frameskip=1 disables the env's own frame skipping. AtariPreprocessing must be the
    # only place that skips, otherwise the two compose into 4 x 4 = 16 real frames per
    # action, which silently breaks the agent's timescale.
    env = gym.make(env_name, render_mode=render_mode, frameskip=1)

    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,  # an episode is a whole game, not a single point
        grayscale_obs=True,           # colour carries nothing useful in Pong
        scale_obs=False,              # keep uint8: 4x less replay memory
    )

    # Axis 0 of the stack is time (oldest -> newest); the result is already
    # channel-first, so it feeds the CNN with no transpose.
    return gym.wrappers.FrameStackObservation(env, stack_size=4)
