"""Atari preprocessing wrappers (standard from the Nature DQN paper).

Origin: ADAPTED from Tutorial_Gymnasium.json (professor's tutorial showing
how to create Atari envs with Gymnasium). The wrappers themselves are
Gymnasium built-in (not SB3), so they are allowed by the assignment.

Pipeline (Mnih et al. 2015, §Methods):
  1. NoopReset — random no-ops at episode start (up to 30).
     Prevents the agent from memorizing fixed starting conditions.
  2. Frame-skip 4 — repeat each action for 4 frames, max over last 2.
     Reduces computation 4×. Max-over-2 handles Atari sprite flickering.
  3. Grayscale + resize to 84×84.
     Color is irrelevant for most games; 84×84 is the paper standard.
  4. Stack 4 consecutive frames → observation shape (84, 84, 4).
     A single frame is ambiguous about direction of motion (e.g., which
     way is the ball moving?). 4 frames let the network infer velocity.

Note: scale_obs=False keeps frames as uint8 (0-255) instead of float32 (0-1).
This saves ~4× memory in the replay buffer. Normalization happens at training
time in dqn.py (_obs_to_tensor / _batch_to_tensor).
"""

import gymnasium as gym
import ale_py

# Register the ALE/* environments with Gymnasium. Modern ale-py (>=0.10) no
# longer auto-registers on import of gymnasium, so this is required or
# gym.make("ALE/Pong-v5") raises NamespaceNotFound.
gym.register_envs(ale_py)


def make_atari_env(env_name="ALE/Pong-v5", render_mode=None):
<<<<<<< Updated upstream
    """Create an Atari env with standard DQN preprocessing.

    Final observation shape is (4, 84, 84): 4 stacked grayscale frames,
    channel-first (PyTorch NCHW convention for the Nature CNN).
    """
    # frameskip=1 disables the -v5 env's built-in frame-skip so that
    # AtariPreprocessing is the single source of the 4-frame skip.
    env = gym.make(env_name, render_mode=render_mode, frameskip=1)
=======
    """Create an Atari env with standard DQN preprocessing."""
    # Create the base Atari environment via the Arcade Learning Environment
    env = gym.make(env_name, render_mode=render_mode)
    # AtariPreprocessing applies the standard preprocessing from Mnih et al. (2015):
>>>>>>> Stashed changes
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,           # random 1-30 no-op actions at episode start (randomize initial state)
        frame_skip=4,          # repeat each action for 4 frames (4× speedup); max over last 2 frames
        screen_size=84,        # resize to 84×84 pixels (reduces computation, matches paper)
        terminal_on_life_loss=False,  # don't end episode on life loss (full-game episodes)
        grayscale_obs=True,    # convert RGB → grayscale (color is irrelevant for gameplay)
        scale_obs=False,       # keep uint8 [0-255] to save ~4× memory in replay buffer
    )
<<<<<<< Updated upstream
    # FrameStack was renamed to FrameStackObservation in Gymnasium 1.x.
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
=======
    # Stack 4 consecutive frames to give the network temporal information
    # (e.g. ball direction in Pong requires seeing motion across frames)
    env = gym.wrappers.FrameStack(env, num_stack=4)
>>>>>>> Stashed changes
    return env
