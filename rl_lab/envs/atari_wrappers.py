import gymnasium as gym
import ale_py
import rl_lab.configs as conf

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
        noop_max=conf.PONG["noop_max"],
        frame_skip=conf.PONG["frame_skip"],
        screen_size=conf.PONG["screen_size"],
        terminal_on_life_loss=False,
        grayscale_obs=conf.PONG["greyscale_obs"],
        scale_obs=False,
    )

    # Axis 0 of the stack is time (oldest -> newest); the result is already
    # channel-first, so it feeds the CNN with no transpose.
    return gym.wrappers.FrameStackObservation(env, stack_size=conf.PONG["stack_size"])
