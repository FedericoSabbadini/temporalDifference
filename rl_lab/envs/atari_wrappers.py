"""
Atari preprocessing pipeline used by the original Nature DQN
(Mnih et al., Nature 2015).

Unlike CartPole and Acrobot, Atari environments require additional
preprocessing before observations can be fed to a convolutional neural
network (CNN). Raw Atari observations are large RGB images that contain
more information than necessary for learning and are expensive to process.

This environment reproduces the standard preprocessing pipeline proposed
in the original DQN paper using Gymnasium's official wrappers.

Origin:
    Adapted from the professor's Gymnasium tutorial. Only built-in
    Gymnasium wrappers are used (no Stable-Baselines3 wrappers),
    therefore the implementation fully complies with the assignment.

Pipeline
--------
1. NoopReset (up to 30 random NOOP actions)
   At the beginning of every episode, the environment performs a random
   number of "do nothing" actions. This prevents every game from starting
   from exactly the same situation, reducing overfitting to a fixed
   opening sequence.

2. Frame Skip = 4
   The agent selects one action, but that action is repeated for the next
   four Atari frames. Consequently, the agent makes one decision every
   four game frames instead of every frame, reducing computation by
   approximately 4×.

   During these four frames the game evolves normally (the ball and the
   paddles continue moving); only the selected action remains unchanged.

   AtariPreprocessing also computes the pixel-wise maximum over the last
   two frames. This compensates for Atari rendering artefacts where some
   objects (such as the Pong ball) may temporarily disappear on
   alternating frames ("sprite flickering").

3. Grayscale + Resize (84×84)
   RGB images are converted to grayscale because colour carries almost no
   useful information for Pong. Each frame is then resized to 84×84
   pixels, matching the input resolution used by the original Nature DQN.

4. Frame Stack (4 observations)
   A single image tells the agent where the ball currently is, but not
   where it is moving. Therefore, the four most recent observations are
   stacked together so the CNN can infer motion (direction and approximate
   velocity) from changes across time.

Memory optimization
-------------------
scale_obs=False keeps observations as uint8 values in [0,255] instead of
float32 values in [0,1]. This reduces replay-buffer memory by about 4×.
Normalization is performed only immediately before observations are fed
to the neural network (see dqn.py).
"""

import gymnasium as gym
import ale_py

# Register all Atari (ALE/*) environments with Gymnasium.
# Recent versions of ale-py no longer register Atari environments
# automatically, therefore without this call
# gym.make("ALE/Pong-v5") would raise NamespaceNotFound.
gym.register_envs(ale_py)


def make_atari_env(env_name="ALE/Pong-v5", render_mode=None):
    """
    Create a Pong environment with the standard Nature DQN preprocessing.

    Final observation
    -----------------
    Shape:
        (4, 84, 84)

    Meaning:
        - 4 recent grayscale observations;
        - each observation is an 84×84 image;
        - channel-first layout (PyTorch convention).

    The stacked observations provide temporal information, allowing the
    CNN to estimate the direction and approximate velocity of the ball,
    which cannot be inferred from a single image.
    """

    # Disable the environment's built-in frame skipping.
    # We want AtariPreprocessing to be the only component responsible for
    # frame skipping. Otherwise both mechanisms would be applied
    # simultaneously (e.g. 4 × 4 = 16 real frames for every chosen action).
    env = gym.make(
        env_name,
        render_mode=render_mode,
        frameskip=1,
    )

    # Apply the standard preprocessing pipeline from the Nature DQN paper.
    env = gym.wrappers.AtariPreprocessing(
        env,

        # Perform between 1 and 30 random "do nothing" (NOOP) actions at
        # the beginning of each episode. This randomizes the initial state
        # and prevents the agent from memorizing a fixed opening sequence.
        noop_max=30,

        # Repeat every chosen action for four consecutive Atari frames.
        # The game keeps evolving normally during these frames while the
        # selected action remains unchanged, reducing computation by about
        # 4×. AtariPreprocessing also returns the pixel-wise maximum over
        # the last two frames to compensate for Atari sprite flickering.
        frame_skip=4,

        # Resize every frame to 84×84 pixels, matching the standard input
        # resolution used by the original DQN architecture.
        screen_size=84,

        # Do not terminate the episode when a life (or point) is lost.
        # Episodes correspond to complete games rather than intermediate
        # events.
        terminal_on_life_loss=False,

        # Convert RGB images to grayscale. Colour is unnecessary for Pong
        # and removing it reduces the amount of information processed by
        # the CNN.
        grayscale_obs=True,

        # Keep observations as uint8 values in [0,255] instead of
        # converting them immediately to float32 in [0,1]. This reduces
        # replay-buffer memory usage by approximately 4×. Normalization is
        # performed only when observations are converted into tensors for
        # the neural network.
        scale_obs=False,
    )

    # Store the four most recent observations.
    #
    # A single image provides only the current ball position, making its
    # direction of motion ambiguous. By stacking four consecutive
    # observations, the CNN can infer both direction and approximate
    # velocity from how the ball position changes over time.
    #
    # Final observation shape:
    #     (4, 84, 84)
    #
    # where:
    #     axis 0 -> time (oldest - newest observation)
    #     axis 1 -> image height
    #     axis 2 -> image width
    env = gym.wrappers.FrameStackObservation(
        env,
        stack_size=4,
    )

    return env