"""Dimension-agnostic state discretizer for tabular RL on continuous envs.

Origin: NEW. The course exercises used pre-built discrete environments
(gridworlds). CartPole and Acrobot have continuous states, so tabular
methods (Q-tables) need a way to convert them to a finite, hashable key.

Theory (S&B §9.4 - State Aggregation):
Tabular methods store Q(s,a) for every (state, action) pair. With continuous
states, there are infinitely many entries - impossible. State aggregation
groups nearby continuous states into the same discrete bin. The result is
a tuple of bin indices usable as a dict key.

Trade-off:
- Few bins - information loss, coarse policy (e.g. 4 bins on Acrobot)
- Many bins - state space explodes exponentially (curse of dimensionality)
  CartPole: 10^4 = 10,000 states (OK)
  Acrobot:  10^6 = 1,000,000 states (too many for tabular)
"""

import numpy as np
import gymnasium as gym


class DiscretizedEnv(gym.ObservationWrapper): # ObservationWrapper is a Gymnasium class that allows you to modify the observations returned by an environment.
    """Wrap a continuous Gymnasium env so observations become discrete tuples.

    The observation pipeline:
    1. Clip each dimension to [clip_low, clip_high]
    2. Scale to [0, n_bins] range
    3. Floor to integer bin index in [0, n_bins-1]
    4. Return as tuple (hashable - usable as dict key for Q-table)

    Args:
        env: a Gymnasium environment with Box observation space.
        n_bins: list of ints - number of bins per dimension.
        clip_low: list of floats - lower clip bound per dimension.
        clip_high: list of floats - upper clip bound per dimension.
    """

    def __init__(self, env, n_bins, clip_low, clip_high):
        super().__init__(env)  # inherit all env methods (step=do action, reset=restart episode, etc.)
        self.n_bins = np.array(n_bins, dtype=int)  # e.g. [10, 10, 10, 10] for CartPole -> creates 10^4 = 10,000 discrete states (combination of all possibilities)
        self.clip_low = np.array(clip_low, dtype=np.float64)  # minimum observable values -> used to clip extreme values put by the environment (e.g. extreme velocities) in the observation space
        self.clip_high = np.array(clip_high, dtype=np.float64)  # maximum observable values -> used to clip extreme values put by the environment (e.g. extreme velocities) in the observation space
        # Width of each bin: (high - low) / n_bins, determines binning resolution
        # Every bin refers to different dimensions: we need to clip to different use case values
        self.bin_widths = (self.clip_high - self.clip_low) / self.n_bins

    def observation(self, obs):
        """Convert a continuous observation to a tuple of bin indices.

        Example for CartPole with 10 bins:
          [0.5, -0.2, 0.1, 0.3] becomes (5, 4, 6, 5)
        """
        # Step 1: clip values outside the configured range (e.g. extreme velocities)
        clipped = np.clip(obs, self.clip_low, self.clip_high)
        # Step 2: scale each dimension from [clip_low, clip_high] to [0, n_bins]
        scaled = (clipped - self.clip_low) / self.bin_widths
        # Step 3: floor to integer bin index; clamp to [0, n_bins-1] for edge cases
        bins = np.clip(scaled.astype(int), 0, self.n_bins - 1)
        # Step 4: tuple is hashable - can be used as a dict key in Q[state]
        return tuple(bins)