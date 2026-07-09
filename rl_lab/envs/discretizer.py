"""Dimension-agnostic state discretizer for tabular RL on continuous envs.

CartPole has 4 continuous dims, Acrobot has 6. Tabular methods (Q-tables)
need a finite, hashable state. This wrapper clips each dimension to a
configured range and bins it into equal-width intervals, producing a tuple
of bin indices usable as a dictionary key.
"""

import numpy as np
import gymnasium as gym


class DiscretizedEnv(gym.ObservationWrapper):
    """Wrap a continuous Gymnasium env so observations become discrete tuples.

    Args:
        env: a Gymnasium environment with Box observation space.
        n_bins: list of ints — number of bins per dimension.
        clip_low: list of floats — lower clip bound per dimension.
        clip_high: list of floats — upper clip bound per dimension.
    """

    def __init__(self, env, n_bins, clip_low, clip_high):
        super().__init__(env)
        self.n_bins = np.array(n_bins, dtype=int)
        self.clip_low = np.array(clip_low, dtype=np.float64)
        self.clip_high = np.array(clip_high, dtype=np.float64)
        self.bin_widths = (self.clip_high - self.clip_low) / self.n_bins

    def observation(self, obs):
        """Convert a continuous observation to a tuple of bin indices."""
        clipped = np.clip(obs, self.clip_low, self.clip_high)
        scaled = (clipped - self.clip_low) / self.bin_widths
        bins = np.clip(scaled.astype(int), 0, self.n_bins - 1)
        return tuple(bins)
