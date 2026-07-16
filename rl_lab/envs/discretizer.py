"""Dimension-agnostic state discretizer for tabular RL on continuous envs.

Origin: NEW. The course exercises used already-discrete gridworlds; CartPole and
Acrobot are continuous, so a Q-table needs a finite, hashable state key.

This is state aggregation (S&B §9.4): nearby continuous states collapse into the same
bin. The resolution is a trade-off, and one I measure rather than assume (see the bin
experiment on Acrobot): too few bins cannot tell distinct states apart, too many blow
the state space up exponentially (bins^dim).
"""

import numpy as np
import gymnasium as gym


class DiscretizedEnv(gym.ObservationWrapper):
    """Turn a continuous Box observation into a tuple of bin indices.

    Args:
        env: Gymnasium environment with a Box observation space.
        n_bins: bins per dimension.
        clip_low / clip_high: clip bounds per dimension (see configs.py for how I
            chose them; clipping is what makes unbounded velocity dims usable).
    """

    def __init__(self, env, n_bins, clip_low, clip_high):
        super().__init__(env)
        self.n_bins = np.array(n_bins, dtype=int)
        self.clip_low = np.array(clip_low, dtype=np.float64)
        self.clip_high = np.array(clip_high, dtype=np.float64)
        self.bin_widths = (self.clip_high - self.clip_low) / self.n_bins

    def observation(self, obs):
        """Clip -> scale -> floor to a bin index -> tuple (hashable Q-table key).
        
        An observation is an array of floats, one per dimension. Each dimension is clipped to its own range, scaled to [0, n_bins), floored to an integer bin index, and returned as a tuple. The tuple is hashable and can be used as a key in a Q-table.
        """
        clipped = np.clip(obs, self.clip_low, self.clip_high) # clip every dim to its own range
        scaled = (clipped - self.clip_low) / self.bin_widths # scale to [0, n_bins)
        bins = np.clip(scaled.astype(int), 0, self.n_bins - 1) # floor to bin index and clip to [0, n_bins-1] in case of rounding errors
        return tuple(bins) # return the tuple of bin indices, which is hashable and can be used as a key in a Q-table
