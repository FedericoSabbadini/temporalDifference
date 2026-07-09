"""Experience Replay buffer for DQN.

Stores transitions (s, a, r, s', done) and samples random mini-batches.
Breaking temporal correlations via random sampling is one of the two key
innovations of the Nature DQN paper (Mnih et al. 2015).

For Atari, states are stored as uint8 to save memory — normalization
to [0, 1] float32 happens at sampling time.
"""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """Fixed-capacity FIFO buffer with uniform random sampling."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a single transition."""
        self.buffer.append((
            np.array(state, copy=False),
            action,
            reward,
            np.array(next_state, copy=False),
            done,
        ))

    def sample(self, batch_size):
        """Sample a random mini-batch of transitions.

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
