"""Experience Replay buffer for DQN (Mnih et al. 2015, §Methods).

Origin: NEW. The concept comes from the Nature DQN paper. Implementation
is minimal, using collections.deque for a fixed-capacity FIFO buffer.

Why Experience Replay is needed:
In standard online learning, the agent trains on transitions (S,A,R,S') in
the order they are experienced. This causes two problems:

1. TEMPORAL CORRELATION: consecutive transitions are strongly correlated
   (state S' at step t becomes state S at step t+1). Gradient descent
   assumes i.i.d. samples - correlated data destabilizes learning.

2. CATASTROPHIC FORGETTING: the agent only trains on recent experiences.
   It "forgets" earlier behavior patterns as new ones overwrite them.

Solution: store transitions in a buffer and sample RANDOM mini-batches.
This breaks temporal correlations and allows each transition to be used
multiple times (data efficiency).

For Atari, states are stored as uint8 (0-255) to save memory.
Normalization to [0, 1] float32 happens at sampling time in dqn.py.
"""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """Fixed-capacity FIFO buffer with uniform random sampling.

    When full, the oldest transition is evicted (deque handles this).
    """

    def __init__(self, capacity):
        # deque with maxlen: when full, oldest transition is automatically dropped (FIFO)
        self.buffer = deque(maxlen=capacity) # deque is faster than list for FIFO operations

    def push(self, state, action, reward, next_state, done):
        """Store a single transition (S, A, R, S', done)."""
        self.buffer.append((
            # copy=False avoids unnecessary copy if state is already an array
            np.array(state, copy=False),
            action,   # integer action index
            reward,   # scalar reward received
            np.array(next_state, copy=False),
            done,     # boolean: True if next_state is terminal
        ))

    def sample(self, batch_size):
        """Sample a random mini-batch of transitions.

        Random sampling is the key to Experience Replay:
        it breaks the temporal correlation between consecutive transitions,
        which would otherwise destabilize gradient descent (S&B §8.4.1).

        Returns numpy arrays for efficient conversion to PyTorch tensors.
        """
        # random.sample draws without replacement from the buffer
        batch = random.sample(self.buffer, batch_size)
        # zip(*batch) transposes list-of-tuples into tuple-of-lists
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),  # shape: (batch, *obs_shape) = (batch, 4, 84, 84) for Atari
            np.array(actions, dtype=np.int64),   # int64 required by torch.gather
            np.array(rewards, dtype=np.float32), # float32 for GPU efficiency
            np.array(next_states),  # same shape as states = (batch, *obs_shape) = (batch, 4, 84, 84) for Atari
            np.array(dones, dtype=np.float32),   # 0.0 or 1.0 for masking
        )

    def __len__(self):
        return len(self.buffer)  # used to check if buffer has enough samples
