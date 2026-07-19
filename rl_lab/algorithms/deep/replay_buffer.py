import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """Fixed-capacity FIFO buffer with uniform random sampling."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) 

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, copy=False),
            action,
            reward,
            np.array(next_state, copy=False),
            done,
        ))

    def sample(self, batch_size):
        """Uniform random mini-batch, as numpy arrays ready for torch."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions, dtype=np.int64),    # int64: required by torch.gather
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),    # 0.0/1.0: masks the bootstrap term
        )

    def __len__(self):
        return len(self.buffer)
