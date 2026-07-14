"""Experience Replay buffer for DQN (Mnih et al. 2015, §Methods).

Origin: NEW. The concept is from the Nature paper; the code is mine (a deque FIFO).

Replay exists to fix two things about learning from the stream as it arrives:
consecutive transitions are heavily correlated (S' at step t is S at step t+1), which
violates the i.i.d. assumption of SGD, and recent experience would otherwise overwrite
older behaviour. Sampling uniformly at random from a large buffer breaks the
correlation and lets each transition be reused. My ablation shows this is not a
refinement but the component the agent cannot learn without.

Frames stay uint8 in here (100k Atari transitions as float32 would not fit in RAM);
they are normalized to [0,1] only when a batch becomes a tensor, in dqn.py.
"""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """Fixed-capacity FIFO buffer with uniform random sampling."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # maxlen evicts the oldest for free

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
