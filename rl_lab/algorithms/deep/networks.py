"""Q-network architectures for DQN.

MLP: for environments with small vector observations (CartPole debug).
NatureCNN: the convolutional network from Mnih et al. (2015) for Atari.
    Input: (batch, 4, 84, 84) — 4 stacked grayscale frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Two-hidden-layer MLP for vector-observation envs (CartPole debug)."""

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class NatureCNN(nn.Module):
    """Nature DQN convolutional network (Mnih et al. 2015).

    Architecture: 3 conv layers → 1 FC layer → Q-values.
    Designed for 84×84 grayscale Atari frames stacked ×4.
    """

    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # After convolutions on 84×84 input: 64 × 7 × 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)
