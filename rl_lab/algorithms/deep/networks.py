"""Q-network architectures for DQN.

Origin: NEW (Standard). Architectures replicate the Nature DQN paper
(Mnih et al. 2015, Table 1). The exact layer sizes are from the paper.
PyTorch implementation is original.

MLP: for environments with small vector observations (CartPole debug).
     Not in the paper - used to validate the DQN training loop on a
     simple env before moving to the expensive Atari CNN.

NatureCNN: the convolutional network from Mnih et al. (2015) for Atari.
     Input: (batch, 4, 84, 84) - 4 stacked grayscale frames.
     Output: Q(s, a) for each of the n_actions.

Both networks output Q-values for ALL actions simultaneously.
To select an action: action = argmax Q(s, :).
This is more efficient than computing Q(s,a) for each action separately.
"""

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Two-hidden-layer MLP for vector-observation envs (CartPole debug).

    Architecture: obs_dim to 128 to 128 to n_actions
    All layers use ReLU activation except the output (raw Q-values).
    """

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)   # input to hidden (e.g. 4 to 128 for CartPole)
        self.fc2 = nn.Linear(hidden, hidden)     # hidden to hidden (128 to 128)
        self.out = nn.Linear(hidden, n_actions)  # hidden to Q-values (128 to 2 for CartPole)

    def forward(self, x):
        # ReLU activation: max(0, x) - introduces nonlinearity between layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer has NO activation - Q-values can be any real number (positive or negative)
        return self.out(x)


class NatureCNN(nn.Module):
    """Nature DQN convolutional network (Mnih et al. 2015, Table 1).

    Architecture:
      Conv2d(4 to 32 channels, 8×8, stride 4), then ReLU
      Conv2d(32 to 64 channels, 4×4, stride 2), then ReLU
      Conv2d(64 to 64 channels, 3×3, stride 1), then ReLU
      Flatten, Linear(3136 to 512), ReLU, Linear(512 to n_actions)

    Input shape: (batch, 4, 84, 84) - 4 stacked 84×84 grayscale frames.
    The 4 channels are consecutive frames, giving the network information
    about motion (e.g., ball direction in Pong).

    Output: Q-value for each action (no activation - raw values).
    """

    def __init__(self, n_actions):
        super().__init__()
        # Conv layers extract spatial features (edges, objects, motion) from frames
        # Each conv reduces spatial dimensions: output_size = (input - kernel) / stride + 1
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)   # 84 to 20 spatial, 4 input channels = 4 stacked frames
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # 20 to 9, increasing channel depth
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # 9 to 7, same depth, smaller spatial
        # After convolutions on 84×84 input: 64 channels × 7 × 7 = 3136 features
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # 3136 to 512 (high-level representation)
        self.out = nn.Linear(512, n_actions)     # 512 to one Q-value per action

    def forward(self, x):
        # x shape: (batch, 4, 84, 84) - 4 stacked grayscale frames
        x = F.relu(self.conv1(x))  # - (batch, 32, 20, 20)
        x = F.relu(self.conv2(x))  # - (batch, 64, 9, 9)
        x = F.relu(self.conv3(x))  # - (batch, 64, 7, 7)
        # Flatten spatial dims into a single vector per sample
        x = x.reshape(x.size(0), -1)  # - (batch, 3136)
        x = F.relu(self.fc1(x))       # - (batch, 512)
        # No activation on output - Q-values are unconstrained real numbers
        return self.out(x)             # - (batch, n_actions)
