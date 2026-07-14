"""Q-network for DQN.

Origin: NEW (Standard architecture). Layer sizes replicate Mnih et al. 2015, Table 1;
the PyTorch implementation is my own.
"""

import torch.nn as nn
import torch.nn.functional as F


class NatureCNN(nn.Module):
    """Nature DQN convnet: (batch, 4, 84, 84) -> one Q-value per action.

    The 4 input channels are 4 consecutive frames: a single frame shows where the ball
    is but not where it is going, so motion has to come from the stack.

    The net emits Q(s, .) for all actions in one forward pass, so acting is a single
    argmax instead of one pass per action.
    """

    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)   # 84 -> 20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # 20 -> 9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # 9 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)                    # 3136 -> 512
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)  # no activation: Q-values are unbounded
