"""On-policy TD control: SARSA.

Update rule:
    Q(S,A) ← Q(S,A) + α [ R + γ Q(S',A') - Q(S,A) ]

Key difference from Q-Learning: the target uses the ACTUAL next action A'
chosen by the ε-greedy policy, not the greedy max. This makes SARSA
ON-POLICY — it evaluates and improves the policy it is currently following.

Consequence: SARSA learns a "safer" policy that accounts for its own
exploration noise. Classic example: on CliffWalking, SARSA avoids the
cliff edge while Q-Learning walks right along it (Sutton & Barto §6.5).
"""

import numpy as np
from collections import defaultdict

from rl_lab.policies import epsilon_greedy_action


def sarsa(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1,
          max_steps=1000):
    """Train SARSA on a discrete (or discretized) Gymnasium env.

    Returns:
        Q, episode_returns, episode_lengths (same format as q_learning).
    """
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
    episode_returns = []
    episode_lengths = []

    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy_action(Q, state, epsilon, n_actions)
        total_reward = 0.0

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Choose next action BEFORE updating (S,A,R,S',A' → SARSA)
            next_action = epsilon_greedy_action(Q, next_state, epsilon,
                                                n_actions)

            # SARSA update: use the actual next action
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state
            action = next_action
            if done:
                break

        episode_returns.append(total_reward)
        episode_lengths.append(step + 1)

    return Q, episode_returns, episode_lengths
