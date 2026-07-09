"""Off-policy TD control: Q-Learning.

Update rule (Bellman optimality):
    Q(S,A) ← Q(S,A) + α [ R + γ max_a Q(S',a) - Q(S,A) ]

Key property: the target uses max over NEXT-STATE actions (greedy w.r.t. Q),
regardless of the action actually taken. This makes Q-Learning OFF-POLICY —
it learns the optimal policy while following an exploratory one (ε-greedy).

Ported from the course Esercitazione_RL / control.py, adapted to Gymnasium API.
"""

import numpy as np
from collections import defaultdict

from rl_lab.policies import epsilon_greedy_action


def q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1,
               max_steps=1000):
    """Train Q-Learning on a discrete (or discretized) Gymnasium env.

    Args:
        env: Gymnasium environment (observation must be hashable, e.g. tuple).
        episodes: number of training episodes.
        alpha: learning rate.
        gamma: discount factor.
        epsilon: exploration rate (constant ε-greedy).
        max_steps: max steps per episode (safety cap).

    Returns:
        Q: learned Q-table (defaultdict).
        episode_returns: list of total rewards per episode.
        episode_lengths: list of episode lengths.
    """
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
    episode_returns = []
    episode_lengths = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = epsilon_greedy_action(Q, state, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Q-Learning update: greedy target (max over next actions)
            td_target = reward + gamma * np.max(Q[next_state]) * (not done)
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state
            if done:
                break

        episode_returns.append(total_reward)
        episode_lengths.append(step + 1)

    return Q, episode_returns, episode_lengths
