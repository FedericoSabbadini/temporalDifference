import numpy as np
from collections import defaultdict

from rl_lab.policies import epsilon_greedy_action, linear_epsilon


def sarsa(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1,
          max_steps=1000, epsilon_end=None):
    """Train SARSA. Same arguments and return format as `q_learning`."""
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
    episode_returns = []
    episode_lengths = []

    for ep in range(episodes):
        if epsilon_end is None:
            eps = epsilon
        else:
            eps = linear_epsilon(epsilon, epsilon_end, ep, max(1, episodes // 2))

        state, _ = env.reset()
        # SARSA needs A before the loop: the update consumes the full (S,A,R,S',A').
        action = epsilon_greedy_action(Q, state, eps, n_actions)
        total_reward = 0.0

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            next_action = epsilon_greedy_action(Q, next_state, eps, n_actions)

            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state
            action = next_action  # carry A' forward so it is not re-sampled
            if done:
                break

        episode_returns.append(total_reward)
        episode_lengths.append(step + 1)

    return Q, episode_returns, episode_lengths
