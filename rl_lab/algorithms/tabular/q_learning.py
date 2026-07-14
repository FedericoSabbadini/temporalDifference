"""Off-policy TD control: Q-Learning (S&B §6.5).

Origin: ADAPTED from the course Esercitazione_RL / control.py. Same update rule,
adapted to the Gymnasium API (terminated/truncated), to discretized tuple states and
to returning per-episode statistics.

    Q(S,A) += alpha * [ R + gamma * max_a Q(S',a) - Q(S,A) ]

The target maxes over the next state's actions regardless of what the epsilon-greedy
behaviour policy actually does: this is what makes it off-policy. It learns the value
of the greedy policy while behaving exploratorily.
"""

import numpy as np
from collections import defaultdict

from rl_lab.policies import epsilon_greedy_action, linear_epsilon


def q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1,
               max_steps=1000, epsilon_end=None):
    """Train Q-Learning on a discrete (or discretized) Gymnasium env.

    Args:
        env: Gymnasium env whose observations are hashable (tuples).
        episodes: number of training episodes.
        alpha: learning rate.
        gamma: discount factor.
        epsilon: exploration rate (the START value if epsilon_end is set).
        max_steps: safety cap on episode length.
        epsilon_end: None (default) keeps epsilon constant; if set, epsilon is annealed
            linearly to it over the first half of training, then held.

    Returns:
        (Q, episode_returns, episode_lengths).
    """
    n_actions = env.action_space.n
    # Unseen states default to Q=0 for every action: no initial bias toward any action.
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
    episode_returns = []
    episode_lengths = []

    for ep in range(episodes):
        if epsilon_end is None:
            eps = epsilon
        else:
            eps = linear_epsilon(epsilon, epsilon_end, ep, max(1, episodes // 2))

        state, _ = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = epsilon_greedy_action(Q, state, eps, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # (not done) zeroes the bootstrap term: a terminal state has no future.
            td_target = reward + gamma * np.max(Q[next_state]) * (not done)
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state
            if done:
                break

        episode_returns.append(total_reward)
        episode_lengths.append(step + 1)

    return Q, episode_returns, episode_lengths
