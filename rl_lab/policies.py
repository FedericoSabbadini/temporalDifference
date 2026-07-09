"""Action-selection policies for tabular RL.

These mirror the style from the course exercises (Esercitazione_RL):
greedy with random tie-breaking, epsilon-greedy, and random.
"""

import numpy as np


def greedy_action(Q, state):
    """Select the action with highest Q-value, breaking ties randomly."""
    qvals = Q[state]
    max_q = np.max(qvals)
    candidates = np.flatnonzero(np.isclose(qvals, max_q))
    return np.random.choice(candidates)


def epsilon_greedy_action(Q, state, epsilon, n_actions):
    """Epsilon-greedy: explore with probability epsilon, exploit otherwise."""
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return greedy_action(Q, state)
