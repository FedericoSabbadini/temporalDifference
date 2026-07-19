import numpy as np


def greedy_action(Q, state):
    """Pick argmax_a Q(s,a), breaking ties uniformly at random.

    Random tie-breaking matters: the Q-table starts at 0 everywhere, so without it the
    agent would deterministically pick action 0 in every state it has not seen yet.
    """
    qvals = Q[state]
    candidates = np.flatnonzero(np.isclose(qvals, np.max(qvals))) 
    # returns the indices of the maximum values in qvals, breaking ties randomly
    return np.random.choice(candidates)


def epsilon_greedy_action(Q, state, epsilon, n_actions):
    """Random action with probability epsilon, greedy action otherwise."""
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return greedy_action(Q, state)


def linear_epsilon(start, end, current, horizon):
    """Linear decay from `start` to `end` over `horizon`, then held at `end`.

    One shared schedule for both the tabular loops and the DQN; only the unit of
    (current, horizon) differs, because those loops are episode-based vs step-based.

    In pong current is the number of steps taken so far, and horizon is a fixed epsilon decay duration.
    In cartpole current is the number of episodes so far, and horizon is the total number of episodes to be taken in the whole training run //2.
    """
    frac = min(1.0, current / horizon)
    return start + (end - start) * frac
