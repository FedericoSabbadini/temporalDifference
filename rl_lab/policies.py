"""Action-selection policies for tabular RL.

Origin: ADAPTED from the course exercises (Esercitazione_RL / control.py).
Same ε-greedy logic, adapted for defaultdict Q-tables with np.flatnonzero
for random tie-breaking among maximum Q-values.

Theory: these implement the exploration/exploitation trade-off (S&B §2.1).
- greedy_action: pure exploitation — always picks argmax Q(s,a)
- epsilon_greedy_action: with prob ε explores (random), otherwise exploits (greedy)
"""

import numpy as np


def greedy_action(Q, state):
    """Select the action with highest Q-value, breaking ties randomly.

    Q is a defaultdict mapping states to arrays of Q-values, one per action. 
    (e.g. 
        Q = {
            (0, 0): np.array([0.5, 0.3]), 
            (0, 1): np.array([0.2, 0.4])
            }
    )
    state is the current state (e.g. a tuple of ints for discrete states).
    (e.g.
        state = (0, 0)
    )
    This implements the greedy policy: π(s) = argmax_a Q(s,a).
    Random tie-breaking is important when multiple actions have Q=0
    (as in the initial Q-table), otherwise the agent would always pick action 0.
    """
    qvals = Q[state]  # array of Q-values, one per action (e.g. [0.5, 0.3] for 2 actions)
    max_q = np.max(qvals)  # the highest Q-value across all actions
    # np.isclose handles floating-point comparison (e.g. 0.500000001 ≈ 0.5)
    # flatnonzero returns the indices where the condition is True
    candidates = np.flatnonzero(np.isclose(qvals, max_q))
    return np.random.choice(candidates)  # break ties uniformly at random


def epsilon_greedy_action(Q, state, epsilon, n_actions):
    """Epsilon-greedy: explore with probability ε, exploit otherwise.

    With prob ε: choose a uniformly random action (exploration)
    With prob 1-ε: choose the greedy action (exploitation)

    This is the simplest exploration strategy and the one used throughout
    Sutton & Barto. More advanced alternatives (UCB, softmax) exist but
    are not required by the assignment.
    """
    if np.random.rand() < epsilon:  # with probability ε...
        return np.random.randint(n_actions)  # ...pick a random action (exploration)
    return greedy_action(Q, state)  # otherwise pick the best known action (exploitation)
