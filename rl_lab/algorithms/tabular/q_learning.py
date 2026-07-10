"""Off-policy TD control: Q-Learning (S&B §6.5).

Origin: ADAPTED from the course Esercitazione_RL / control.py.
Same update rule, adapted to: (1) Gymnasium API (terminated/truncated),
(2) discretized states (tuple keys), (3) episode_returns tracking.

Update rule (Bellman optimality equation):
    Q(S,A) ← Q(S,A) + α [ R + γ max_a Q(S',a) - Q(S,A) ]

Key property: the target uses max over NEXT-STATE actions (greedy w.r.t. Q),
regardless of the action actually taken by the ε-greedy behavior policy.
This makes Q-Learning OFF-POLICY: it learns the optimal policy π* while
following an exploratory policy (ε-greedy).

The term in brackets [R + γ max Q(S',a) - Q(S,A)] is the TD error (δ).
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
        alpha: learning rate — controls how much each update changes Q.
               Too high → instability; too low → slow convergence.
        gamma: discount factor — how much future rewards matter.
               gamma=1 treats all future rewards equally; gamma<1 is "myopic".
        epsilon: exploration rate for ε-greedy (constant, no decay).
        max_steps: safety cap on episode length.

    Returns:
        Q: learned Q-table (defaultdict: state → np.array of Q-values per action).
        episode_returns: list of total rewards per episode (for learning curves).
        episode_lengths: list of steps per episode.
    """
    n_actions = env.action_space.n  # number of discrete actions (e.g. 2 for CartPole)
    # defaultdict: unseen states automatically get Q=0 for all actions.
    # This means every state starts with equal value → no initial bias toward any action.
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
    episode_returns = []  # total reward per episode (for learning curves)
    episode_lengths = []  # steps per episode (for diagnostics)

    for ep in range(episodes):
        # Gymnasium API: reset returns (observation, info_dict)
        state, _ = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            # Behavior policy: ε-greedy (explore with prob ε)
            action = epsilon_greedy_action(Q, state, epsilon, n_actions)

            # Gymnasium API: step returns (obs, reward, terminated, truncated, info)
            # terminated = reached a terminal state (e.g. pole fell)
            # truncated = hit the time limit (e.g. 500 steps in CartPole-v1)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # episode ends either way
            total_reward += reward

            # === Q-Learning update (S&B §6.5, Eq. 6.8) ===
            # target = R + γ max_a Q(S', a)
            # The target uses max (greedy) over NEXT-STATE actions,
            # regardless of the action the ε-greedy policy actually chose → OFF-POLICY.
            # (not done) multiplier: at terminal states S' has no future → γ term = 0
            td_target = reward + gamma * np.max(Q[next_state]) * (not done)
            # α controls the step size: Q moves toward the target by fraction α
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state  # advance to the next state
            if done:
                break

        episode_returns.append(total_reward)
        episode_lengths.append(step + 1)  # +1 because range is 0-indexed

    return Q, episode_returns, episode_lengths
