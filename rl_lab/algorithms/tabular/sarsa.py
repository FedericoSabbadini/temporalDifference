"""On-policy TD control: SARSA (S&B §6.4).

Origin: ADAPTED from the course Esercitazione_RL / control.py.
Same update rule and SARSA pattern (choose A' before update).

Update rule:
    Q(S,A) += α [ R + γ Q(S',A') - Q(S,A) ]

Key difference from Q-Learning: the target uses the ACTUAL next action A'
chosen by the ε-greedy policy, not the greedy max. This makes SARSA
ON-POLICY - it evaluates and improves the policy it is currently following.

Consequence: SARSA learns a "safer" policy that accounts for its own
exploration noise. Classic example (S&B §6.5, Fig. 6.5): on CliffWalking,
SARSA avoids the cliff edge (because it knows it will sometimes explore
and fall off), while Q-Learning walks right along it (because the target
assumes greedy behavior, ignoring the exploration risk).

The name SARSA comes from the quintuple used in each update:
(S, A, R, S', A') - State, Action, Reward, next State, next Action.
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
    n_actions = env.action_space.n  # number of discrete actions
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
    episode_returns = []  # total reward per episode (for learning curves)
    episode_lengths = []  # steps per episode (for diagnostics)

    for ep in range(episodes):
        state, _ = env.reset()
        # SARSA needs the first action BEFORE entering the loop - 
        # unlike Q-Learning, which only needs the action inside the loop.
        # This is because SARSA uses the (S, A, R, S', A') quintuple.
        action = epsilon_greedy_action(Q, state, epsilon, n_actions)
        total_reward = 0.0

        for step in range(max_steps):
            # Take action A, observe R and S'
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Choose next action A' BEFORE updating - this is the key SARSA pattern.
            # A' is chosen by the SAME ε-greedy policy the agent is following.
            # The full quintuple (S, A, R, S', A') is now available for the update.
            next_action = epsilon_greedy_action(Q, next_state, epsilon,
                                                n_actions)

            # === SARSA update (S&B §6.4, Eq. 6.7) ===
            # target = R + γ Q(S', A') - uses the ACTUAL next action A' (not max)
            # This makes SARSA ON-POLICY: the target respects the exploration policy.
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state
            # Carry A' forward: it becomes the action A at the next step.
            # This avoids choosing A' twice (once for update, once for action).
            action = next_action
            if done:
                break

        episode_returns.append(total_reward)
        episode_lengths.append(step + 1)

    return Q, episode_returns, episode_lengths
