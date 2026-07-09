"""TD(0) prediction for V(s).

TD(0) estimates the state-value function V(s) for a FIXED policy π.
Unlike MC, it updates V(s) at EVERY step using a bootstrap target:

    V(S) ← V(S) + α [ R + γ V(S') - V(S) ]

The term [R + γ V(S') - V(S)] is the TD error (δ).

Key properties vs MC:
  - TD(0) is BIASED (bootstraps from its own estimate V(S'))
  - TD(0) has LOWER VARIANCE (uses only one step of actual reward)
  - TD(0) can learn ONLINE (updates every step, no episode end needed)
  - TD(0) converges faster in practice for many problems

Ported from the course Esercitazione_RL / prediction.py.
"""

import numpy as np
from collections import defaultdict


def td0_prediction(env, policy_fn, episodes=5000, alpha=0.1, gamma=1.0,
                   max_steps=1000):
    """TD(0) prediction for V(s).

    Args:
        env: Gymnasium environment (discrete observations).
        policy_fn: callable(state) → action. The fixed policy to evaluate.
        episodes: number of episodes.
        alpha: learning rate.
        gamma: discount factor.
        max_steps: max steps per episode.

    Returns:
        V: estimated state-value function (defaultdict).
        episode_returns: list of total returns per episode (for plotting).
    """
    V = defaultdict(float)
    episode_returns = []

    for ep in range(episodes):
        state, _ = env.reset()
        ep_return = 0.0

        for _ in range(max_steps):
            action = policy_fn(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

            # TD(0) update: bootstrap from V(S')
            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]
            V[state] += alpha * td_error

            state = next_state
            if done:
                break

        episode_returns.append(ep_return)

    return V, episode_returns
