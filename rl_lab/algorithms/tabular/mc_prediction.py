"""Monte Carlo prediction for V(s).

MC prediction estimates the state-value function V(s) for a FIXED policy π.
It generates complete episodes, computes the actual return G from each
visited state, and averages over all visits:

    V(s) = mean of all returns G observed starting from state s

First-visit MC: only the FIRST occurrence of s in an episode counts.
Every-visit MC: all occurrences count.

Key properties vs TD(0):
  - MC is UNBIASED (uses the true return, no bootstrapping)
  - MC has HIGHER VARIANCE (depends on entire episode trajectory)
  - MC must wait until episode END to update (no online learning)

Ported from the course Esercitazione_RL / prediction.py.
"""

import numpy as np
from collections import defaultdict


def mc_prediction(env, policy_fn, episodes=5000, gamma=1.0,
                  first_visit=True, max_steps=1000):
    """First-visit (or every-visit) MC prediction for V(s).

    Args:
        env: Gymnasium environment (discrete observations).
        policy_fn: callable(state) → action. The fixed policy to evaluate.
        episodes: number of episodes to generate.
        gamma: discount factor.
        first_visit: if True, use first-visit MC; else every-visit.
        max_steps: max steps per episode.

    Returns:
        V: estimated state-value function (defaultdict).
        episode_returns: list of total returns per episode (for plotting).
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)
    episode_returns = []

    for ep in range(episodes):
        # --- Generate a complete episode following policy_fn ---
        episode = []
        state, _ = env.reset()

        for _ in range(max_steps):
            action = policy_fn(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state
            if done:
                break

        ep_return = sum(r for _, _, r in episode)
        episode_returns.append(ep_return)

        # --- Compute returns backward and update V(s) ---
        G = 0.0
        visited = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            if first_visit and s in visited:
                continue
            visited.add(s)

            returns_sum[s] += G
            returns_count[s] += 1.0
            V[s] = returns_sum[s] / returns_count[s]

    return V, episode_returns
