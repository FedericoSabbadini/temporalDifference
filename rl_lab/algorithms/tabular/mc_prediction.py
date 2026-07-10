"""Monte Carlo prediction for V(s) (S&B §5.1).

Origin: ADAPTED from the course Esercitazione_RL / prediction.py and
random_walk_incomplete.json. Same structure: generate episode → compute
returns backward → incremental average. Adapted to Gymnasium API.

MC prediction estimates the state-value function V(s) for a FIXED policy π.
It is a PREDICTION algorithm (not control): it does not learn or improve
the policy, it only estimates V(s) given a policy to follow.

Method:
  1. Generate a complete episode following π: S0,A0,R1, S1,A1,R1, ..., ST
  2. For each state S visited, compute the actual return:
     G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ...
  3. Average all observed returns for each state:
     V(s) = mean of all G_t where S_t = s

First-visit MC: only the FIRST occurrence of s in an episode counts.
Every-visit MC: all occurrences count. (Both converge to V^π.)

Key properties vs TD(0) (S&B §6.1):
  - MC is UNBIASED: G_t is the true return, no bootstrapping needed
  - MC has HIGHER VARIANCE: G_t depends on the entire trajectory (all
    future randomness compounds)
  - MC must wait until episode END to update (cannot learn online)
  - MC does NOT bootstrap (does not use its own estimates in the target)
"""

import numpy as np
from collections import defaultdict


def mc_prediction(env, policy_fn, episodes=5000, gamma=1.0,
                  first_visit=True, max_steps=1000):
    """First-visit (or every-visit) MC prediction for V(s).

    Args:
        env: Gymnasium environment (discrete observations).
        policy_fn: callable(state) → action. The fixed policy to evaluate.
                   This function is never modified — MC is pure prediction.
        episodes: number of episodes to generate.
        gamma: discount factor (γ=1.0 = undiscounted, standard for episodic tasks).
        first_visit: if True, use first-visit MC; else every-visit.
        max_steps: max steps per episode.

    Returns:
        V: estimated state-value function (defaultdict: state → float).
        episode_returns: list of total returns per episode (for plotting).
    """
    returns_sum = defaultdict(float)    # cumulative sum of returns G per state
    returns_count = defaultdict(float)  # number of times each state was visited
    V = defaultdict(float)  # estimated V(s) = returns_sum[s] / returns_count[s]
    episode_returns = []  # total return per episode (for plotting)

    for ep in range(episodes):
        # --- Step 1: Generate a COMPLETE episode following policy_fn ---
        # MC requires the full episode before any update (unlike TD which updates online)
        episode = []  # list of (state, action, reward) tuples
        state, _ = env.reset()

        for _ in range(max_steps):
            action = policy_fn(state)  # follow the fixed policy (no learning here)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))  # record the transition
            state = next_state
            if done:
                break

        # Total undiscounted return for this episode (used for plotting only)
        ep_return = sum(r for _, _, r in episode)
        episode_returns.append(ep_return)

        # --- Step 2: Compute returns G backward and update V(s) ---
        # Walking backward through the episode lets us compute G incrementally:
        # G_t = R_{t+1} + γ * G_{t+1}  (recursive definition of return)
        G = 0.0  # running return, built from the end of the episode backward
        visited = set()  # tracks which states have been seen (for first-visit MC)

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            # Accumulate return: each step adds its reward + discounts the future
            G = gamma * G + r  # G_t = R_{t+1} + γ * G_{t+1}

            if first_visit and s in visited:
                continue  # first-visit MC: only count the FIRST occurrence of s
            visited.add(s)

            # Incremental mean: V(s) = sum(all G observed at s) / count(visits to s)
            # This converges to the true V^π(s) by the law of large numbers
            returns_sum[s] += G
            returns_count[s] += 1.0
            V[s] = returns_sum[s] / returns_count[s]

    return V, episode_returns
