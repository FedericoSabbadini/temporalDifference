"""TD(0) prediction for V(s) (S&B §6.1).

Origin: ADAPTED from the course Esercitazione_RL / prediction.py and
random_walk_incomplete.json. Same step-by-step update. Adapted to Gymnasium
API and added episode return tracking for plotting.

TD(0) estimates the state-value function V(s) for a FIXED policy π.
Like MC, it is a PREDICTION algorithm — does not learn or improve the policy.

Update rule:
    V(S) ← V(S) + α [ R + γ V(S') - V(S) ]
                      ↑___________________↑
                           TD target

The term δ = R + γ V(S') - V(S) is called the TD error.
It measures how "surprising" the transition was: if the next state S' has
a higher value than expected, δ > 0 and we increase V(S).

Key properties vs MC (S&B §6.2):
  - TD(0) is BIASED: the target uses the current estimate V(S'), which
    may be wrong. The bias decreases as V converges.
  - TD(0) has LOWER VARIANCE: only one step of actual reward (R), not
    the entire trajectory. Less randomness → smoother updates.
  - TD(0) can learn ONLINE: updates at every step, no need to wait for
    episode end. This is crucial for continuing (non-episodic) tasks.
  - TD(0) BOOTSTRAPS: it uses its own estimate V(S') in the target.
    This is the defining characteristic of TD methods.

The name "Temporal Difference" comes from the fact that the TD error δ
is the difference between two successive time estimates of the value.
This is also the origin of the project name: TemporalDifference.
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
        alpha: learning rate — controls how much V(s) changes per step.
               Unlike MC (which uses exact averages), TD(0) uses a fixed α
               which means recent experiences weigh more (exponential recency).
        gamma: discount factor.
        max_steps: max steps per episode.

    Returns:
        V: estimated state-value function (defaultdict: state → float).
        episode_returns: list of total returns per episode (for plotting).
    """
    V = defaultdict(float)  # V(s) = 0 for all states initially
    episode_returns = []  # total return per episode (for plotting)

    for ep in range(episodes):
        state, _ = env.reset()
        ep_return = 0.0  # accumulate reward for this episode

        for _ in range(max_steps):
            action = policy_fn(state)  # follow the fixed policy (no learning here)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

            # === TD(0) update (S&B §6.1, Eq. 6.2) ===
            # target = R + γ V(S') — uses the CURRENT estimate V(S') (bootstrapping)
            # At terminal states, there is no S' → the γ V(S') term must be 0
            td_target = reward + gamma * V[next_state] * (not done)
            # TD error δ = target - current estimate
            # δ > 0 means "S was better than we thought" → increase V(S)
            # δ < 0 means "S was worse than we thought" → decrease V(S)
            td_error = td_target - V[state]
            # Update V(S) by a fraction α of the error
            V[state] += alpha * td_error

            state = next_state  # advance to the next state
            if done:
                break

        episode_returns.append(ep_return)

    return V, episode_returns
