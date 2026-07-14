"""TD(0) prediction of V(s) for a fixed policy (S&B §6.1).

Origin: ADAPTED from the course Esercitazione_RL / prediction.py (random_walk).

    V(S) += alpha * [ R + gamma * V(S') - V(S) ]

The target bootstraps on the current estimate V(S'), so TD(0) is biased while V is
still wrong, but has much lower variance than MC (one step of randomness instead of a
whole trajectory) and can update online. The bracket is the TD error, delta.
"""

from collections import defaultdict


def td0_prediction(env, policy_fn, episodes=5000, alpha=0.1, gamma=1.0,
                   max_steps=1000, track_state=None):
    """TD(0) prediction. Same contract as `mc_prediction`.

    Note the alpha: unlike MC's exact running average, a constant alpha weights recent
    experience exponentially, so TD(0) never fully settles.
    """
    V = defaultdict(float)
    episode_returns = []
    value_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        ep_return = 0.0

        for _ in range(max_steps):
            action = policy_fn(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

            # (not done) zeroes the bootstrap term at terminal states.
            td_target = reward + gamma * V[next_state] * (not done)
            V[state] += alpha * (td_target - V[state])

            state = next_state
            if done:
                break

        episode_returns.append(ep_return)

        if track_state is not None:
            value_history.append(V[track_state])

    return V, episode_returns, value_history
