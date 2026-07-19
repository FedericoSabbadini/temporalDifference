from collections import defaultdict


def td0_prediction(env, policy_fn, episodes=5000, alpha=0.1, gamma=1.0,
                   max_steps=1000, track_state=None):
    """TD(0) prediction. Same contract as `mc_prediction`.

    Both MC and TD(0) use a constant alpha, weighting recent experience
    exponentially — this makes the comparison fair.
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

            td_target = reward + gamma * V[next_state] * (not done)
            V[state] += alpha * (td_target - V[state])

            state = next_state
            if done:
                break

        episode_returns.append(ep_return)

        if track_state is not None:
            value_history.append(V[track_state])

    return V, episode_returns, value_history
