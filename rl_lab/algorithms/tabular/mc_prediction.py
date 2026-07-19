from collections import defaultdict


def mc_prediction(env, policy_fn, episodes=5000, alpha=0.1, gamma=1.0,
                  first_visit=True, max_steps=1000, track_state=None):
    """First-visit (or every-visit) MC prediction.

    Args:
        env: Gymnasium env with discrete/discretized observations.
        policy_fn: callable(state) -> action. The FIXED policy being evaluated.
        episodes: number of episodes to generate.
        alpha: constant step size for the incremental update V(s) += alpha*(G - V(s)).
        gamma: discount factor.
        first_visit: True -> count only the first occurrence of a state per episode.
        max_steps: cap on episode length.
        track_state: if given, V(track_state) is recorded after every episode.

    Returns:
        (V, episode_returns, value_history).
    """
    V = defaultdict(float)
    episode_returns = []
    value_history = []

    for ep in range(episodes):
        episode = []
        state, _ = env.reset()

        for _ in range(max_steps):
            action = policy_fn(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if terminated or truncated:
                break

        episode_returns.append(sum(r for _, _, r in episode))

        # Compute returns backward (G_t = R_{t+1} + gamma * G_{t+1}).
        G = 0.0
        returns = [0.0] * len(episode)
        for t in reversed(range(len(episode))):
            _, _, r = episode[t]
            G = gamma * G + r
            returns[t] = G

        # Update forward so the *first* visit is the one that counts.
        visited = set()
        for t in range(len(episode)):
            s = episode[t][0]
            if first_visit and s in visited:
                continue
            visited.add(s)
            V[s] += alpha * (returns[t] - V[s])

        if track_state is not None:
            value_history.append(V[track_state])

    return V, episode_returns, value_history
