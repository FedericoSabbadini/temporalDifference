"""Monte Carlo prediction of V(s) for a fixed policy (S&B §5.1).

Origin: ADAPTED from the course Esercitazione_RL / prediction.py (random_walk).

MC is PREDICTION, not control: it never changes the policy, it only estimates V^pi.
It averages the actual returns G_t observed after each state, so it is unbiased but
high-variance, and it can only update once the episode has ended.
"""

from collections import defaultdict


def mc_prediction(env, policy_fn, episodes=5000, gamma=1.0,
                  first_visit=True, max_steps=1000, track_state=None):
    """First-visit (or every-visit) MC prediction.

    Args:
        env: Gymnasium env with discrete/discretized observations.
        policy_fn: callable(state) -> action. The FIXED policy being evaluated.
        episodes: number of episodes to generate.
        gamma: discount factor.
        first_visit: True -> count only the first occurrence of a state per episode.
        max_steps: cap on episode length.
        track_state: if given, V(track_state) is recorded after every episode. This is
            the quantity that actually separates MC from TD(0): the episode return
            cannot, since both methods follow the same fixed policy and therefore
            produce identical returns by construction.

    Returns:
        (V, episode_returns, value_history).
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)
    episode_returns = []
    value_history = []

    for ep in range(episodes):
        # MC needs the whole episode before it can update anything.
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

        # Walking backward makes the return recursive: G_t = R_{t+1} + gamma * G_{t+1}.
        G = 0.0
        visited = set()

        for t in reversed(range(len(episode))):
            s, _, r = episode[t]
            G = gamma * G + r

            if first_visit and s in visited:
                continue
            visited.add(s)

            returns_sum[s] += G
            returns_count[s] += 1.0
            V[s] = returns_sum[s] / returns_count[s]

        if track_state is not None:
            value_history.append(V[track_state])

    return V, episode_returns, value_history
