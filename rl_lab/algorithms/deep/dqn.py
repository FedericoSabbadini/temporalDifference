"""DQN training loop, written from scratch in PyTorch (Mnih et al. 2015).

Origin: NEW. This is the most original part of the project: the course only ever
provided an SB3-based DQN, which the assignment forbids for the core.

Two stabilisers, each behind an ablation flag so the assignment's "con vs senza"
comparison is a one-line change:
  - experience replay: random mini-batches instead of the correlated live stream;
  - target network: a frozen copy supplies the TD target, so a gradient step does not
    move prediction and target at the same time.

Double DQN (bonus, van Hasselt et al. 2016) changes only the target:
    DQN:        y = R + gamma * max_a' Q_target(S', a')
    Double DQN: y = R + gamma * Q_target(S', argmax_a' Q_online(S', a'))
Selection (online) is split from evaluation (target), which curbs the maximization
bias of taking a max over noisy estimates (S&B §6.7).

The loop is STEP-based, not episode-based: on Atari the budget that matters is
environment steps, and episodes simply start and end inside it.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from rl_lab.algorithms.deep.replay_buffer import ReplayBuffer
from rl_lab.algorithms.deep.networks import NatureCNN
from rl_lab.policies import linear_epsilon


def _obs_to_tensor(obs, device):
    """Single observation -> (1, 4, 84, 84) float tensor in [0, 1].

    FrameStackObservation already yields channel-first uint8, so this normalizes and
    adds a batch dim; transposing here would silently scramble the frames.
    """
    obs = np.asarray(obs, dtype=np.float32) / 255.0
    return torch.from_numpy(obs).unsqueeze(0).to(device)


def _batch_to_tensor(batch, device):
    """Batch of observations -> (B, 4, 84, 84) float tensor in [0, 1]."""
    arr = np.asarray(batch, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


def _update_network(online_net, target_net, buffer, optimizer, batch_size,
                    gamma, device, use_double_dqn):
    """One gradient step on a mini-batch.

    This is the tabular update with the table replaced by a network: instead of moving
    Q[s][a] toward the TD target in place, we take a gradient step on the squared TD
    error with respect to the weights.
    """
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    states_t = _batch_to_tensor(states, device)
    next_states_t = _batch_to_tensor(next_states, device)
    actions_t = torch.from_numpy(actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    dones_t = torch.from_numpy(dones).to(device)

    # Q(s, a) for the action actually taken.
    q_values = online_net(states_t)
    q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():  # the target is a fixed reference, never differentiated
        if use_double_dqn:
            best_actions = online_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = target_net(next_states_t).gather(1, best_actions).squeeze(1)
        else:
            next_q = target_net(next_states_t).max(dim=1)[0]

        # (1 - done) removes the bootstrap term at terminal states.
        td_target = rewards_t + gamma * next_q * (1.0 - dones_t)

    loss = F.mse_loss(q_value, td_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_dqn(
    env,
    total_steps=200_000,
    gamma=0.99,
    lr=1e-4,
    buffer_capacity=100_000,
    batch_size=32,
    target_update_freq=1000,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=50_000,
    learning_starts=5000,
    use_target_network=True,
    use_replay=True,
    use_double_dqn=False,
    device="cpu",
    log_interval=10,
):
    """Train a DQN agent.

    Args:
        env: Gymnasium environment (already Atari-preprocessed).
        total_steps: environment steps (the budget), not episodes.
        gamma, lr, batch_size: standard.
        buffer_capacity: replay size. I cap it at 100k: 1M Atari transitions would be
            ~28 GB, which does not fit in 32 GB of RAM.
        target_update_freq: steps between hard target syncs.
        epsilon_start/end/decay_steps: linear exploration schedule.
        learning_starts: transitions to collect before the first gradient step, so the
            first batches are not drawn from an almost-empty (correlated) buffer.
        use_target_network: False -> the target IS the online net (ablation).
        use_replay: False -> no buffer, learn from the latest transitions (ablation).
        use_double_dqn: True -> Double DQN target (bonus).

    Returns:
        (episode_returns, episode_lengths, online_net).
    """
    n_actions = env.action_space.n

    online_net = NatureCNN(n_actions).to(device)

    if use_target_network:
        target_net = copy.deepcopy(online_net)
        target_net.eval()
    else:
        # Ablation: same object, so every gradient step also moves the target.
        target_net = online_net

    optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)

    if use_replay:
        buffer = ReplayBuffer(buffer_capacity)
        min_buffer = max(learning_starts, batch_size)
    else:
        # Ablation: a buffer of exactly one batch degenerates to online learning on the
        # most recent, highly correlated transitions.
        buffer = ReplayBuffer(batch_size)
        min_buffer = batch_size

    episode_returns = []
    episode_lengths = []
    step = 0

    while step < total_steps:
        state, _ = env.reset()
        ep_return = 0.0
        ep_steps = 0

        while True:
            # Same shared schedule as the tabular algorithms, but the horizon is in env
            # steps rather than episodes.
            epsilon = linear_epsilon(epsilon_start, epsilon_end, step,
                                     epsilon_decay_steps)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = online_net(_obs_to_tensor(state, device))
                    action = q_vals.argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_steps += 1

            buffer.push(state, action, reward, next_state, float(done))

            if len(buffer) >= min_buffer:
                _update_network(
                    online_net, target_net, buffer, optimizer,
                    batch_size, gamma, device, use_double_dqn,
                )

            if use_target_network and step % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

            state = next_state
            step += 1

            if done or step >= total_steps:
                break

        # Only record episodes that actually ENDED. When the step budget runs out
        # mid-game the loop breaks with done=False, and `ep_return` is the score of a
        # game that was cut short — a partial return that is always far too optimistic
        # (a game stopped after conceding 1 point reads as -1, not -21). Appending it
        # put a spurious upward spike at the right edge of every learning curve,
        # including the configurations that never learned anything.
        if not done:
            break

        episode_returns.append(ep_return)
        episode_lengths.append(ep_steps)

        if log_interval and len(episode_returns) % log_interval == 0:
            recent = episode_returns[-log_interval:]
            print(f"  ep {len(episode_returns):>5d} | "
                  f"step {step:>7d}/{total_steps} | "
                  f"mean return (last {log_interval}): {np.mean(recent):.1f} | "
                  f"eps: {epsilon:.3f}")

    return episode_returns, episode_lengths, online_net
