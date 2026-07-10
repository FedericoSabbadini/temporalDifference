"""DQN training loop — from scratch in PyTorch.

Implements the Deep Q-Network algorithm (Mnih et al. 2015) with:
  1. Experience Replay — random mini-batch sampling (breaks correlations)
  2. Target Network — separate, slowly-updated copy of Q-net (stabilises targets)

Ablation flags allow disabling each component independently (required by
the assignment) and enabling Double DQN (bonus).

The training loop is step-based (standard for Atari) with episode tracking.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from rl_lab.algorithms.deep.replay_buffer import ReplayBuffer
from rl_lab.algorithms.deep.networks import MLP, NatureCNN


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _obs_to_tensor(obs, device):
    """Convert a single observation to a batched float tensor.

    For images (C,H,W uint8): normalise to [0,1] → (1,C,H,W).
    For vectors (float): just wrap and add batch dim → (1,D).

    Atari frames arrive already channel-first (4,84,84) from
    FrameStackObservation, so no transpose is needed here.
    """
    obs = np.asarray(obs)
    if obs.ndim >= 3:
        # Image observation (C, H, W) uint8 → normalise
        obs = obs.astype(np.float32) / 255.0
    else:
        obs = obs.astype(np.float32)
    return torch.from_numpy(obs).unsqueeze(0).to(device)


def _batch_to_tensor(batch, device):
    """Convert a batch of observations to a float tensor.

    For images (B,C,H,W uint8): normalise → (B,C,H,W).
    For vectors (B,D): just convert.

    Frames are already channel-first, so no transpose is needed.
    """
    arr = np.asarray(batch)
    if arr.ndim >= 4:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr).to(device)


# ---------------------------------------------------------------------------
# Core update
# ---------------------------------------------------------------------------

def _update_network(online_net, target_net, buffer, optimizer, batch_size,
                    gamma, device, use_double_dqn):
    """One gradient step on a mini-batch from the replay buffer."""
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    states_t = _batch_to_tensor(states, device)
    next_states_t = _batch_to_tensor(next_states, device)
    actions_t = torch.from_numpy(actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    dones_t = torch.from_numpy(dones).to(device)

    # Current Q(s, a)
    q_values = online_net(states_t)
    q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # Target: r + γ max_a' Q_target(s', a')  (or Double DQN variant)
    with torch.no_grad():
        if use_double_dqn:
            # Double DQN: select action with ONLINE net, evaluate with TARGET
            best_actions = online_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = target_net(next_states_t).gather(1, best_actions).squeeze(1)
        else:
            next_q = target_net(next_states_t).max(dim=1)[0]

        td_target = rewards_t + gamma * next_q * (1.0 - dones_t)

    loss = F.mse_loss(q_value, td_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

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
        env: Gymnasium environment.
        total_steps: total environment steps to train for.
        gamma: discount factor.
        lr: learning rate for Adam.
        buffer_capacity: replay buffer size (ignored if use_replay=False).
        batch_size: mini-batch size.
        target_update_freq: steps between target-net hard updates.
        epsilon_start/end/decay_steps: linear ε-decay schedule.
        learning_starts: fill buffer this many steps before first update.
        use_target_network: if False, target net = online net (ablation).
        use_replay: if False, update on each transition (batch_size=1).
        use_double_dqn: if True, use Double DQN action selection (bonus).
        device: "cpu" or "cuda".
        log_interval: print progress every N episodes.

    Returns:
        episode_returns: list of per-episode total rewards.
        episode_lengths: list of per-episode lengths.
        online_net: trained Q-network.
    """
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    # --- Build networks ---
    if len(obs_shape) == 1:
        online_net = MLP(obs_shape[0], n_actions).to(device)
    else:
        online_net = NatureCNN(n_actions).to(device)

    if use_target_network:
        target_net = copy.deepcopy(online_net)
        target_net.eval()
    else:
        target_net = online_net  # same object — no separate target

    optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)

    # --- Replay buffer ---
    if use_replay:
        buffer = ReplayBuffer(buffer_capacity)
        min_buffer = max(learning_starts, batch_size)
    else:
        # Without replay: store 1 transition and "sample" it immediately
        buffer = ReplayBuffer(batch_size)
        min_buffer = batch_size

    # --- Training loop (step-based) ---
    episode_returns = []
    episode_lengths = []
    step = 0

    while step < total_steps:
        state, _ = env.reset()
        ep_return = 0.0
        ep_steps = 0

        while True:
            # ε-greedy action selection
            epsilon = max(
                epsilon_end,
                epsilon_start - step * (epsilon_start - epsilon_end)
                / epsilon_decay_steps,
            )

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

            # Gradient update
            if len(buffer) >= min_buffer:
                _update_network(
                    online_net, target_net, buffer, optimizer,
                    batch_size, gamma, device, use_double_dqn,
                )

            # Target network sync
            if use_target_network and step % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

            state = next_state
            step += 1

            if done or step >= total_steps:
                break

        episode_returns.append(ep_return)
        episode_lengths.append(ep_steps)

        if log_interval and len(episode_returns) % log_interval == 0:
            recent = episode_returns[-log_interval:]
            print(f"  ep {len(episode_returns):>5d} | "
                  f"step {step:>7d}/{total_steps} | "
                  f"mean return (last {log_interval}): {np.mean(recent):.1f} | "
                  f"ε: {epsilon:.3f}")

    return episode_returns, episode_lengths, online_net
