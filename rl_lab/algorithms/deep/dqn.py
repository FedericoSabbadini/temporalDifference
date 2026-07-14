"""DQN training loop - from scratch in PyTorch (Mnih et al. 2015).

Origin: NEW. This is the most complex and entirely original part of the project.
The training loop is written from scratch following the Nature DQN paper.

Implements Deep Q-Network with:
  1. Experience Replay - random mini-batch sampling (breaks temporal correlations)
  2. Target Network - separate, slowly-updated copy of Q-net (stabilizes targets)

Ablation flags allow disabling each component independently (required by
the assignment: "confrontare DQN con vs senza target network / replay").

Double DQN (bonus, van Hasselt et al. 2016):
  Standard DQN: target = R + γ max_a' Q_target(S', a')
  Double DQN:   target = R + γ Q_target(S', argmax_a' Q_online(S', a'))
  This decouples action SELECTION (online net) from action EVALUATION (target net),
  reducing the maximization bias of standard Q-Learning (S&B §6.7).

The training loop is STEP-BASED (not episode-based), which is standard for
Atari: a fixed number of environment steps, with episodes starting and
ending naturally within that budget.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from rl_lab.algorithms.deep.replay_buffer import ReplayBuffer
from rl_lab.algorithms.deep.networks import NatureCNN


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _obs_to_tensor(obs, device):
    """Convert a single Atari observation to a batched float tensor.

    Atari frames arrive already channel-first (4, 84, 84) as uint8 from
    FrameStackObservation, so we only normalise to [0, 1] and add a batch
    dimension (no transpose needed): (4, 84, 84) becomes (1, 4, 84, 84).
    """
    obs = np.asarray(obs, dtype=np.float32) / 255.0
    return torch.from_numpy(obs).unsqueeze(0).to(device)


def _batch_to_tensor(batch, device):
    """Convert a batch of Atari observations (B, 4, 84, 84) uint8 to a float
    tensor normalised to [0, 1]. Frames are already channel-first: no transpose.
    """
    arr = np.asarray(batch, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device)  # move to CPU or GPU


# ---------------------------------------------------------------------------
# Core update step
# ---------------------------------------------------------------------------

def _update_network(online_net, target_net, buffer, optimizer, batch_size,
                    gamma, device, use_double_dqn):
    """One gradient step on a mini-batch from the replay buffer.

    This is the neural-network equivalent of the tabular Q-Learning update:
      Tabular:  Q[s][a] += α * (target - Q[s][a]) -> update is made in-place
      DQN:      Q_wtarget[s][a] = neural net output -> update is made via gradient descent on the loss, no in-place modification of Q-values.
                     w are the network weights (parameters) updated by gradient descent: w = w - α * ∇_w loss
                     (if w = w_online update every step, if w = w_target update every N steps)
                     ∇_w = (R + γ max_a' Q_wtarget(S', a') - Q_wonline(S, A)) * ∇_w Q_wonline(S, A)


    Args:
        online_net: the Q-network being trained (θ)
        target_net: the target Q-network (θ')
        buffer: the replay buffer
        optimizer: the optimizer for updating the online network
        batch_size: the size of the mini-batch
        gamma: the discount factor
        device: the device to run the computation on (CPU or GPU)
        use_double_dqn: whether to use Double DQN
    """
    # Sample a random mini-batch from the replay buffer
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    # Convert numpy arrays to PyTorch tensors on the correct device (CPU/GPU)
    states_t = _batch_to_tensor(states, device)
    next_states_t = _batch_to_tensor(next_states, device)
    actions_t = torch.from_numpy(actions).to(device)   # (batch,) - action indices
    rewards_t = torch.from_numpy(rewards).to(device)   # (batch,) - scalar rewards
    dones_t = torch.from_numpy(dones).to(device)       # (batch,) - 0.0 or 1.0

    # --- Compute current Q(s, a) ---
    # Forward pass gives Q-values for ALL actions: shape (batch, n_actions)
    q_values = online_net(states_t)
    # gather picks the Q-value of the action that was actually taken
    # unsqueeze/squeeze: gather needs 2D index - result back to 1D
    q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (batch,)

    # --- Compute target: r + γ * max_a' Q_target(s', a') ---
    with torch.no_grad():  # no gradients through the target (it's a fixed reference)
        if use_double_dqn:
            # Double DQN (van Hasselt 2016): decouples selection from evaluation
            # Step 1: ONLINE net selects the best action in S'
            best_actions = online_net(next_states_t).argmax(dim=1, keepdim=True)
            # Step 2: TARGET net evaluates Q at that action
            q_values_next = target_net(next_states_t)
            # This prevents the overestimation caused by using max on noisy Q-values
            next_q = q_values_next.gather(1, best_actions).squeeze(1) #
        else:
            # Standard DQN: target net does both selection AND evaluation
            # max(dim=1)[0] returns the maximum Q-value across actions (the [0] gets values, [1] would get indices)
            next_q = target_net(next_states_t).max(dim=1)[0]

        # (1 - done) zeros out future value at terminal states (no future after done)
        td_target = rewards_t + gamma * next_q * (1.0 - dones_t)

    # --- Gradient descent step ---
    # MSE loss: (Q_predicted - Q_target)² - same idea as the tabular TD error²
    loss = F.mse_loss(q_value, td_target)

    optimizer.zero_grad()  # clear old gradients from previous step
    loss.backward()        # compute gradients via backpropagation
    optimizer.step()       # update network weights (Adam optimizer)

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
        total_steps: total environment steps (not episodes). Standard for Atari.
        gamma: discount factor.
        lr: learning rate for Adam optimizer.
        buffer_capacity: replay buffer size (ignored if use_replay=False).
        batch_size: mini-batch size for gradient updates.
        target_update_freq: steps between target-net hard updates (θ⁻ - θ).
        epsilon_start/end/decay_steps: linear ε-decay schedule.
            Starts at epsilon_start (full exploration), decays linearly to
            epsilon_end over decay_steps, then stays at epsilon_end.
        learning_starts: fill buffer with this many transitions before updating. Necessary to i.i.d. sampling from the buffer.
        use_target_network: if False, target net = online net (ablation).
        use_replay: if False, update on each transition (no buffer).
        use_double_dqn: if True, use Double DQN action selection (bonus).
        device: "cpu" or "cuda".
        log_interval: print progress every N episodes.

    Returns:
        episode_returns: list of per-episode total rewards.
        episode_lengths: list of per-episode lengths.
        online_net: trained Q-network (can be saved with torch.save).
    """
    n_actions = env.action_space.n  # 6 for Pong

    # --- Build network ---
    # The Nature CNN architecture, for stacked-frame image observations (Atari).
    online_net = NatureCNN(n_actions).to(device)

    if use_target_network:
        # Deep copy creates a SEPARATE network with the same initial weights.
        # The target net's weights are frozen and only updated periodically (hard update).
        # This prevents the "moving target" problem: without it, every gradient step
        # changes both the prediction AND the target simultaneously - instability.
        target_net = copy.deepcopy(online_net)
        target_net.eval()  # eval mode: disables dropout/batchnorm (not used here, but good practice)
    else:
        # Ablation: target net IS the online net (same Python object - same weights)
        # Every gradient step changes the target too - the "moving target" problem
        target_net = online_net # pointer to the same object, not a copy

    # Adam optimizer: adaptive learning rate per parameter (better than plain SGD for RL)
    optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)

    # --- Replay buffer ---
    if use_replay:
        buffer = ReplayBuffer(buffer_capacity)
        # Don't start learning until buffer has enough transitions for diverse batches
        min_buffer = max(learning_starts, batch_size)
    else:
        # Without replay: buffer holds only batch_size transitions (essentially online learning).
        # Each update uses the most recent transitions - correlated data, less stable.
        buffer = ReplayBuffer(batch_size)
        min_buffer = batch_size  # start learning as soon as one batch is available/stored

    # --- Training loop (STEP-BASED, standard for Atari) ---
    # Unlike tabular algorithms which loop over episodes, DQN counts total env steps.
    # Episodes start and end naturally within the step budget.
    episode_returns = []  # per-episode total reward (for learning curves)
    episode_lengths = []  # per-episode step count
    step = 0  # global step counter across all episodes

    while step < total_steps:
        state, _ = env.reset()  # start a new episode
        ep_return = 0.0  # accumulate reward for this episode
        ep_steps = 0

        while True:
            # Linear ε-decay: ε decreases linearly from epsilon_start to epsilon_end
            # over epsilon_decay_steps, then stays at epsilon_end forever.
            # Early training - high ε - mostly random (exploration)
            # Late training - low ε - mostly greedy (exploitation)
            epsilon = max(
                epsilon_end,
                epsilon_start - step * (epsilon_start - epsilon_end)
                / epsilon_decay_steps,
            )

            # ε-greedy action selection using the ONLINE network
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # random action (exploration)
            else:
                # Greedy action: forward pass through online net, pick argmax
                with torch.no_grad():  # no gradient needed for action selection
                    q_vals = online_net(_obs_to_tensor(state, device))
                    action = q_vals.argmax(dim=1).item()  # .item() extracts scalar from tensor

            # Execute action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_steps += 1

            # Store transition in replay buffer (even before learning starts)
            buffer.push(state, action, reward, next_state, float(done))

            # Gradient update (only after buffer has enough transitions for a diverse batch)
            # This if is not executed only in the first few steps of training, until the buffer is filled enough.
            if len(buffer) >= min_buffer:
                _update_network(
                    online_net, target_net, buffer, optimizer,
                    batch_size, gamma, device, use_double_dqn,
                )

            # Periodically sync target network: θ⁻ - θ (hard update)
            # This copies ALL weights from online net to target net at once.
            # Between syncs, the target net is frozen - stable TD targets.
            if use_target_network and step % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

            state = next_state  # advance to the next state
            step += 1  # count global steps (across episodes)

            if done or step >= total_steps:
                break  # episode ended naturally or step budget exhausted

        episode_returns.append(ep_return)
        episode_lengths.append(ep_steps)

        # Progress logging: print mean return of recent episodes
        if log_interval and len(episode_returns) % log_interval == 0:
            recent = episode_returns[-log_interval:]
            print(f"  ep {len(episode_returns):>5d} | "
                  f"step {step:>7d}/{total_steps} | "
                  f"mean return (last {log_interval}): {np.mean(recent):.1f} | "
                  f"ε: {epsilon:.3f}")

    return episode_returns, episode_lengths, online_net
