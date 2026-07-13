"""Configurations for each environment.

All hyperparameters and discretization settings are defined here
as plain Python dicts — no YAML, no external dependencies.

Origin: NEW. Centralizes parameters that were previously scattered across
notebooks. Clip ranges come from Gymnasium environment documentation.
"""

# --- Discretization configs ---
# Each env needs: name, number of bins per dimension, clipping bounds.
# The discretizer will clip observations to [clip_low, clip_high] then bin.
# x_clipped=min(max(x,clip_low),clip_high) --> without clipping I should manage huge values or infinite values.

CARTPOLE = {
    # CartPole-v1: balance a pole on a cart by moving left/right (2 actions)
    # Reward: +1 every step the pole stays upright; episode ends on failure
    "env_name": "CartPole-v1",
    # Obs (4D continuous): [cart_pos, cart_vel, pole_angle, pole_ang_vel].
    # 10 bins/dim → 10^4 = 10,000 discrete states: a reasonable resolution/size
    # trade-off (more bins grow the state space exponentially).
    "n_bins": [10, 10, 10, 10], # 10 bins for each of 4 dimension
    #
    # IMPORTANT design choice: the clip range should match the range the agent
    # ACTUALLY visits, not the full observation-space bounds. CartPole-v1
    # TERMINATES when |cart_pos| > 2.4 or |pole_angle| > 0.2095 rad (±12°).
    # States beyond those thresholds never occur (the episode is already over),
    # so clipping to the wider observation bounds (±4.8, ±0.418) would waste
    # ~half of the position and angle bins on states that can't happen.
    # We therefore clip to the TERMINATION bounds, roughly doubling the useful
    # resolution on the two most important dimensions.
    #   - Position: ±2.4  (termination threshold)
    #   - Angle:    ±0.21 (≈ ±12°, termination threshold)
    #   - Velocities: technically unbounded → ±3.0 covers the practical range
    "clip_low": [-2.4, -3.0, -0.21, -3.0],
    "clip_high": [2.4, 3.0, 0.21, 3.0],
}

ACROBOT = {
    # Acrobot-v1: swing a two-link robot arm above a threshold (3 actions: +1, 0, -1 torque)
    # Reward: -1 every step → the agent is penalized until it reaches the goal
    "env_name": "Acrobot-v1",
    # Obs (6D continuous): [cos(θ1), sin(θ1), cos(θ2), sin(θ2), dθ1, dθ2]
    # 1 is the angle of the first link relative to the vertical, 2 is the angle of the second link relative to the first link
    # (θ1, θ2 = the two joint angles; dθ = their angular velocities).
    # 6 bins/dim → 6^6 = 46,656 discrete states (curse of dimensionality).
    "n_bins": [6, 6, 6, 6, 6, 6], # 6 bins for each of 6 dimension
    # cos/sin: naturally bounded [-1, 1]
    # Angular velocities: bounded per Gymnasium docs (dθ1 ≤ 4π ≈ 12.566, dθ2 ≤ 9π ≈ 28.274)
    # These max values come from the Gymnasium source, not from observation
    "clip_low": [-1.0, -1.0, -1.0, -1.0, -12.566, -28.274],
    "clip_high": [1.0, 1.0, 1.0, 1.0, 12.566, 28.274],
}

PONG = {
    # ALE/Pong-v5: Atari Pong via the Arcade Learning Environment
    # 6 actions (NOOP, FIRE, UP, DOWN, etc.); reward: +1 (score) / -1 (concede)
    "env_name": "ALE/Pong-v5",
    # No discretization needed — DQN uses the raw pixel observations
    # It makes no sense to discretize a 210x160x3 image into a Q-table, so we leave this empty.
    # We should have a number of states of the order of 210*160*3 = 100,800, which is huge and not feasible for a tabular method.
    # Why can't we aggregate by pixel intensity? Because the agent needs to know the position of the ball and paddle,
    #  which is lost if we aggregate.
}

# --- Common experiment settings ---

# Assignment requires: "media su 3 seed diversi"
SEEDS = [42, 123, 456]
