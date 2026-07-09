"""Configurations for each environment.

All hyperparameters and discretization settings are defined here
as plain Python dicts — no YAML, no external dependencies.
"""

# --- Discretization configs ---

CARTPOLE = {
    "env_name": "CartPole-v1",
    "n_bins": [10, 10, 10, 10],
    # CartPole obs: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    # Velocities are unbounded → clip to reasonable range
    "clip_low": [-4.8, -3.0, -0.418, -3.0],
    "clip_high": [4.8, 3.0, 0.418, 3.0],
}

ACROBOT = {
    "env_name": "Acrobot-v1",
    "n_bins": [6, 6, 6, 6, 6, 6],
    # Acrobot obs: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), dθ1, dθ2]
    # cos/sin bounded [-1, 1]; velocities bounded per Gymnasium docs
    "clip_low": [-1.0, -1.0, -1.0, -1.0, -12.566, -28.274],
    "clip_high": [1.0, 1.0, 1.0, 1.0, 12.566, 28.274],
}

PONG = {
    "env_name": "ALE/Pong-v5",
}

# --- Common experiment settings ---

SEEDS = [42, 123, 456]
