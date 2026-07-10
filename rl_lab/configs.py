"""Configurations for each environment.

All hyperparameters and discretization settings are defined here
as plain Python dicts — no YAML, no external dependencies.

Origin: NEW. Centralizes parameters that were previously scattered across
notebooks. Clip ranges come from Gymnasium environment documentation.
"""

# --- Discretization configs ---
# Each env needs: name, number of bins per dimension, clipping bounds.
# The discretizer will clip observations to [clip_low, clip_high] then bin.

CARTPOLE = {
    # CartPole-v1: balance a pole on a cart by moving left/right (2 actions)
    # Reward: +1 every step the pole stays upright; episode ends on failure
    "env_name": "CartPole-v1",
    # 4D continuous state → 10^4 = 10,000 discrete states (manageable)
    #    why 4 dimensions? CartPole obs: [cart_pos=the position of the cart, cart_vel=the velocity of the cart, pole_angle=the angle of the pole, pole_ang_vel=the angular velocity of the pole]
    #    why 10 bins for each dimension? 10 bins is a reasonable trade-off between resolution and state space size. More bins would increase the state space exponentially, making learning slower and more difficult.
    "n_bins": [10, 10, 10, 10], 
    # CartPole obs: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    # Position: bounded ±4.8 (env terminates outside)
    # Velocity: technically unbounded, but ±3.0 covers practical range
    # Angle: ±0.418 rad (≈ ±24°, env terminates beyond ±12°)
    "clip_low": [-4.8, -3.0, -0.418, -3.0], 
    # why these values? These values are based on the physical constraints of the CartPole environment. 
    # The cart position is limited to ±4.8 units, the cart velocity is limited to ±3.0 units, 
    # the pole angle is limited to ±0.418 radians (approximately ±24 degrees), and the pole angular velocity is limited to ±3.0 units. 
    # These limits ensure that the agent learns within a reasonable range of states and avoids extreme values that could destabilize learning.
    # where do these values come from? These values are derived from the Gymnasium environment's specifications and are commonly used in reinforcement 
    # learning literature for the CartPole problem. They represent the practical limits of the environment's state space, ensuring that the agent operates within a feasible range of observations.
    "clip_high": [4.8, 3.0, 0.418, 3.0],
    # why these values? These values are based on the physical constraints of the CartPole environment.
    # The cart position is limited to ±4.8 units, the cart velocity is limited to ±3.0 units, the pole angle is limited to ±0.418 radians (approximately ±24 degrees), 
    # and the pole angular velocity is limited to ±3.0 units. These limits ensure that the agent learns within a reasonable range of states and avoids extreme values that could destabilize learning.
}

ACROBOT = {
    # Acrobot-v1: swing a two-link robot arm above a threshold (3 actions: +1, 0, -1 torque)
    # Reward: -1 every step → the agent is penalized until it reaches the goal
    "env_name": "Acrobot-v1",
    # 6D continuous state → 6^6 = 46,656 discrete states (curse of dimensionality)
    #   why 6 dimensions? Acrobot obs: [cos(θ1)=cosine of the first joint angle, sin(θ1)=sine of the first joint angle, 
    # cos(θ2)=cosine of the second joint angle, sin(θ2)=sine of the second joint angle, dθ1=angular velocity of the first joint, dθ2=angular velocity of the second joint]
    "n_bins": [6, 6, 6, 6, 6, 6],
    # Acrobot obs: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), dθ1, dθ2]
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
}

# --- Common experiment settings ---

# Assignment requires: "media su 3 seed diversi"
SEEDS = [42, 123, 456]
