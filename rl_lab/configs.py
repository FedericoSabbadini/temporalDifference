"""Environment configurations: bins, clip ranges, seeds.

Origin: NEW. Plain Python dicts, no YAML.
"""

# The discretizer clips each observation dimension to [clip_low, clip_high] and then
# bins it. Clipping is what makes unbounded dimensions (the velocities) usable at all.

CARTPOLE = {
    "env_name": "CartPole-v1",
    # Obs: [cart_pos, cart_vel, pole_angle, pole_ang_vel]. 10 bins/dim -> 10^4 states.
    "n_bins": [10, 10, 10, 10],
    #
    # I clip to the TERMINATION bounds, not to the observation-space bounds.
    # CartPole-v1 ends the episode as soon as |cart_pos| > 2.4 or |angle| > 0.2095 rad,
    # so states beyond those thresholds never occur: clipping to the wider observation
    # bounds (±4.8, ±0.418) would waste half of the position and angle bins on states
    # that cannot happen. This roughly doubles the useful resolution where it matters.
    # The velocities have no official bound; I picked ±3.0 because while balancing both
    # stay well inside it, and larger values only occur just before a fall, where being
    # clipped into the edge bin is harmless.
    "clip_low": [-2.4, -3.0, -0.21, -3.0],
    "clip_high": [2.4, 3.0, 0.21, 3.0],
}

ACROBOT = {
    "env_name": "Acrobot-v1",
    # Obs: [cos θ1, sin θ1, cos θ2, sin θ2, dθ1, dθ2]. 6 bins/dim -> 6^6 = 46,656 states.
    "n_bins": [6, 6, 6, 6, 6, 6],
    # cos/sin are naturally in [-1, 1]. For the angular velocities I reuse the official
    # bounds from the Gymnasium Acrobot-v1 source (MAX_VEL_1 = 4π, MAX_VEL_2 = 9π), which
    # the env itself uses to clip the observation: no bin is wasted and none saturates.
    "clip_low": [-1.0, -1.0, -1.0, -1.0, -12.566, -28.274],
    "clip_high": [1.0, 1.0, 1.0, 1.0, 12.566, 28.274],
}

PONG = {
    "env_name": "ALE/Pong-v5",
    # No discretization: the DQN works on raw pixels. Binning a 210x160x3 image into a
    # Q-table is hopeless, and aggregating by pixel intensity would destroy exactly the
    # information the agent needs (ball and paddle positions).
}

# Assignment: "media su 3 seed diversi".
SEEDS = [42, 123, 456]
