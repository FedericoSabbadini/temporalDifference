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
    "n_actions": 2, # left, right
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
    "n_actions": 3, # torque: -1, 0, +1
}

PONG = {
    "env_name": "ALE/Pong-v5",
    # No discretization: the DQN works on raw pixels. Binning a 210x160x3 image into a
    # Q-table is hopeless, and aggregating by pixel intensity would destroy exactly the
    # information the agent needs (ball and paddle positions).
    "n_actions": 6, # NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE (action set ALE di Pong)

    # Nature DQN preprocessing pipeline (Mnih et al. 2015):
    "frame_skip": 4, # Being the game fps=60 we skip 4 frames per action, so the agent sees 15 frames/sec.
    #  This choice is a compromise between speed and information: skipping more would make the agent
    #  faster but it would miss the ball more often, while skipping less would make it slower but it would see the ball more often.
    "screen_size": 84,  # Final observation: (4, 84, 84), uint8. The original Pong screen is 210x160x3, 
    # but the Nature DQN paper resizes it to 84x84 and converts it to grayscale.
    "stack_size": 4,  # The agent sees the last 4 frames, so it can infer the ball's velocity and direction.
     # The original Pong screen is 210x160x3, but the Nature DQN paper resizes it to 84x84 and converts it to grayscale.
    # Without stacking, the agent would only see a single frame, and it would not be able to infer the ball's velocity and direction. 
    # This is because the ball is a small object that moves fast, and it can be in different positions in different frames. 
    # By stacking 4 frames, the agent can see the ball's trajectory and predict where it will be in the next frame.
    "greyscale_obs": True, # The original Pong screen is 210x160x3, but the Nature DQN paper converts it to grayscale. 
    #This is because color does not carry useful information for the agent, and it reduces the input size from 3 channels to 1 channel, 
    # which makes the network faster and easier to train.
    "noop_max": 30,  # Random no-ops on reset (Nature DQN). This is to make the game start at different positions, 
    # so the agent does not overfit to a single starting position. The original Pong screen is 210x160x3, but the Nature DQN paper resizes it to 84x84 and converts it to grayscale.
}

# Assignment: "media su 3 seed diversi".
SEEDS = [42, 123, 456]
