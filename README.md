# TemporalDifference

Reinforcement Learning course project — M.Sc. Computer Engineering, University of Brescia (A.Y. 2025/26).

Implements and compares RL algorithms **from scratch** across three environments:

1. **CartPole** (common) — Q-Learning, SARSA, Monte Carlo prediction, TD(0) — tabular via state discretization.
2. **Acrobot** (group-specific) — same algorithms; analysis of exploration difficulty and discretization limits.
3. **Pong** (Atari) — DQN from scratch with experience replay + target network; ablation study + Double DQN bonus.

Every result is averaged over **3 seeds** with standard-deviation bands. Focus is on **understanding and experimental analysis**, not absolute performance.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# RTX 5060 (Blackwell) — install the CUDA 12.8 PyTorch build:
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Verify GPU
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Running experiments

```bash
# Env 1: CartPole (tabular) — ~15 min
python3 scripts/run_cartpole.py

# Env 2: Acrobot (tabular) — ~30 min
python3 scripts/run_acrobot.py

# Env 3: Pong (DQN, GPU recommended) — several hours
python3 scripts/run_pong.py
```

Figures are saved to `figures/`, raw data to `results/`.

## Project structure

```
TemporalDifference/
├── rl_lab/                       # main package
│   ├── configs.py                # environment configs
│   ├── policies.py               # greedy, ε-greedy
│   ├── envs/
│   │   ├── discretizer.py        # continuous → discrete wrapper
│   │   └── atari_wrappers.py     # Atari preprocessing
│   ├── algorithms/
│   │   ├── tabular/              # Q-Learning, SARSA, MC, TD(0)
│   │   └── deep/                 # DQN, replay buffer, networks
│   ├── experiments/runner.py     # multi-seed experiment runner
│   └── utils/                    # seeding, plotting
├── scripts/                      # experiment entry points
│   ├── run_cartpole.py
│   ├── run_acrobot.py
│   └── run_pong.py
├── figures/                      # generated plots
├── results/                      # raw experiment data
├── report/                       # LaTeX report
└── CLAUDE.md                     # full project context
```

## Full context

See **[CLAUDE.md](CLAUDE.md)** for the complete project memory: assignment details, architecture, roadmap, coding guidelines, and current status.
