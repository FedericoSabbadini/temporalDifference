# TemporalDifference

Reinforcement Learning course project — M.Sc. Computer Engineering, University of Brescia (A.Y. 2025/26).

I implement the core RL algorithms **from scratch** (no high-level RL library) and compare them
across three environments, with sensitivity analyses, explicit failure cases and a DQN ablation
study. Following the assignment, the goal is **understanding and experimental analysis**, not
absolute performance.

## Environments and algorithms

| Env | Environment | Algorithms | Main analyses |
|---|---|---|---|
| 1 | **CartPole-v1** (common) | Q-Learning, SARSA, MC prediction, TD(0) | ε/α/γ sensitivity, SARSA vs Q-Learning, MC vs TD(0), failure cases |
| 2 | **Acrobot-v1** (assigned) | same tabular algorithms | exploration difficulty, bin resolution, discretization limits |
| 3 | **ALE/Pong-v5** (assigned) | DQN from scratch (PyTorch) | ablation ±target network, ±experience replay, Double DQN (bonus) |

Every reported result is the mean over **3 seeds** (42, 123, 456) with ±1σ bands.

CartPole and Acrobot have a continuous state, so tabular methods need discretization: I bin each
dimension uniformly (`rl_lab/envs/discretizer.py`). MC and TD(0) are *prediction* algorithms and
therefore need a fixed policy: I evaluate the greedy policy extracted from a Q-Learning run, plus a
random policy as a reference.

## Setup

```bash
uv sync                                     # creates .venv and installs everything

# GPU check (RTX 5060 → needs the CUDA 12.8 PyTorch build, pulled automatically on Linux/Windows)
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Running the experiments

```bash
uv run python scripts/run_cartpole.py       # Env 1 — tabular, CPU
uv run python scripts/run_acrobot.py        # Env 2 — tabular, CPU
uv run python scripts/run_pong.py           # Env 3 — DQN, GPU, several hours
uv run python report/make_report_tables.py  # regenerates report/tables/*.tex from the .npz
```

Figures go to `figures/<env>/`, raw arrays to `results/<env>/*.npz`, DQN checkpoints to
`results/pong/<config>/`. The report tables are generated from those `.npz` files, so no number in
the report is hand-written.

## Results in one paragraph

Both tabular control algorithms learn on CartPole (SARSA 260 ± 38, Q-Learning 224 ± 51 after
10 000 episodes, still improving) but never reach the 500 ceiling: the limit is the discretization
grid, not the algorithm. On Acrobot they plateau around −350 against an optimum of ≈ −100. Two
Acrobot results **contradicted my initial hypotheses** — finer bins turned out *better*, not worse,
and *low* ε beat high ε — and I kept and analyzed them instead of tuning them away. On Pong the
ablation is clear-cut and per-seed: with replay + target network the DQN learns on all 3 seeds
(−13.5), without replay it never learns (flat at random on every seed), and without the target
network learning becomes a lottery — only 1 seed of 3 leaves the random baseline. Replay makes
learning *possible*, the target network makes it *reliable*.

## Project structure

```
TemporalDifference/
├── rl_lab/                    # the library
│   ├── configs.py             # env configs: bins, clip ranges, seeds
│   ├── policies.py            # greedy / ε-greedy / linear ε schedule
│   ├── envs/                  # discretizer (continuous → discrete), Atari wrappers
│   ├── algorithms/tabular/    # q_learning, sarsa, mc_prediction, td0
│   ├── algorithms/deep/       # networks (Nature CNN), replay_buffer, dqn
│   ├── experiments/runner.py  # multi-seed runner (3 seeds → array + std)
│   └── utils/                 # seeding, plotting (curves with ±1σ bands)
├── scripts/                   # one orchestrator per environment
├── report/                    # LaTeX report (Italian) + auto-generated tables
├── docs/                      # code map and code-origin declaration
├── figures/  results/         # generated plots and raw data
```

## Code origin

The four tabular algorithms are adapted from the course exercises; the DQN, the discretizer, the
multi-seed infrastructure and all the experiment scripts are written from scratch. No
Stable-Baselines3 code is used anywhere. Full breakdown:
[`docs/origine_codice.md`](docs/origine_codice.md). Code layout reference:
[`docs/mappa_codice.md`](docs/mappa_codice.md).

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
- Mnih et al. (2015), *Human-level control through deep reinforcement learning*
- van Hasselt et al. (2016), *Deep Reinforcement Learning with Double Q-Learning*
