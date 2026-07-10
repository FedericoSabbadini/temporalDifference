# TemporalDifference

Reinforcement Learning course project — M.Sc. Computer Engineering, University of Brescia (A.Y. 2025/26).

Implements and compares RL algorithms **from scratch** across three environments, with full experimental analysis, failure cases, and ablation study.

## Objective

The project evaluates:
- **Understanding** of RL algorithms (theory ↔ code connection)
- **Implementation** correctness (from scratch, no high-level RL libraries)
- **Experimental analysis** (learning curves, sensitivity, comparisons, failures)

Focus is on **understanding and experimental analysis**, not absolute performance.

## Environments and algorithms

| Env | Environment | Algorithms | Key analysis |
|---|---|---|---|
| 1 | **CartPole** (common) | Q-Learning, SARSA, MC prediction, TD(0) | Sensitivity ε/α/γ, SARSA vs Q-Learning, MC vs TD(0), failure cases |
| 2 | **Acrobot** (assigned) | Same tabular algorithms | Exploration difficulty, curse of dimensionality, bin resolution |
| 3 | **Pong** (Atari, assigned) | DQN from scratch (PyTorch) | Ablation: ±target network, ±experience replay, Double DQN bonus |

Every result is averaged over **3 seeds** (42, 123, 456) with standard-deviation bands.

## Project phases

The project is organized into 4 phases. Detailed guides for each phase are in [`docs/`](docs/).

### Phase 1 — Infrastructure ([guide](docs/guida_fase1_introduzione.md))
Base infrastructure: state discretization (`DiscretizedEnv`), ε-greedy policies, multi-seed runner, plotting utilities, seeding. No experiments — prepares the ground for phases 2–4.

### Phase 2 — CartPole ([guide](docs/guida_fase2_cartpole.md))
All 4 tabular algorithms on CartPole-v1 with discretized state (10 bins/dim → 10,000 states). Produces: control comparison (Q-Learning vs SARSA), ε/α/γ sensitivity (6 plots), MC vs TD(0) prediction, failure cases (high α, ε=0).

### Phase 3 — Acrobot ([guide](docs/guida_fase3_acrobot.md))
Same algorithms on Acrobot-v1 (6-D state, 6 bins/dim → 46,656 states). Key experiments: bin resolution analysis (4/6/8/10 bins showing curse of dimensionality), exploration difficulty (ε sensitivity with sparse reward).

### Phase 4 — Pong / DQN ([guide](docs/guida_fase4_pong.md))
DQN from scratch in PyTorch on ALE/Pong-v5. Experience replay + target network + Double DQN (bonus). Ablation study: 5 configurations × 3 seeds. Goal: show learning vs random policy, not high performance.

## Setup

```bash
# Install uv if not already installed
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (creates .venv automatically)
uv sync

# Verify GPU (on the Alienware with RTX 5060)
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

On Linux/Windows with the RTX 5060, `uv sync` automatically pulls the CUDA 12.8 PyTorch build. On macOS it uses the default CPU-only wheel.

## Running experiments

```bash
# Phase 2: CartPole (tabular, CPU) — ~15 min
uv run python scripts/run_cartpole.py

# Phase 3: Acrobot (tabular, CPU) — ~30 min
uv run python scripts/run_acrobot.py

# Phase 4: Pong (DQN, GPU recommended) — several hours
uv run python scripts/run_pong.py
```

Figures are saved to `figures/`, raw data to `results/`.

## Reading the results

- **Learning curves**: reward (y) vs episode (x), smoothed with moving average. Solid line = mean over 3 seeds, shaded band = ±1 std.
- **Sensitivity plots**: one curve per hyperparameter value. Shows how changing ε, α, or γ affects learning speed and final performance.
- **Ablation plots** (Pong): compare DQN with/without each component. The random baseline (≈ -20.4) is the reference — anything above it means the agent learned something.

## Report

The final report (Italian, LaTeX) is in [`report/`](report/) — see [report/README.md](report/README.md) to compile it (`latexmk -pdf main.tex`). Result tables are generated from the saved `.npz` data, not hand-written:

```bash
uv run python scripts/make_report_tables.py   # → report/tables/*.tex
```

Two experimental results deliberately **contradict** the initial hypotheses (finer Acrobot bins are *better*; lower ε is *better* on Acrobot). These are analyzed honestly in the report's critical-analysis section rather than hidden.

## Project structure

```
TemporalDifference/
├── pyproject.toml                # dependencies and project metadata (uv)
├── uv.lock                      # pinned dependency versions
├── rl_lab/                       # main package
│   ├── configs.py                # environment configs (Python dicts)
│   ├── policies.py               # greedy, ε-greedy action selection
│   ├── envs/
│   │   ├── discretizer.py        # continuous → discrete state wrapper
│   │   └── atari_wrappers.py     # Atari preprocessing (84×84, frame stack)
│   ├── algorithms/
│   │   ├── tabular/              # Q-Learning, SARSA, MC prediction, TD(0)
│   │   └── deep/                 # DQN, replay buffer, CNN/MLP networks
│   ├── experiments/runner.py     # multi-seed experiment runner
│   └── utils/                    # seeding, plotting with std bands
├── scripts/                      # one script per environment
│   ├── run_cartpole.py           # all Env 1 experiments
│   ├── run_acrobot.py            # all Env 2 experiments
│   └── run_pong.py               # DQN ablations + Double DQN
├── docs/                         # comprehension guides (per-phase)
│   ├── guida_fase1_introduzione.md
│   ├── guida_fase2_cartpole.md
│   ├── guida_fase3_acrobot.md
│   ├── guida_fase4_pong.md
│   └── origine_codice.md         # code origin tracking
├── figures/                      # generated plots for report
├── results/                      # raw experiment data (.npz)
├── report/                       # LaTeX report (10-15 pages, Italian)
└── CLAUDE.md                     # full project context and memory
```

## Code origin

See [`docs/origine_codice.md`](docs/origine_codice.md) for a detailed breakdown of what comes from course material, what is adapted, and what is new.

**Summary**: the 4 tabular algorithms are adapted from the course exercises (Esercitazione_RL). The DQN, infrastructure, and all experiment scripts are written from scratch. No SB3 code is used anywhere.

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
- Mnih et al. (2015), *Human-level control through deep reinforcement learning* (Nature DQN paper)
- van Hasselt et al. (2016), *Deep Reinforcement Learning with Double Q-Learning* (Double DQN)
