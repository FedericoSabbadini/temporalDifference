# CLAUDE.md — TemporalDifference

> **Persistent project memory.** A future Claude Code session should be able to start
> from this file alone and continue the project without losing context. Keep it updated
> as decisions are made and steps are completed.

---

## 1. Project Overview

**What this is:** the final project for the *Reinforcement Learning* course (M.Sc. Computer
Engineering, University of Brescia, A.Y. 2025/26). A single-student project.

**Project name:** **TemporalDifference** ✓ — captures the foundational concept unifying TD(0),
SARSA, Q-Learning, and the entire course arc. Folder: `/TemporalDifference/`.

**Goal:** implement RL algorithms **from scratch** and compare them across three environments,
analyzing behaviour, limits, and performance. The graded emphasis is on **understanding and
experimental analysis, NOT on absolute performance** (stated three times in the assignment).

**Final deliverables:**
1. **Code** — organized, documented, executable, reproducible.
2. **Report** — 10–15 pages (in Italian), including a **mandatory failure analysis**.
3. (Implicit) **Oral discussion** — derived from the report.

**Definition of done:** all four tabular algorithms + DQN implemented and validated; every
result averaged over **3 seeds with std bands**; all required experiments produced as figures;
report written with critical + failure analysis.

---

## 2. Assignment Summary (official requirements)

Three environments, each with mandatory analyses.

### Env 1 — CartPole (common to all groups)
- Implement **from scratch**: **Q-Learning, SARSA, Monte Carlo (prediction), TD(0)**.
- **Technical necessity (not spelled out in the PDF but unavoidable):** CartPole has a
  **continuous 4-D state** → tabular methods require **state discretization** (binning).
- Q-Learning & SARSA are **control** (learn the policy); MC & TD(0) are **prediction**
  (estimate `V(s)` of a *fixed* policy). This mirrors the course's `Esercitazione_RL`
  prediction/control split.
- Analyses: learning curves (reward vs episodes); sensitivity to **ε, α, γ**; stability &
  learning speed; **SARSA vs Q-Learning** (empirical + theoretical); **MC vs TD(0)**
  (bias/variance/stability); **≥1 failure case** (instability / non-convergence / suboptimal).

### Env 2 — **Acrobot** (assigned)
- Apply the same implemented algorithms.
- 6-D continuous state: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), dθ1, dθ2]. 3 actions.
- Discretisation with 6 bins/dim → 6^6 = 46656 states (curse of dimensionality).
- Reward: -1 per step until goal (swing tip above line) or 500 steps.
- Key discussion: **exploration difficulty** (sparse reward, complex dynamics) and
  **need for approximation** (coarse bins lose angular precision → suboptimal).

### Env 3 — **Pong** (ALE/Pong-v5) (assigned)
- Implement **DQN from scratch** with **Experience Replay + Target Network** (PyTorch).
- Ablation: DQN **with vs without target network**, **with vs without experience replay**.
- **Not required to reach high performance** — sufficient to show **learning vs random**.
- Nature CNN (84×84×4 grayscale), 6 actions.
- **Double DQN** bonus confirmed.

### Experimental requirements (ALL environments)
- Every result = **mean over 3 different seeds** + **std deviation / bands**.
- Clearly **reproducible**.

### Technical constraints
- **Allowed libraries:** Gymnasium, PyTorch / TensorFlow.
- **FORBIDDEN:** high-level RL libraries (e.g. **Stable-Baselines3**) for **Env 1 and Env 2**.
  Allowed **only** as an *optional baseline* for Atari (bonus). Core RL code must be hand-written.
  → **Consequence:** the student's existing SB3-based `Deep RL` code is NOT reusable for the core.

### Grading (max 25 pts)
| Component | Points |
|---|---|
| Implementation & correctness | 10 |
| Experimental analysis | 9 |
| Deep RL (Atari) | 4 |
| Report | 2 |
| **Bonus** (Double DQN, Dueling, PER, SB3 baseline, advanced analyses) | +3 (cap at 25) |

**Strategic reading:** 19/25 points live in Env 1+2 (tabular + analysis) — exactly the part the
student can already do. Do **not** over-invest in Atari (only 4 pts, highest risk/cost).
Negative/failure results **must be analyzed, not hidden**. Incomplete implementations without
analysis are penalized.

**Deadline:** *(unknown — see Open Questions).*

---

## 3. Student Knowledge Profile

**Who:** Federico Sabbadini, 24, M.Sc. Computer Engineering @ UniBS, Year 5, GPA ≈ 27.97/30.
Self-assessed AI level 6–7/10. Learns best project-based. Prefers atomic, well-structured docs.

**Completed relevant coursework:** Machine Learning (27), Deep Learning (29, used **Keras**),
Neural Network Architectures (30L), Symbolic AI (29). RL course currently in progress.

**Strong / already has working code for:**
- All four required **tabular algorithms** (Q-Learning, SARSA, MC prediction, TD(0)) — plus
  Expected SARSA, Double Q-Learning, n-step, SARSA(λ), TD(λ). Source: course repos
  `prediction.py`, `control.py`, `Tabular RL/algorithms.py`, and the `Esercitazione_RL` notebook.
- **Gymnasium** API (env loop, spaces, seeding) — `Tutorial_Gymnasium`, custom `WorkflowRoutingEnv`.
- **Experiment structuring**: multi-run averaging (`run_bias_experiment`, 100–200 runs),
  hyperparameter sweeps (`compare_epsilons`), moving-average learning curves, matplotlib plots.
- **Failure-analysis material already in hand**: maximization-bias experiment (`MaxBiasGridEnv`),
  SARSA-vs-Q-Learning cliff-edge behaviour. Reuse these as ready-made failure cases.
- Concepts of experience replay & target network (Deep RL README, DQN Nature paper cited).

**Weak / genuinely new (the real learning in this project):**
- **DQN from scratch in PyTorch** — has only ever used SB3's `DQN`. Never wired the RL loop,
  replay buffer, or target-network sync by hand in PyTorch.
- **Discretization of continuous state spaces** — all his prior tabular envs were already
  discrete grid-worlds; binning wrappers are new (but simple).

**Coding style (match it):** `@dataclass StepResult(next_state, reward, done)`; Q-tables as
`defaultdict(lambda: np.zeros(n_actions))`; functions return tuples like
`(Q, episode_returns, episode_lengths)`; `moving_average(x, window)` for curves; explicit
`set_seed(seed)`; heavy inline comments; NumPy + matplotlib. English identifiers/comments in
code; the **report and oral defense are in Italian**.

---

## 4. Chosen Technical Direction (and why)

- **PyTorch** for the deep part (not TF/Keras, despite his Keras background): it is the RL
  standard, all course references (Nature DQN, SB3) are PyTorch-based, and reference material is
  abundant. TF/Keras is allowed but would cost time. **Decision fixed.**
- **Uniform per-dimension binning** for discretization (not tile coding): simplest,
  course-aligned, sufficient for CartPole (4-D) and MountainCar (2-D). For Acrobot (6-D) the
  binning blow-up becomes *itself* an analysis point ("why approximation is needed here").
- **Dimension-agnostic discretizer**: one wrapper, `n_bins` configurable per dimension, so the
  same code serves CartPole / MountainCar / Acrobot. Handles unbounded velocity dims by clipping
  to a configured range before binning.
- **MC/TD(0) prediction target policy:** since these are *prediction* algorithms, they evaluate
  `V(s)` of a **fixed** policy. Chosen approach: evaluate the **greedy policy learned by
  Q-Learning** (fallback: a fixed heuristic or random policy), then compare MC vs TD(0) on
  bias/variance/convergence — same schema as the `random_walk` exercise. **Document this choice
  in the report.**
- **DQN scope kept minimal but correct:** Nature CNN + replay + target net, short training,
  the two required ablations, 3 seeds. Debug the DQN **first on CartPole** (fast MLP version)
  before moving to the expensive Atari CNN.
- **Reuse over rewrite:** port existing tabular code rather than rewriting; wrap it in a clean
  package with a seed-averaging runner on top.

---

## 5. Architecture

```
TemporalDifference/
├── CLAUDE.md                 # this file — persistent context
├── README.md
├── pyproject.toml            # dependencies & metadata (uv)
├── uv.lock                   # pinned dependency versions
├── .gitignore
├── rl_lab/                   # main package
│   ├── configs.py            # env configs (Python dicts, no YAML)
│   ├── policies.py           # greedy / ε-greedy
│   ├── envs/
│   │   ├── discretizer.py    # continuous → discrete binning wrapper
│   │   └── atari_wrappers.py # AtariPreprocessing + FrameStack
│   ├── algorithms/
│   │   ├── tabular/
│   │   │   ├── q_learning.py
│   │   │   ├── sarsa.py
│   │   │   ├── mc_prediction.py
│   │   │   └── td0.py
│   │   └── deep/
│   │       ├── networks.py       # MLP + Nature CNN
│   │       ├── replay_buffer.py
│   │       └── dqn.py            # train loop + ablation flags + Double DQN
│   ├── experiments/
│   │   └── runner.py         # multi-seed runner (3 seeds + std)
│   └── utils/
│       ├── seeding.py
│       └── plotting.py       # learning curves + sensitivity plots
├── scripts/                  # one script per environment
│   ├── run_cartpole.py       # all Env 1 experiments
│   ├── run_acrobot.py        # all Env 2 experiments
│   └── run_pong.py           # DQN ablations + Double DQN bonus
├── notebooks/                # exploration / figure tweaks
├── results/                  # raw .npz data (gitignored)
├── figures/                  # generated plots for report
└── report/                   # LaTeX report (Overleaf)
```

**Data flow:** `scripts/` → call `rl_lab` algorithms via `experiments/runner.py` (3 seeds) →
save raw arrays to `results/` → `utils/plotting.py` renders to `figures/` → figures go into
`report/`. Keeping raw results separate from figures guarantees reproducibility of every plot.

---

## 6. Development Roadmap

Ordered; each step ends in a verifiable outcome. Steps 1–5 earn the bulk of the grade (19 pts).

1. **Scaffolding + GPU setup.** requirements, package skeleton (done), install PyTorch `cu128`,
   verify `torch.cuda.is_available()` + device name = RTX 5060. *Outcome:* reproducible env.
2. **Discretizer + CartPole sanity check.** Dimension-agnostic binning wrapper; random-policy
   rollout works (reuse `Tutorial_Gymnasium`). *Outcome:* discrete CartPole ready.
3. **Port the 4 tabular algorithms to CartPole.** Adapt Q-Learning/SARSA/MC/TD(0) to the
   Gymnasium API + discretized state. *Outcome:* all four learn/evaluate, single seed.
4. **Seed-averaging runner + std bands.** Generalize `run_bias_experiment`. *Outcome:* reusable
   3-seed experiment engine for every env/algorithm.
5. **Full Env 1 analysis (CartPole).** Learning curves, ε/α/γ sweeps, SARSA-vs-QL, MC-vs-TD(0),
   failure case. *Outcome:* ~9 analysis pts + most of the 10 implementation pts covered.
6. **Env 2 (MountainCar/Acrobot).** Same algorithms; narrative on exploration difficulty
   (sparse reward) and discretization limits. *Outcome:* Env 2 complete + approximation discussion.
7. **DQN from scratch (PyTorch).** ReplayBuffer, QNetwork (MLP first for CartPole debug, then
   CNN), target-net sync, ε-decay, ablation flags. Debug on CartPole before Atari.
   *Outcome:* validated DQN.
8. **Env 3 (Atari) + ablations.** Atari preprocessing wrappers, short training, with/without
   target net & replay, 3 seeds. *Outcome:* 4 Atari pts + deep failure analysis.
9. **Report + bonus (Double DQN) + polish.** Assemble figures, write 10–15 pages **in LaTeX**,
   implement Double DQN bonus (~5 lines over base DQN). *Outcome:* final deliverable.
   Report format: **LaTeX (Overleaf recommended)**.

**Current status: Steps 1–8 code written. PyTorch cu128 (2.11.0+cu128, CUDA 12.8) installed and
GPU verified (RTX 5060 detected, `torch.cuda.is_available()==True`). DQN validated end-to-end on
GPU for both the MLP (CartPole) and CNN (Pong) paths. No full experiments run yet.**

---

## 7. Coding Guidelines

- **Reproducibility first.** Every experiment takes an explicit `seed`. `set_seed` seeds
  `numpy`, `torch` (CPU+CUDA), and the env (`env.reset(seed=...)`). All reported results are
  **mean over 3 seeds** with std bands. Save raw arrays to `results/` so any figure can be
  regenerated without re-training.
- **Config-driven.** Hyperparameters live in `rl_lab/configs.py` (Python dicts), not hard-coded in scripts.
- **Separation of concerns.** Algorithms are pure (env in, results out); scripts orchestrate;
  plotting is isolated in `utils/plotting.py`. No plotting inside training loops.
- **Style match.** Follow the student's conventions (§3): dataclasses for step results,
  `defaultdict` Q-tables for tabular, tuple returns, `moving_average`, heavy comments.
- **Documentation.** Docstrings on every public function (Args/Returns), matching the style
  already used in `random_walk`/`rental_car` notebooks. Comments explain *why*, not *what*.
- **Plotting standard.** Learning curves = reward vs episodes, moving-average smoothed, mean line
  + shaded ±1 std band, labeled axes, legend, title. Consistent color per algorithm across figs.
- **Experiment tracking.** Figures saved to `figures/<env>/`. Raw arrays can be saved to
  `results/` if needed. Keep it simple; no MLflow/W&B at this scale.
- **Repo hygiene.** `results/`, `figures/` outputs, and Atari ROMs are gitignored. Keep the
  package importable (`pip install -e .` optional). English in code; Italian in the report.

---

## 8. Libraries

| Library | Purpose | Notes |
|---|---|---|
| `gymnasium` | Environments (CartPole, MountainCar/Acrobot, Atari) | Core, allowed. `gymnasium[atari]` for Env 3. |
| `ale-py` | Atari backend for Gymnasium | Bundles ROMs in recent versions. |
| `torch` | DQN implementation | **Install the `cu128` build** (RTX 5060 = Blackwell, needs CUDA 12.8 / PyTorch ≥ 2.7). |
| `numpy` | Tabular algorithms, arrays, math | The student's default. |
| `matplotlib` | All plots / learning curves | The student's default. |
| *(no pyyaml)* | Configs are Python dicts in `configs.py` | No YAML dependency needed. |
| `tqdm` | Progress bars (already used in `rental_car`) | Long DQN / sweep runs. |
| `pandas` *(optional)* | Aggregating multi-seed results | Only if convenient. |
| `stable-baselines3` *(optional)* | **Atari baseline ONLY** (bonus) | **Never** for Env 1/2 — forbidden by assignment. |

**Package manager: `uv`.** Install everything with `uv sync`. On Linux/Windows the CUDA 12.8
PyTorch wheel is pulled automatically via `pyproject.toml` source config; on macOS it falls
back to the default CPU-only wheel. Run scripts with `uv run python scripts/run_*.py`.

---

## 9. Hardware Constraints

**Machine (confirmed):** Alienware, **RTX 5060 (8 GB VRAM, Blackwell)**, **32 GB system RAM**.
This is the primary dev + experiment machine. Google Colab is a fallback only.

- **VRAM is not the bottleneck.** The Nature-CNN DQN (~1.7M params) uses a few hundred MB; 8 GB
  is abundant. A dedicated GPU beats Colab here (no session timeouts → run the full 3-seed ×
  ablation matrix overnight).
- **System RAM is the real limit — via the replay buffer.** 1M Atari transitions at full
  4-frame stacks ≈ 28 GB (too much for 32 GB). **Mitigation:** cap the replay buffer at
  **100k–250k** transitions, store **single frames as `uint8`**, and stack on sample. That fits
  comfortably and is more than enough to show learning-vs-random (the only Atari requirement).
- **Blackwell driver caveat.** RTX 50xx needs CUDA 12.8+ and PyTorch ≥ 2.7 (`cu128` wheel). With
  older builds the GPU is not detected. Verify at Step 1 with `torch.cuda.get_device_name()`.
- **Tabular work (Env 1 & 2) is CPU-only** and runs anywhere.

---

## 10. Future Conversation Context (read before writing any code)

- Solo project. Report + oral defense in **Italian**; code + docs in **English**.
- **Env 2 = Acrobot, Atari = Pong** (professor-assigned). Configs in `rl_lab/configs.py`.
- **SB3 is forbidden for the hand-written core (Env 1/2).** The student's existing `Deep RL`
  folder uses SB3 — reuse its *structure/README ideas* but NOT its code for the core DQN.
- The student **already has correct tabular implementations** (grid-world versions). The job is
  to **port + generalize** them to Gymnasium continuous envs via the discretizer — not to
  reinvent them. Match his existing style.
- Grade is dominated by Env 1+2 analysis (19 pts). Keep Atari minimal and correct.
- "Failure analysis" is mandatory and worth real points — the maximization-bias and
  SARSA-vs-QL-cliff experiments are ready-made material for it.
- MC & TD(0) are **prediction** (fixed policy), Q-Learning & SARSA are **control**. Don't
  conflate them: the report needs a clear "which policy does prediction evaluate?" answer.
- Reference texts the course uses: **Sutton & Barto** (2nd ed.) and the **DQN Nature paper**
  (Mnih et al. 2015).

---

## 11. Implementation Checklist

Follow sequentially. Check off in this file as you go.

- [x] **Step 1** — `requirements.txt`, `.gitignore`, README, package skeleton. *(PyTorch cu128 installed; GPU verified — RTX 5060, CUDA 12.8. Also added `opencv-python` for AtariPreprocessing.)*
- [x] **Step 2** — `envs/discretizer.py` (dimension-agnostic binning). *(Code written, not tested.)*
- [x] **Step 3** — `algorithms/tabular/{q_learning,sarsa,mc_prediction,td0}.py` + `policies.py`. *(Code written, not tested.)*
- [x] **Step 4** — `experiments/runner.py` + `utils/{seeding,plotting}.py`. *(Code written, not tested.)*
- [ ] **Step 5** — Run `scripts/run_cartpole.py`: learning curves, ε/α/γ sweeps, SARSA-vs-QL, MC-vs-TD(0), failure case → figures.
- [ ] **Step 6** — Run `scripts/run_acrobot.py`: same + bin-resolution experiment → figures.
- [x] **Step 7** — `algorithms/deep/{networks,replay_buffer,dqn}.py` + Double DQN. *(Code written, not tested.)*
- [ ] **Step 8** — Run `scripts/run_pong.py`: DQN ablations (target net, replay) + Double DQN × 3 seeds → figures.
- [ ] **Step 9** — Write `report/` (10–15 pp, Italian, LaTeX): algorithms, setup, results, critical +
      failure analysis, final section. Double DQN bonus.

---

## 12. Open Questions (awaiting professor + student decisions)

1. **Deadline / time budget?** *(unknown — drives pacing, but not blocking)*

### Decided:
- ✅ Project name: **TemporalDifference**
- ✅ Report format: **LaTeX (Overleaf)**
- ✅ Bonus: **Double DQN** (confirmed; implemented in `dqn.py` via `use_double_dqn` flag)
- ✅ Env 2: **Acrobot** (professor-assigned)
- ✅ Atari game: **Pong** (professor-assigned)
