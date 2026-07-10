# Riferimento librerie — cosa usiamo e perché

> Per ogni dipendenza: **cos'è**, **perché la usiamo qui**, e le **API concrete**
> effettivamente usate nel progetto (con il file dove cercarle). Non è la
> documentazione completa delle librerie: è la fetta che tocchiamo davvero.

Le dipendenze sono dichiarate in `pyproject.toml` e bloccate in `uv.lock`.
Installazione: `uv sync`.

---

## Gymnasium — gli ambienti

**Cos'è**: la libreria standard per ambienti RL (erede di OpenAI Gym). Fornisce
l'interfaccia MDP: `reset` → stato iniziale, `step(azione)` → transizione.

**Perché qui**: è l'unica libreria di ambienti consentita dalla consegna.
Fornisce CartPole, Acrobot e (con `ale-py`) gli Atari.

**API usate nel progetto**:

| API | A cosa serve | Dove |
|---|---|---|
| `gym.make(name)` | crea l'ambiente | ogni `make_*` negli script |
| `env.reset(seed=…)` → `(obs, info)` | stato iniziale (riproducibile) | tutti gli algoritmi, `runner.py` |
| `env.step(a)` → `(obs, reward, terminated, truncated, info)` | esegue un'azione | tutti gli algoritmi |
| `env.action_space.n` | numero di azioni | `q_learning`, `dqn`, … |
| `env.observation_space.shape` | forma dell'osservazione (sceglie MLP vs CNN) | `dqn.py` |
| `gym.ObservationWrapper` | classe base per trasformare le osservazioni | `envs/discretizer.py` |
| `gym.wrappers.AtariPreprocessing` | grayscale, resize 84×84, frame-skip | `envs/atari_wrappers.py` |
| `gym.wrappers.FrameStackObservation` | impila 4 frame (in Gymnasium 1.x; era `FrameStack`) | `envs/atari_wrappers.py` |
| `gym.register_envs(ale_py)` | registra gli env `ALE/*` | `envs/atari_wrappers.py` |

**Da ricordare (API cambiata rispetto al vecchio `gym`)**: `step` restituisce
**cinque** valori; `done = terminated or truncated`. È la differenza principale
rispetto al codice del corso.

---

## ale-py — il backend Atari

**Cos'è**: l'Arcade Learning Environment, il motore che esegue i giochi Atari.

**Perché qui**: serve a Gymnasium per far girare Pong. Le versioni recenti
includono le ROM.

**API usate**: solo `import ale_py` + `gym.register_envs(ale_py)`
(📄 `atari_wrappers.py`). Senza la registrazione, `gym.make("ALE/Pong-v5")`
fallisce con `NamespaceNotFound`.

---

## NumPy — array, RNG, matematica

**Cos'è**: la libreria numerica di base di Python.

**Perché qui**: è il "linguaggio" dei dati del progetto — Q-values, batch di
osservazioni, array di return multi-seed, salvataggi.

**API usate nel progetto**:

| API | A cosa serve | Dove |
|---|---|---|
| `np.random.seed(s)` | riproducibilità (RNG globale) | `utils/seeding.py` |
| `np.random.rand/randint/choice` | esplorazione ε-greedy, tie-breaking | `policies.py` |
| `np.zeros(n_actions)` | valore di default di una cella Q-table | `q_learning`, `sarsa` |
| `np.max`, `np.flatnonzero`, `np.isclose` | target greedy, tie-breaking | `q_learning`, `policies` |
| `np.clip`, `.astype(int)` | discretizzazione | `envs/discretizer.py` |
| `np.array`, `np.transpose`, `np.full` | batch, padding NaN | `dqn.py`, `run_pong.py` |
| `np.mean/std/nanmean/nanstd` | media ± banda multi-seed | `plotting.py`, `make_report_tables.py` |
| `np.convolve` | media mobile (smoothing) | `plotting.py` (`moving_average`) |
| `np.savez` / `np.load` | salva/carica risultati grezzi `.npz` | script, `make_report_tables.py` |

**Perché `defaultdict(lambda: np.zeros(n))` per la Q-table**: alloca solo gli
stati visitati (spazio sparso) e restituisce Q=0 per gli stati mai visti.

---

## PyTorch — la rete del DQN

**Cos'è**: il framework di deep learning usato per il DQN. Scelto perché è lo
standard RL e tutti i riferimenti (paper Nature, SB3) sono PyTorch.

**Perché qui**: implementa la rete `Q(s,·;θ)`, il gradient descent e la GPU.
Usato **solo** nella Fase 4 (i tabular non lo toccano — per questo l'import è
"lazy" in `algorithms/deep/__init__.py`).

**API usate nel progetto**:

| API | A cosa serve | Dove |
|---|---|---|
| `nn.Module` | classe base delle reti | `networks.py` |
| `nn.Linear`, `nn.Conv2d` | layer FC e convoluzionali | `networks.py` |
| `F.relu`, `F.mse_loss` | attivazione e loss di regressione | `networks.py`, `dqn.py` |
| `torch.optim.Adam` | ottimizzatore | `dqn.py` |
| `torch.no_grad()` | disattiva i gradienti (calcolo del target) | `dqn.py` |
| `tensor.gather(1, idx)` | seleziona Q dell'azione presa | `dqn.py` (`_update_network`) |
| `tensor.max(dim=1)` / `argmax` | target greedy / Double DQN | `dqn.py` |
| `torch.from_numpy(...).to(device)` | numpy → tensore su CPU/GPU | `dqn.py` |
| `copy.deepcopy(net)` | crea la target network | `dqn.py` |
| `net.load_state_dict(...)` | sync `θ⁻ ← θ` | `dqn.py` |
| `torch.save(net.state_dict())` | checkpoint del modello | `run_pong.py` |
| `torch.manual_seed`, `torch.cuda.manual_seed_all` | riproducibilità | `utils/seeding.py` |
| `torch.cuda.is_available()` | scelta device | `run_pong.py` |

**Nota GPU**: la RTX 5060 (Blackwell) richiede la build **CUDA 12.8**; `pyproject.toml`
la seleziona automaticamente su Linux/Windows (su macOS ripiega su CPU).

---

## Matplotlib — i grafici

**Cos'è**: la libreria di plotting di Python.

**Perché qui**: genera tutte le learning curve e i grafici di sensitivity.

**API usate**: `matplotlib.use("Agg")` (backend non interattivo, salva su file
senza GUI), poi `pyplot`: `figure`, `plot`, `fill_between` (la banda ±σ),
`savefig`, `close`. Tutto in 📄 `utils/plotting.py`.

---

## Libreria standard Python

| Modulo | Uso | Dove |
|---|---|---|
| `collections.defaultdict` | Q-table / V-table sparse | tabular |
| `collections.deque(maxlen=…)` | replay buffer FIFO | `replay_buffer.py` |
| `random.sample` | campionamento del mini-batch | `replay_buffer.py` |
| `copy.deepcopy` | target network | `dqn.py` |
| `os`, `sys` | path, creazione cartelle | script |

---

## Dipendenze dichiarate ma **non** usate nel codice

- **`tqdm`** (barre di progresso): è in `pyproject.toml` ma **non è importato da
  nessun file**. Retaggio dei notebook del corso. Può essere rimosso senza alcun
  effetto sugli output. Lasciato per ora per non toccare il lockfile.
- **`stable-baselines3`**: dipendenza **opzionale** (`[baseline]`), consentita
  dalla consegna solo come baseline Atari. **Non usata** nel core (vietata per
  Env 1/2). Non installata da `uv sync` di default.
- **`pandas`**: menzionata come opzionale in passato, **non** tra le dipendenze
  e **non** usata.
