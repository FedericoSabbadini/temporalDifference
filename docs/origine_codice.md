# Origine del codice

Ogni modulo del progetto è classificato in base alla sua provenienza.

## Legenda

| Simbolo | Significato |
|---|---|
| **Adattato** | Logica ripresa dal materiale del corso, adattata a Gymnasium moderno e alla struttura del progetto |
| **Ispirato** | Concetto o pattern ripreso, ma implementazione sostanzialmente diversa |
| **Nuovo** | Scritto da zero per il progetto, senza equivalente diretto nel materiale del corso |
| **Standard** | Implementazione di un algoritmo/architettura nota dalla letteratura (Sutton & Barto, Mnih et al.) |

---

## Tabella dettagliata

### Package `rl_lab/`

| File | Classificazione | Fonte | Note |
|---|---|---|---|
| `configs.py` | **Nuovo** | — | I range di clipping sono dalla documentazione Gymnasium. Centralizza parametri prima sparsi nei notebook. |
| `policies.py` | **Adattato** | `Esercitazione_RL/control.py` | Stessa logica ε-greedy del corso. Adattato per `defaultdict` Q-table con `np.flatnonzero` per il tie-breaking. |

### Package `rl_lab/envs/`

| File | Classificazione | Fonte | Note |
|---|---|---|---|
| `discretizer.py` | **Nuovo** | — | Il corso usava ambienti già discreti (gridworld). Il wrapper è necessario perché CartPole e Acrobot hanno stati continui. Usa `gym.ObservationWrapper` (standard Gymnasium). |
| `atari_wrappers.py` | **Adattato** | `Tutorial_Gymnasium.json` (prof.) | Il tutorial del professore mostra come creare un env Atari con Gymnasium. I wrapper `AtariPreprocessing` e `FrameStack` sono di Gymnasium. Parametri dal paper Mnih et al. 2015. |

### Package `rl_lab/algorithms/tabular/`

| File | Classificazione | Fonte | Note |
|---|---|---|---|
| `q_learning.py` | **Adattato** | `Esercitazione_RL/control.py`, `Tabular RL/algorithms.py` | Stessa regola di update. Adattato a: (1) Gymnasium API moderna (`terminated`/`truncated`), (2) stati discretizzati (tuple come chiavi), (3) ritorno di `episode_returns` e `episode_lengths`. |
| `sarsa.py` | **Adattato** | `Esercitazione_RL/control.py` | Come Q-Learning ma on-policy. Il pattern "scegli A' prima dell'update" è lo stesso del corso. |
| `mc_prediction.py` | **Adattato** | `Esercitazione_RL/prediction.py`, `random_walk_incomplete.json` | Struttura identica al corso: genera episodio → calcola return backward → media incrementale. Adattato a Gymnasium API. |
| `td0.py` | **Adattato** | `Esercitazione_RL/prediction.py`, `random_walk_incomplete.json` | Update step-by-step identico. Adattato a Gymnasium API e aggiunto tracking del return per plotting. |

### Package `rl_lab/algorithms/deep/`

| File | Classificazione | Fonte | Note |
|---|---|---|---|
| `networks.py` — `NatureCNN` | **Nuovo / Standard** | Mnih et al. 2015, Table 1 | Architettura replicata dal paper. I numeri (32@8×8/4, 64@4×4/2, 64@3×3/1, FC 512) sono esatti. Implementazione in PyTorch è originale. |
| `networks.py` — `MLP` | **Nuovo** | — | Rete minimale per debug DQN su CartPole prima di passare ad Atari. Non ha equivalente nel corso. |
| `replay_buffer.py` | **Nuovo** | Mnih et al. 2015, §Methods | Implementazione minimale con `collections.deque`. Il concetto è dal paper; il codice è scritto da zero. |
| `dqn.py` | **Nuovo** | Mnih et al. 2015 | Training loop scritto interamente da zero. È la parte più complessa e originale del progetto. Include ablation flags e Double DQN. Il corso non forniva codice DQN da zero (solo SB3). |

### Package `rl_lab/experiments/`

| File | Classificazione | Fonte | Note |
|---|---|---|---|
| `runner.py` | **Ispirato** | Pattern `run_bias_experiment` dall'Esercitazione_RL | Il corso faceva 100-200 run su un singolo ambiente. Qui generalizzato per funzionare con qualsiasi algoritmo e salvare returns per plotting. |

### Package `rl_lab/utils/`

| File | Classificazione | Fonte | Note |
|---|---|---|---|
| `seeding.py` | **Nuovo** | — | Pattern standard (seed numpy + torch). Non ha equivalente diretto nel corso. |
| `plotting.py` — `moving_average` | **Adattato** | Funzione `moving_average` usata negli esercizi del corso | Stessa logica, `np.convolve` con finestra uniforme. |
| `plotting.py` — `plot_learning_curves` | **Nuovo** | — | Aggiunge banda std (mean ± std su più seed). Il corso faceva plot singoli senza bande. |
| `plotting.py` — `plot_sensitivity` | **Nuovo** | — | Ispirato a `compare_epsilons` del corso ma generalizzato per qualsiasi parametro. |

### Script `scripts/`

| File | Classificazione | Fonte | Note |
|---|---|---|---|
| `run_cartpole.py` | **Nuovo** | — | Orchestratore degli esperimenti Env 1. Il design degli esperimenti (quali sweep, quali confronti) segue la consegna. |
| `run_acrobot.py` | **Nuovo** | — | Orchestratore Env 2. Include `exp_bin_resolution()` — esperimento originale che non ha equivalente nel corso. |
| `run_pong.py` | **Nuovo** | — | Orchestratore Env 3. Design ablation study e gestione multi-seed con padding per episodi di lunghezza diversa. |

---

## Riepilogo quantitativo

| Categoria | File |
|---|---|
| **Adattato dal corso** | `policies.py`, `q_learning.py`, `sarsa.py`, `mc_prediction.py`, `td0.py`, `atari_wrappers.py`, `moving_average` |
| **Ispirato dal corso** | `runner.py`, `plot_sensitivity` |
| **Nuovo / da zero** | `configs.py`, `discretizer.py`, `networks.py`, `replay_buffer.py`, `dqn.py`, `seeding.py`, `plot_learning_curves`, tutti gli script |
| **Standard (da paper)** | `NatureCNN` (Mnih 2015), Double DQN (van Hasselt 2016) |

Il codice RL "core" (i 4 algoritmi tabular) è adattato dal materiale del corso.
Il DQN e tutta l'infrastruttura sperimentale sono nuovi.
