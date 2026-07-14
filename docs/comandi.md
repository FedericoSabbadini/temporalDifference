# Comandi e workflow - cheat sheet

> Tutti i comandi che servono, dal setup al report, in un posto solo.
> Regola generale: si lancia sempre con **`uv run`** (usa il `.venv` gestito da
> `uv`), dalla **radice del repo**.

---

## Cheat sheet (i comandi che userai davvero)

```bash
# Setup una tantum
uv sync                                        # installa tutte le dipendenze

# Eseguire gli esperimenti (rigenerano results/ e figures/)
uv run python scripts/run_cartpole.py          # Fase 2 - CartPole   (~15 min, CPU)
uv run python scripts/run_acrobot.py           # Fase 3 - Acrobot    (~30 min, CPU)
uv run python scripts/run_pong.py              # Fase 4 - Pong/DQN   (ore, GPU)

# Rigenerare le tabelle del report dai dati salvati
uv run python scripts/make_report_tables.py    # legge results/*.npz - report/tables/*.tex

# Compilare il report
cd report && latexmk -pdf main.tex             # - report/main.pdf

# Verifica GPU (sulla macchina con RTX 5060)
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Workflow del progetto (la pipeline completa)

Ogni figura e ogni tabella del report è **rigenerabile** da zero. Il flusso:

```
 configs.py                (parametri: env, bin, seed)
     │
     ▼
 scripts/run_<env>.py      (orchestratore dell'ambiente)
     │
     ├─ make_env()  ── DiscretizedEnv / make_atari_env      (ambiente)
     ├─ algoritmo   ── q_learning / sarsa / … / train_dqn   (apprendimento)
     └─ run_multi_seed()   (ripete su 3 seed)  ──set_seed()
     │
     ▼
 results/<env>/*.npz        (array grezzi)  +  results/pong/*.pt (checkpoint)
     │
     ├─────────────utils/plotting.py ──figures/<env>/*.png
     │
     └─make_report_tables.py ──report/tables/*.tex
                                        │
                                        ▼
                         report/main.tex ──(latexmk)──report/main.pdf
```

**Principio chiave**: i dati grezzi (`results/`) sono la *fonte di verità*. Le
figure e le tabelle si derivano da lì - non ci sono numeri scritti a mano.

---

## Comandi per fase

### Fase 2 - CartPole
```bash
uv run python scripts/run_cartpole.py
```
Produce: `figures/cartpole/` (control, 6 sensitivity, 2 failure, 4 prediction) e
`results/cartpole/*.npz`. Esegue in ordine: `exp_control_comparison`,
`exp_sensitivity`, `exp_prediction_comparison`, `exp_failure_case`.

### Fase 3 - Acrobot
```bash
uv run python scripts/run_acrobot.py
```
Produce: `figures/acrobot/` (control, bin_resolution, sensitivity_epsilon,
prediction) e `results/acrobot/*.npz`.

### Fase 4 - Pong / DQN
```bash
uv run python scripts/run_pong.py
```
Produce: `figures/pong/` (3 grafici di ablazione) e `results/pong/` (checkpoint
`.pt` per 5 config × 3 seed + `ablation.npz`). **GPU fortemente consigliata**;
il banner iniziale stampa il device in uso.

---

## Comandi utili di sviluppo/verifica

```bash
# Verifica che il package si importi (infrastruttura ok)
uv run python -c "import rl_lab; print('ok')"

# Controllo qualità del codice (import inutili, ecc.)
uvx ruff check rl_lab scripts --select F

# Test rapido di un algoritmo (poche righe, deterministico)
uv run python -c "
import gymnasium as gym
from rl_lab.configs import CARTPOLE
from rl_lab.envs.discretizer import DiscretizedEnv
from rl_lab.algorithms.tabular.q_learning import q_learning
from rl_lab.utils.seeding import set_seed
e=gym.make(CARTPOLE['env_name']); e=DiscretizedEnv(e,CARTPOLE['n_bins'],CARTPOLE['clip_low'],CARTPOLE['clip_high'])
set_seed(42); Q,r,_=q_learning(e,episodes=200,alpha=0.1,gamma=0.99,epsilon=0.1)
print('ultimi return:', [round(x) for x in r[-5:]])
"
```

---

## Tempi e risorse (indicativi)

| Comando | Tempo | Risorsa |
|---|---|---|
| `run_cartpole.py` | ~15 min | CPU |
| `run_acrobot.py` | ~30 min | CPU |
| `run_pong.py` | ore (200k step × 5 config × 3 seed) | GPU (RTX 5060) |
| `make_report_tables.py` | secondi | CPU |
| `latexmk -pdf main.tex` | ~10 s | CPU |

---

## Riproducibilità

- Ogni esperimento usa i seed `{42, 123, 456}` (in `configs.py`).
- `set_seed` fissa numpy + torch (CPU e CUDA) prima di ogni run; l'env è seedato
  con `env.reset(seed=…)`.
- Rilanciando lo stesso script si ottengono gli **stessi** numeri (a parità di
  ambiente/hardware). I risultati Pong dipendono dal non-determinismo GPU e
  possono variare leggermente.
