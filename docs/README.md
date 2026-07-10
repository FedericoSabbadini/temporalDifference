# Documentazione — indice e guida allo studio

Benvenuto nel *learning companion* del progetto **TemporalDifference**.
Queste guide sono pensate per essere lette **con il codice sorgente aperto a
fianco**: ogni guida è un tour commentato dei file corrispondenti.

> Obiettivo: fra un anno, avendo dimenticato i dettagli, dovresti poter
> ricostruire l'intero progetto leggendo queste guide mentre segui il codice.

---

## 🚦 Da dove iniziare (in base al tempo che hai)

| Ho… | Leggi |
|---|---|
| **10 minuti** | [`panoramica_10_minuti.md`](panoramica_10_minuti.md) — il quadro d'insieme |
| **1 ora** | [`walkthrough_1_ora.md`](walkthrough_1_ora.md) — tour guidato guide+codice |
| **poco, prima dell'orale** | [`ripasso_orale.md`](ripasso_orale.md) — Q&A e punti scomodi |
| **una fase intera da studiare** | la `guida_fase*` corrispondente (sotto) |
| **un termine da ritrovare** | [`glossario.md`](glossario.md) |
| **un parametro/variabile da capire** | [`simboli_e_parametri.md`](simboli_e_parametri.md) |

---

## 📚 Le guide

### Guide di fase (il percorso principale)
Ognuna ha in cima un **box "Percorso di lettura"** con prerequisiti, tempo,
file da aprire e guida precedente.

1. [`guida_fase1_introduzione.md`](guida_fase1_introduzione.md) —
   **Infrastruttura**: discretizzazione, policy, runner multi-seed, plotting.
   *Nessun esperimento: prepara il terreno.*
2. [`guida_fase2_cartpole.md`](guida_fase2_cartpole.md) —
   **CartPole**: i 4 algoritmi tabular, sensitivity, MC vs TD, failure cases.
3. [`guida_fase3_acrobot.md`](guida_fase3_acrobot.md) —
   **Acrobot**: gli stessi algoritmi su un problema più duro; curse of
   dimensionality; i risultati che smentiscono le ipotesi.
4. [`guida_fase4_pong.md`](guida_fase4_pong.md) —
   **Pong / DQN**: deep RL da zero, experience replay, target network, ablation,
   Double DQN.

### Documenti di supporto
- [`origine_codice.md`](origine_codice.md) — cosa è adattato dal corso, cosa è
  nuovo, cosa è standard da paper.
- [`glossario.md`](glossario.md) — ogni termine RL → dove vive nel codice.
- [`simboli_e_parametri.md`](simboli_e_parametri.md) — variabili, tensori,
  iperparametri, strutture dati, forme.
- [`panoramica_10_minuti.md`](panoramica_10_minuti.md),
  [`walkthrough_1_ora.md`](walkthrough_1_ora.md),
  [`ripasso_orale.md`](ripasso_orale.md) — i tre "livelli di zoom".

---

## 🗺️ Ordine di studio consigliato

```
panoramica_10_minuti
        │
        ▼
   guida_fase1 ──────────────► (infrastruttura: la base di tutto)
        │
        ▼
   guida_fase2 ──────────────► (i 4 algoritmi; la fase più densa)
        │
        ├──► guida_fase3 ────► (stessi algoritmi, ambiente più duro)
        │
        └──► guida_fase4 ────► (DQN: parte indipendente dai tabular)
        │
        ▼
   ripasso_orale (quando serve consolidare)
```

- **Fase 1 è prerequisito di tutto** (definisce discretizzazione, runner,
  plotting usati ovunque).
- **Fase 2 è prerequisito di Fase 3** (Fase 3 riusa gli stessi algoritmi).
- **Fase 4 (DQN) è quasi indipendente**: puoi leggerla dopo la Fase 1 senza aver
  fatto le 2–3, se ti interessa solo il deep RL. Condivide però il *target di
  Bellman* con la Fase 2.

---

## 🔗 Grafo delle dipendenze guide ↔ codice

Quali file di `rl_lab/` e `scripts/` apre ciascuna guida:

| Guida | File di codice principali |
|---|---|
| Fase 1 | `configs.py`, `envs/discretizer.py`, `policies.py`, `experiments/runner.py`, `utils/{seeding,plotting}.py` |
| Fase 2 | `algorithms/tabular/{q_learning,sarsa,mc_prediction,td0}.py`, `scripts/run_cartpole.py` |
| Fase 3 | (gli stessi tabular della Fase 2) + `configs.py:ACROBOT`, `scripts/run_acrobot.py` |
| Fase 4 | `algorithms/deep/{networks,replay_buffer,dqn}.py`, `envs/atari_wrappers.py`, `scripts/run_pong.py` |
| trasversale | `scripts/make_report_tables.py`, `report/` |

Dipendenze tra moduli di codice (chi importa chi):

```
scripts/run_*.py
   ├── rl_lab/configs.py            (parametri)
   ├── rl_lab/envs/*                (ambiente: discretizer / atari)
   ├── rl_lab/algorithms/*          (l'algoritmo)
   ├── rl_lab/experiments/runner.py (loop multi-seed) ──► rl_lab/utils/seeding.py
   └── rl_lab/utils/plotting.py     (grafici)
```

---

## ✅ Coerenza con il resto del progetto

- I **numeri** citati nelle guide sono i risultati reali (media su 3 seed) e
  coincidono con `report/tables/` e `results/*.npz`.
- La sezione "Come leggere i risultati" di ogni guida di fase riporta i
  risultati **osservati**, inclusi quelli che contraddicono le ipotesi iniziali.
- Codice in **inglese**, guide e report in **italiano** (come da impostazione del
  progetto).

## Come eseguire (promemoria)

```bash
uv sync                                   # dipendenze
uv run python scripts/run_cartpole.py     # Fase 2
uv run python scripts/run_acrobot.py      # Fase 3
uv run python scripts/run_pong.py         # Fase 4 (GPU consigliata)
uv run python scripts/make_report_tables.py   # tabelle del report dai .npz
```
Vedi il [README principale](../README.md) per i dettagli.
