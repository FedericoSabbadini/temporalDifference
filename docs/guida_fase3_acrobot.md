# Fase 3 — Acrobot (Env 2, assegnato dal professore)

## Obiettivo della fase

Applicare gli **stessi algoritmi** della Fase 2 a un ambiente più complesso:
Acrobot-v1. L'obiettivo non è solo ripetere gli esperimenti, ma **discutere le
difficoltà specifiche** che emergono:

1. **Spazio degli stati 6-dimensionale** → esplosione combinatoria della Q-table
2. **Reward sparso** (-1 per step) → esplorazione difficile
3. **Necessità di discretizzazione** e i suoi limiti → motivazione per function approximation

Questa fase dimostra concretamente **perché i metodi tabular non scalano** e
perché servono i metodi con approssimazione (DQN, Fase 4).

## Collegamento con la teoria

### Curse of dimensionality (Sutton & Barto §9.1)

Con CartPole: 4 dimensioni × 10 bin = 10^4 = **10.000 stati**. Gestibile.
Con Acrobot: 6 dimensioni × 6 bin = 6^6 = **46.656 stati**. Già problematico.
Se usassimo 10 bin: 10^6 = **1.000.000 stati**. Impraticabile per tabular.

Questo è il cuore della **curse of dimensionality**: lo spazio degli stati cresce
**esponenzialmente** con il numero di dimensioni. È il motivo principale per cui
i metodi tabular non funzionano in ambienti complessi e si passa alla
**value function approximation** (reti neurali, DQN).

### Reward sparso e problema dell'esplorazione

Acrobot dà reward = -1 ad ogni step fino a che il tip non supera la linea
(oppure l'episodio viene troncato a 500 step). Questo significa che:

- Un agente casuale ottiene return ≈ -500 (non raggiunge mai il goal)
- Non c'è **reward shaping** per guidare l'agente nella direzione giusta
- L'agente deve scoprire per caso una sequenza di azioni che porta al goal,
  e poi rinforzare quel comportamento

Questo rende l'esplorazione molto più difficile rispetto a CartPole, dove
il reward è +1 per ogni step di sopravvivenza (feedback immediato e denso).

### Acrobot: la fisica dell'ambiente

Acrobot è un sistema a due link (come un pendolo doppio) con un attuatore
solo nel giunto intermedio. L'obiettivo è oscillare il tip sopra una linea.

Osservazione (6D): `[cos(θ1), sin(θ1), cos(θ2), sin(θ2), dθ1, dθ2]`
- cos/sin degli angoli: bounded in [-1, 1]
- velocità angolari: bounded (dθ1 ∈ [-12.57, 12.57], dθ2 ∈ [-28.27, 28.27])

Azioni: 3 (torque -1, 0, +1 sul giunto intermedio)

La dinamica è complessa: serve dare energia al sistema con oscillazioni
sincronizzate, simile a come ci si dondola su un'altalena.

## File coinvolti

| File | Ruolo |
|---|---|
| `rl_lab/configs.py` → `ACROBOT` | Configurazione: 6 bin/dim, range di clipping |
| `scripts/run_acrobot.py` | Orchestratore: 4 esperimenti specifici per Acrobot |
| + tutti i file tabular della Fase 2 | Stessi algoritmi, diverso ambiente |

## Flusso del codice

```
run_acrobot.py
│
├── exp_control_comparison()
│   Q-Learning × 3 seed (10000 ep)  ──┐
│   SARSA × 3 seed (10000 ep)  ───────┤→ control_comparison.png
│
├── exp_bin_resolution()               ← ESPERIMENTO CHIAVE
│   Per n_bins in [4, 6, 8, 10]:
│     Q-Learning × 3 seed  → bin_resolution.png
│   Mostra la curse of dimensionality in azione
│
├── exp_sensitivity_epsilon()
│   Q-Learning con ε ∈ [0.05, 0.1, 0.2, 0.4]  → sensitivity_epsilon.png
│   Mostra quanto l'esplorazione impatta su Acrobot
│
└── exp_prediction()
    1. Allena Q-Learning → greedy policy
    2. Valuta con MC e TD(0)  → prediction_comparison.png
```

## Scelte implementative

- **10.000 episodi** (vs 5000 su CartPole): lo spazio degli stati è molto più grande,
  servono più episodi per esplorare. Con 6^6 = 46.656 stati possibili, 10.000 episodi
  non sono neanche sufficienti a visitarli tutti.

- **max_steps = 500**: Acrobot-v1 tronca a 500 step. Oltre è inutile.

- **6 bin per dimensione** come default: compromesso tra risoluzione e dimensionalità.
  L'esperimento `exp_bin_resolution` mostra esplicitamente cosa succede con 4, 6, 8, 10 bin.

- **ε = 0.2 nell'esperimento bin_resolution**: più alto del default (0.1) perché
  con più stati serve più esplorazione.

- **Sensitivity solo su ε** (non α e γ): su Acrobot il parametro più critico è
  l'esplorazione (reward sparso). La consegna per Env 2 chiede di discutere le
  "difficoltà di esplorazione", non di replicare tutti gli sweep di Env 1.

## Cosa è ripreso e cosa è nuovo

| Componente | Origine |
|---|---|
| Algoritmi (Q-Learning, SARSA, MC, TD(0)) | **Stessi** della Fase 2, nessuna modifica. Questo è il punto: la generalità degli algoritmi. |
| `ACROBOT` config in `configs.py` | **Nuovo.** I range di clipping sono presi dalla documentazione Gymnasium di Acrobot-v1. |
| `run_acrobot.py` | **Nuovo.** Struttura ispirata a `run_cartpole.py` ma con esperimenti specifici: bin_resolution e focus su esplorazione. |
| `exp_bin_resolution()` | **Nuovo.** Non ha equivalente nel corso. È l'analisi più originale del progetto — mostra la curse of dimensionality in modo concreto. |

## Come leggere i risultati

### control_comparison.png
- **Cosa aspettarsi**: curve che partono da -500 (random) e migliorano lentamente.
  Con 6 bin, il miglior return raggiungibile è circa -100 / -150 (non il -80
  ottimale, perché la discretizzazione perde informazione).
- **Q-Learning vs SARSA**: Q-Learning dovrebbe convergere leggermente più in alto
  (off-policy, cerca la policy ottimale), SARSA potrebbe essere più stabile.
- **Confronto con CartPole**: convergenza molto più lenta e return finale più basso
  (relativamente al massimo). Questo illustra la difficoltà dell'ambiente.

### bin_resolution.png
- **4 bin** (4^6 = 4.096 stati): troppo grossolano, non impara quasi nulla
- **6 bin** (6^6 = 46.656 stati): impara qualcosa ma lentamente
- **8 bin** (8^6 = 262.144 stati): potenzialmente migliore ma servirebbero
  molti più episodi per esplorare tutti gli stati
- **10 bin** (10^6 = 1.000.000 stati): troppi stati, l'agente non ha abbastanza
  episodi per visitarli → performance peggiore dei 6 bin
- **Messaggio**: non basta aumentare la risoluzione. La Q-table non scala.
  Serve function approximation.

### sensitivity_epsilon.png
- **ε basso (0.05)**: esplora troppo poco in uno spazio grande → non trova il goal
- **ε alto (0.4)**: esplora molto ma sfrutta poco → return più rumoroso
- **ε medio (0.1–0.2)**: compromesso migliore
- **Confronto con CartPole**: su CartPole ε=0.1 funziona bene; su Acrobot
  potrebbe servire ε più alto per l'esplorazione iniziale.

### prediction_comparison.png
- Stessa interpretazione di CartPole: MC vs TD(0) sulla policy appresa
- Su Acrobot la varianza di MC è ancora più evidente perché gli episodi
  sono più lunghi (fino a 500 step → return G calcolato su 500 step)

## Limiti e semplificazioni

- **Binning uniforme su tutte le dimensioni**: cos/sin degli angoli e velocità
  angolari hanno scale e importanze diverse. Un binning adattivo per dimensione
  potrebbe dare risultati migliori.

- **10.000 episodi possono non bastare**: con 46.656 stati e reward sparso,
  l'agente potrebbe non aver converguto completamente. Ma il punto non è
  ottenere la performance ottimale — è mostrare i limiti del tabular.

- **Non c'è α-decay**: un α decrescente nel tempo migliorerebbe la convergenza
  finale ma complicherebbe il confronto tra configurazioni.

- **L'ambiente Gymnasium tronca a 500 step**: questo impedisce di vedere
  le prestazioni reali su episodi più lunghi, ma è il default di Acrobot-v1.
