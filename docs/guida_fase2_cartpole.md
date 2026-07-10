# Fase 2 — CartPole (Env 1, comune a tutti i gruppi)

## Obiettivo della fase

Implementare e confrontare i 4 algoritmi tabular RL richiesti dalla consegna
su CartPole-v1, producendo tutte le analisi obbligatorie:

1. **Learning curves** — reward vs episodi per ogni algoritmo
2. **Confronto control** — SARSA vs Q-Learning (differenze empiriche e teoriche)
3. **Confronto prediction** — Monte Carlo vs TD(0) (bias, varianza, stabilità)
4. **Sensitivity** — impatto di ε, α, γ su Q-Learning e SARSA
5. **Failure case** — almeno un caso di instabilità o mancata convergenza

Questa fase copre la parte più pesante del voto: 10 punti (implementazione) +
9 punti (analisi sperimentale) passano in gran parte da qui.

## Collegamento con la teoria

### Q-Learning vs SARSA (Sutton & Barto §6.5)

Entrambi sono algoritmi di **TD control** — imparano una policy ottimale
interagendo con l'ambiente. La differenza è nel **target dell'update**:

| | Q-Learning (off-policy) | SARSA (on-policy) |
|---|---|---|
| **Update target** | R + γ **max**_a Q(S', a) | R + γ Q(S', **A'**) |
| **A' scelto come?** | Non importa (usa il max) | Con ε-greedy (l'azione effettiva) |
| **Cosa impara** | La policy ottimale π* | La policy che sta seguendo (ε-greedy) |
| **Rischio** | Sovrastima Q (maximization bias) | Più conservativo, a volte subottimale |

L'esempio classico è **CliffWalking** (Sutton & Barto Fig. 6.5): Q-Learning cammina
sul bordo del precipizio (perché il target assume comportamento greedy), mentre
SARSA impara un percorso più sicuro (perché sa che esplorerà con probabilità ε).

Su CartPole il divario è meno drammatico perché non ci sono "cliff" equivalenti,
ma la differenza on-policy/off-policy è comunque visibile nella stabilità delle curve.

### Monte Carlo vs TD(0) (Sutton & Barto §6.1–6.2)

Entrambi sono algoritmi di **prediction** — stimano V(s) per una **policy fissa**
(non imparano la policy, la ricevono come input).

| | Monte Carlo | TD(0) |
|---|---|---|
| **Tipo di update** | G (return completo dell'episodio) | R + γ V(S') (bootstrap a 1 step) |
| **Bias** | Nessuno (G è il vero return) | Sì (usa la stima V(S') corrente) |
| **Varianza** | Alta (G dipende dall'intera traiettoria) | Bassa (usa solo 1 step di reward) |
| **Quando aggiorna** | A fine episodio | A ogni step |
| **Convergenza** | Garantita (con visite infinite) | Garantita (con α decrescente) |

Il confronto MC vs TD(0) è lo stesso dell'esercizio **random_walk** del corso,
dove si confrontano le stime V(s) di MC e TD(0) con i valori veri.
Qui lo applichiamo a CartPole: valutiamo la policy appresa da Q-Learning con MC e TD(0).

### Failure cases (perché sono obbligatori)

La consegna dice esplicitamente: "Risultati negativi devono essere analizzati, non
ignorati." Due failure case classici:

1. **α troppo alto** → l'update è troppo aggressivo, i Q-values oscillano
   senza convergere. Corrisponde al concetto di learning rate instability
   (Sutton & Barto §2.4 per il caso bandit, §6.7 per TD).

2. **ε = 0** → nessuna exploration. L'agente segue la policy greedy dal primo
   episodio e non scopre stati migliori. Corrisponde al dilemma
   exploration/exploitation (Sutton & Barto §2.1).

## File coinvolti

| File | Ruolo |
|---|---|
| `rl_lab/algorithms/tabular/q_learning.py` | Implementazione Q-Learning |
| `rl_lab/algorithms/tabular/sarsa.py` | Implementazione SARSA |
| `rl_lab/algorithms/tabular/mc_prediction.py` | MC prediction (first-visit) |
| `rl_lab/algorithms/tabular/td0.py` | TD(0) prediction |
| `scripts/run_cartpole.py` | Orchestratore: lancia tutti gli esperimenti |

## Flusso del codice

```
run_cartpole.py
│
├── exp_control_comparison()
│   Q-Learning × 3 seed  ──┐
│   SARSA × 3 seed  ───────┤→ plot: control_comparison.png
│                           └→ salva: results/cartpole/control_comparison.npz
│
├── exp_sensitivity()
│   Per ogni algoritmo (Q-Learning, SARSA):
│     ε sweep [0.01, 0.05, 0.1, 0.3]  → sensitivity_QLearning_epsilon.png
│     α sweep [0.01, 0.05, 0.1, 0.5]  → sensitivity_QLearning_alpha.png
│     γ sweep [0.9, 0.95, 0.99, 1.0]  → sensitivity_QLearning_gamma.png
│     (stessa cosa per SARSA)
│
├── exp_prediction_comparison()
│   1. Allena Q-Learning → ottiene Q*
│   2. Estrae la greedy policy: π(s) = argmax_a Q(s,a)
│   3. Valuta π con MC prediction × 3 seed
│   4. Valuta π con TD(0) prediction × 3 seed
│   5. Ripete con random policy per confronto
│   → prediction_learned_policy.png, prediction_random_policy.png
│
└── exp_failure_case()
    α alto [0.1, 0.5, 1.0]     → failure_high_alpha.png
    ε nullo [0.0, 0.01, 0.1]   → failure_no_exploration.png
```

## Scelte implementative

- **5000 episodi** per il training tabular: CartPole è relativamente semplice dopo
  la discretizzazione. Con 10 bin per dimensione → 10^4 = 10000 stati possibili,
  ma in pratica solo ~200 vengono visitati. 5000 episodi bastano per convergere.

- **α = 0.1, γ = 0.99, ε = 0.1** come default: valori standard dalla letteratura
  e dalle esercitazioni del corso. Non c'è ottimizzazione degli iperparametri —
  il punto è confrontare gli algoritmi a parità di condizioni.

- **3000 episodi per prediction**: meno del control perché la policy è fissa
  (non c'è apprendimento della policy, solo stima dei valori).

- **Policy da valutare = greedy su Q appresa**: scelta necessaria perché MC e TD(0)
  sono algoritmi di **prediction**, non control. Servono una policy fissa da valutare.
  Si allena prima Q-Learning, poi si usa la sua greedy policy come input.
  Si aggiunge anche la random policy per mostrare il contrasto.

- **Sensitivity solo su Q-Learning e SARSA**: ε, α, γ sono parametri del training
  control. MC prediction non ha ε né α; TD(0) ha α ma non ε (segue una policy
  fissa). La consegna dice "per ogni algoritmo" ma si riferisce al contesto control.

## Cosa è ripreso e cosa è nuovo

| Componente | Origine |
|---|---|
| `q_learning.py` | **Adattato** da `control.py` dell'Esercitazione_RL. Stessa logica, adattato a Gymnasium API (`terminated`, `truncated`, `env.reset()` con seed). |
| `sarsa.py` | **Adattato** da `control.py`. Il pattern SARSA (scegliere A' prima dell'update) è lo stesso del corso. |
| `mc_prediction.py` | **Adattato** da `prediction.py` / `random_walk_incomplete.json`. La struttura (generare episodio → calcolare return → aggiornare V) è la stessa. |
| `td0.py` | **Adattato** da `prediction.py` / `random_walk_incomplete.json`. L'update step-by-step è identico al corso. |
| `run_cartpole.py` | **Nuovo.** Orchestratore che combina tutti gli esperimenti. Il pattern di sweep su iperparametri è ispirato a `compare_epsilons` del corso. |

### Differenze chiave rispetto al codice del corso

1. **Gymnasium API**: il corso usava ambienti custom o vecchia API `gym`. Qui si usa
   `gymnasium` moderno: `obs, info = env.reset(seed=...)`,
   `obs, reward, terminated, truncated, info = env.step(action)`,
   `done = terminated or truncated`.

2. **Stati discretizzati**: il corso usava gridworld (già discreti). Qui c'è il
   wrapper `DiscretizedEnv` che converte osservazioni continue in tuple.

3. **Multi-seed averaging**: il corso faceva singole run o averaging interno.
   Qui ogni esperimento è ripetuto con 3 seed e si plottano media + std.

## Come leggere i risultati

### Learning curves (control_comparison.png)
- **Asse X**: episodio (smoothed con moving average, window=20)
- **Asse Y**: return totale dell'episodio (somma dei reward)
- **Linea continua**: media su 3 seed
- **Banda colorata**: ±1 deviazione standard
- **Osservato**: entrambi salgono e si stabilizzano attorno a **~184** (non 500).
  Il tetto è imposto dalla **discretizzazione** a 10 bin, non dagli algoritmi:
  stati fisicamente diversi collassano nello stesso bin. Q-Learning è più stabile
  tra i seed (±8) di SARSA (±18); il livello medio finale è lo stesso.

### Sensitivity plots
- Ogni curva corrisponde a un valore diverso del parametro variato
- **α**: α=0.01 troppo lento (~50), α=0.1 miglior compromesso, α=0.5 veloce ma rumoroso
- **ε**: valori troppo bassi rallentano, valori alti (0.3) aggiungono rumore alla policy
- **γ** (osservato): valori **alti** sono i migliori — γ=1.0 e γ=0.99 raggiungono
  ~200–230, γ=0.9 si ferma a ~140. La temuta instabilità di γ=1.0 non si manifesta
  (episodi finiti). Atteso: in un compito di sopravvivenza conviene valorizzare il
  lungo termine.

### Prediction plots — ⚠️ limite metodologico
- **Osservato**: le curve di MC e TD(0) sono **praticamente sovrapposte**, con
  banda di varianza molto ampia (return medio ≈230 sulla policy appresa).
- **Perché**: il grafico traccia il **return degli episodi** generati sotto la
  policy fissa. Ma MC e TD(0) seguono la **stessa** policy, quindi generano lo
  stesso return *per costruzione*. Il grafico NON mostra la differenza
  bias/varianza tra i due **stimatori**, che vive nelle stime V(s), non nel
  return della policy.
- **Cosa mostrerebbe la differenza**: tracciare l'errore (es. RMSE) delle stime
  V(s) rispetto a un riferimento, oppure la varianza dei target di aggiornamento
  — come nell'esercizio `random_walk` del corso. Questo è il principale punto
  migliorabile del progetto ed è discusso apertamente nel report.

### Failure case plots
- **High α**: con α=1.0 la curva resta bloccata in basso (~60), molto sotto ad
  α=0.1 (~200); α=0.5 impara ma è più rumorosa.
- **No exploration**: con ε=0.0 (e quasi con ε=0.01) la curva resta bassa (~35),
  mentre ε=0.1 raggiunge ~175.

## Limiti e semplificazioni

- **ε costante**: nella pratica si usa ε-decay (alto all'inizio, basso alla fine).
  Qui si usa ε fisso per semplicità e perché il confronto è più pulito.
- **Nessun ε-decay** significa che l'agente continua a esplorare anche dopo aver
  imparato, il che abbassa il return medio nelle learning curves.
- **Discretizzazione a 10 bin**: sufficiente per CartPole ma non ottimale.
  Il trade-off bin resolution è esplorato in Fase 3 (Acrobot).
- **Nessun early stopping**: si allena per un numero fisso di episodi, anche
  se l'agente ha già converguto. Questo è standard per confronti sperimentali.
