# Fase 2 - CartPole (Env 1, comune a tutti i gruppi)

> ### Percorso di lettura
> - **Prerequisiti**: [Fase 1](guida_fase1_introduzione.md) (discretizzazione, runner, plotting). Teoria: TD control, MC/TD prediction (S&B §5-6).
> - **Guida precedente**: [Fase 1](guida_fase1_introduzione.md). **Successiva**: [Fase 3](guida_fase3_acrobot.md).
> - **Tempo stimato**: ~15 minuti (è la fase più densa).
> - **Apri in parallelo**: `algorithms/tabular/q_learning.py`, `sarsa.py`, `mc_prediction.py`, `td0.py` (corti, ~70 righe: leggili tutti) e `scripts/run_cartpole.py`.
> - **Requisito consegna coperto**: Env 1 - i 4 algoritmi da zero, learning curves, sensitivity (ε/α/γ), SARSA vs Q-Learning, MC vs TD(0), ≥1 failure case. È la fetta più grande del voto (10 pt implementazione + 9 pt analisi).

## Obiettivo della fase

Implementare e confrontare i 4 algoritmi tabular RL richiesti dalla consegna
su CartPole-v1, producendo tutte le analisi obbligatorie:

1. **Learning curves** - reward vs episodi per ogni algoritmo
2. **Confronto control** - SARSA vs Q-Learning (differenze empiriche e teoriche)
3. **Confronto prediction** - Monte Carlo vs TD(0) (bias, varianza, stabilità)
4. **Sensitivity** - impatto di ε, α, γ su Q-Learning e SARSA
5. **Failure case** - almeno un caso di instabilità o mancata convergenza

Questa fase copre la parte più pesante del voto: 10 punti (implementazione) +
9 punti (analisi sperimentale) passano in gran parte da qui.

## Collegamento con la teoria

### Q-Learning vs SARSA (Sutton & Barto §6.5)

Entrambi sono algoritmi di **TD control** - imparano una policy ottimale
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

### Monte Carlo vs TD(0) (Sutton & Barto §6.1-6.2)

Entrambi sono algoritmi di **prediction** - stimano V(s) per una **policy fissa**
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

1. **α troppo alto** - l'update è troppo aggressivo, i Q-values oscillano
   senza convergere. Corrisponde al concetto di learning rate instability
   (Sutton & Barto §2.4 per il caso bandit, §6.7 per TD).

2. **ε = 0** - nessuna exploration. L'agente segue la policy greedy dal primo
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
│   SARSA × 3 seed  ───────┤ - plot: control_comparison.png
│                           └ - salva: results/cartpole/control_comparison.npz
│
├── exp_sensitivity()
│   Per ogni algoritmo (Q-Learning, SARSA):
│     ε sweep [0.01, 0.05, 0.1, 0.3] - sensitivity_QLearning_epsilon.png
│     α sweep [0.01, 0.05, 0.1, 0.5] - sensitivity_QLearning_alpha.png
│     γ sweep [0.9, 0.95, 0.99, 1.0] - sensitivity_QLearning_gamma.png
│     (stessa cosa per SARSA)
│
├── exp_prediction_comparison()
│   1. Allena Q-Learning - ottiene Q*
│   2. Estrae la greedy policy: π(s) = argmax_a Q(s,a)
│   3. Valuta π con MC prediction × 3 seed
│   4. Valuta π con TD(0) prediction × 3 seed
│   5. Ripete con random policy per confronto
│ - prediction_<policy>_return.png (return, identico) +
│     prediction_<policy>_value.png (stima V, il confronto vero)
│
└── exp_failure_case()
    α alto [0.1, 0.5, 1.0] - failure_high_alpha.png
    ε nullo [0.0, 0.01, 0.1] - failure_no_exploration.png
```

## Scelte implementative

- **5000 episodi** per il training tabular. Con 10 bin per dimensione - 10^4 =
  10000 stati possibili. Nota: con i bin ristretti ai limiti di terminazione la risoluzione utile è più fine, quindi si visitano più stati e
  5000 episodi **non bastano più a convergere del tutto**: le curve sono ancora
  in salita a fine training (vedi "Come leggere i risultati").

- **α = 0.1, γ = 0.99, ε = 0.1** come default: valori standard dalla letteratura
  e dalle esercitazioni del corso. Non c'è ottimizzazione degli iperparametri - 
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
| `mc_prediction.py` | **Adattato** da `prediction.py` / `random_walk_incomplete.json`. La struttura (generare episodio - calcolare return - aggiornare V) è la stessa. |
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
- **Osservato**: entrambi salgono ma sono **ancora in salita a
  5000 episodi** (non un plateau). SARSA arriva a **~225**, Q-Learning a **~174**:
  con la griglia più fine SARSA supera Q-Learning (prima erano appaiati a ~184).
  Q-Learning resta meno rumoroso (±39 vs ±61 di SARSA). Questa scelta alza il tetto ma rallenta la convergenza e aumenta la varianza - miglioramento a doppio taglio.

### Sensitivity plots
- Ogni curva corrisponde a un valore diverso del parametro variato
- **α**: α=0.01 troppo lento (~30), α=0.1 e α=0.5 arrivano entrambi a ~180-200
  (α=0.5 più veloce con la griglia fine), α=0.05 intermedio
- **ε**: valori troppo bassi rallentano, valori alti (0.3) aggiungono rumore alla policy
- **γ** (osservato): valori **alti** sono i migliori - γ=1.0 raggiunge ~250 e
  γ=0.99 ~180, γ=0.9/0.95 si fermano a ~130. La temuta instabilità di γ=1.0 non si manifesta
  (episodi finiti). Atteso: in un compito di sopravvivenza conviene valorizzare il
  lungo termine.

### Prediction plots - DUE viste
Ci sono due grafici, con nomi `prediction_<policy>_return.png` e
`prediction_<policy>_value.png`.

- **`_return.png` (return degli episodi)**: le curve di MC e TD(0) sono
  **identiche per costruzione** (≈162 sulla policy appresa), perché entrambi
  seguono la **stessa** policy fissa. Questo grafico da solo NON mostra la
  differenza tra i due stimatori - è una lezione metodologica in sé.
- **`_value.png` (stima V(stato di riferimento))**: QUI si vede la differenza.
  Si traccia V dello stato di reset discretizzato (sempre visitato) episodio per episodio:
  - **MC**: traiettoria **rumorosa** (non distorta, alta varianza)
  - **TD(0)**: traiettoria **liscia** (distorta all'inizio, bassa varianza)
  È il compromesso bias/varianza classico (Sutton & Barto §6.2).
- **Perché due viste**: il return da solo produce curve sovrapposte; la vista
  sulle stime V(s) rende il confronto significativo.
- **Limite residuo**: è una misura *locale* (un solo stato). Una misura globale
  (RMSE di V su tutti gli stati) richiederebbe un V di riferimento non
  disponibile qui.

### Failure case plots
- **High α**: con α=1.0 la curva resta bloccata in basso (~65) - il fallimento.
  Attenzione: qui i valori moderati NON falliscono, anzi α=0.5 (~160) è il
  migliore, sopra ad α=0.1 (~140). Il problema è specifico del valore estremo
  α=1.0, non genericamente dell'"α alto".
- **No exploration**: con ε=0.0 (e quasi con ε=0.01) la curva resta bassa (~40),
  mentre ε=0.1 raggiunge ~140.

## Modello mentale (cosa fa l'agente, episodio per episodio)

Segui un singolo episodio di **Q-Learning** come se fossi l'agente:

- **Cosa so**: una Q-table inizialmente tutta a zero. Per ogni stato (tupla di
  bin) conosco un valore stimato per ciascuna delle 2 azioni.
- **Cosa faccio ad ogni step**: guardo lo stato corrente, con prob. ε tiro un
  dado (azione casuale), altrimenti prendo l'azione con Q più alto. Eseguo,
  ricevo `+1` e il nuovo stato.
- **Cosa cambia dopo lo step**: aggiorno **una sola cella** della Q-table,
  `Q[s][a]`, spostandola di `α` verso `r + γ max Q[s']`. Nient'altro cambia.
- **Cosa cambia dopo l'episodio**: nulla di speciale - la Q-table è già stata
  aggiornata step per step. Registro solo il return totale per la curva.
- **Perché migliora**: all'inizio i valori sono casuali e l'agente cade subito.
  Ma ogni volta che sopravvive un po' di più, propaga "questo stato porta a
  return alti" all'indietro. Dopo migliaia di episodi, gli stati vicini alla
  posizione bilanciata hanno Q alti e la policy greedy li insegue.

Differenze da tenere a mente:
- **SARSA** fa la stessa cosa, ma per aggiornare usa l'azione `A'` che
  *effettivamente* prenderà (scelta prima dell'update), non il `max`.
- **MC/TD prediction** *non* toccano la policy: la ricevono fissa e si limitano
  a stimare quanto vale ogni stato sotto quella policy.

Cosa è "memorizzato": solo la Q-table (control) o la V-table (prediction). Non
c'è nessun buffer, nessuna rete: tutta la conoscenza è in quel dizionario.

## Debugging e trappole comuni

| Sintomo | Causa probabile | Dove guardare |
|---|---|---|
| SARSA e Q-Learning danno curve identiche | Probabile bug: SARSA usa il `max` invece di `Q[S'][A']` | `sarsa.py`: il target deve usare `next_action` scelto *prima* dell'update |
| I valori esplodono / NaN | Manca il fattore `(not done)`: si continua a sommare futuro dopo il terminale | `q_learning.py`/`td0.py`: `... * (not done)` |
| L'agente sceglie sempre l'azione 0 | Tie-breaking assente: con Q tutti a 0, `argmax` restituisce sempre il primo | `policies.py`: `greedy_action` usa `np.flatnonzero` + scelta casuale |
| Prediction: return "troppo bello" | Stai valutando la policy sbagliata (es. greedy invece di quella richiesta) | `run_cartpole.py`: `policy_fn` passato a `mc_prediction`/`td0_prediction` |
| Curve MC e TD(0) sovrapposte nel plot | **Non è un bug**: il *return* è identico per costruzione. La differenza è nella stima V | vedi "Prediction plots" sopra |
| Le due curve del value-plot partono uguali | Corretto: `V` inizializzata a 0; TD sale lentamente, MC salta subito | `td0.py`/`mc_prediction.py` |

**Come si isola un problema**: riduci a `episodes=200` e un solo seed, poi
stampa `episode_returns[-10:]`. Se non salgono affatto, il problema è
nell'update o nel discretizzatore, non nell'analisi.

## Limiti e semplificazioni

- **ε costante**: nella pratica si usa ε-decay (alto all'inizio, basso alla fine).
  Qui si usa ε fisso per semplicità e perché il confronto è più pulito.
- **Nessun ε-decay** significa che l'agente continua a esplorare anche dopo aver
  imparato, il che abbassa il return medio nelle learning curves.
- **Discretizzazione a 10 bin**: sufficiente per CartPole ma non ottimale.
  Il trade-off bin resolution è esplorato in Fase 3 (Acrobot).
- **Nessun early stopping**: si allena per un numero fisso di episodi, anche
  se l'agente ha già converguto. Questo è standard per confronti sperimentali.
