# Fase 3 — Acrobot (Env 2, assegnato dal professore)

> ### 📖 Percorso di lettura
> - **Prerequisiti**: [Fase 2](guida_fase2_cartpole.md) — Acrobot **riusa gli stessi algoritmi** senza modifiche. Teoria: state aggregation, curse of dimensionality (S&B §9.1).
> - **Guida precedente**: [Fase 2](guida_fase2_cartpole.md). **Successiva**: [Fase 4](guida_fase4_pong.md) (indipendente).
> - **Tempo stimato**: ~8 minuti.
> - **Apri in parallelo**: `scripts/run_acrobot.py` e `rl_lab/configs.py` (blocco `ACROBOT`). Gli algoritmi sono quelli già letti in Fase 2.
> - **Requisito consegna coperto**: Env 2 — applicare gli algoritmi implementati e **discutere le difficoltà specifiche**: esplorazione (reward sparso) e limiti della discretizzazione. Vale ~9 pt (parte dell'analisi sperimentale).

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
Con Acrobot: 6 dimensioni × 6 bin = 6^6 = **46.656 stati**. Già grande.
Con 10 bin: 10^6 = **1.000.000 stati** — in *principio* impraticabile per tabular.
(In pratica, però, l'esperimento `bin_resolution` mostra che entro 10.000 episodi
10 bin resta la scelta migliore: la sparsità non è ancora il collo di bottiglia a
questo budget. Vedi la sezione "Come leggere i risultati".)

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
    2. Valuta con MC e TD(0)  → prediction_return.png (return, identico) +
       prediction_value.png (stima V, il confronto vero)
```

## Scelte implementative

- **10.000 episodi** (vs 5000 su CartPole): lo spazio degli stati è molto più grande,
  servono più episodi per esplorare. Con 6^6 = 46.656 stati possibili, 10.000 episodi
  non sono neanche sufficienti a visitarli tutti.

- **max_steps = 500**: Acrobot-v1 tronca a 500 step. Oltre è inutile.

- **6 bin per dimensione** come default: compromesso tra risoluzione e dimensionalità.
  L'esperimento `exp_bin_resolution` mostra esplicitamente cosa succede con 4, 6, 8, 10 bin.
  (Nota: i risultati indicano che 6 bin non è in realtà l'ottimo a 10k episodi — vedi
  "Come leggere i risultati"; resta un default ragionevole per gli altri esperimenti.)

- **ε = 0.2 nell'esperimento bin_resolution**: scelta a priori (più alto del default 0.1).
  L'esperimento di sensibilità mostra però che su Acrobot ε basso funziona meglio: ε=0.2
  è quindi una scelta conservativa, non l'ottima.

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

> **Nota**: questa sezione riporta i risultati **effettivamente osservati** dopo
> l'esecuzione (media su 3 seed). Alcuni contraddicono le ipotesi iniziali: sono
> mantenuti e spiegati così come sono, non "aggiustati". I numeri esatti sono in
> `report/tables/acrobot_control.tex` e nella Sez. Risultati del report.

### control_comparison.png
- **Osservato**: entrambe le curve partono da ≈-470 (vicino al random -500) e
  migliorano fino a un plateau rumoroso attorno a **-345 (Q-Learning) / -355
  (SARSA)**. Lontano dall'ottimo (≈-100): l'agente impara qualche spinta utile
  ma non uno swing-up efficiente.
- **Q-Learning vs SARSA**: come atteso, Q-Learning è più stabile tra i seed
  (±10 contro ±31 di SARSA). Il livello medio finale è simile.
- **Confronto con CartPole**: convergenza più lenta, più rumorosa e più lontana
  dall'ottimo — coerente con reward sparso e spazio degli stati grande.

### bin_resolution.png — ⚠️ risultato che smentisce l'ipotesi
- **Ipotesi iniziale** (annotata anche nel codice): più bin → peggio, per la
  sparsità (10 bin = 10^6 stati "impraticabili").
- **Osservato**: **l'opposto**. Entro 10.000 episodi, **10 bin è la
  configurazione MIGLIORE** (≈-300) e **4 bin la PEGGIORE** (≈-420); 6 e 8 bin
  stanno in mezzo.
- **Interpretazione corretta**: le dinamiche di Acrobot richiedono precisione
  angolare. Con soli 4 bin la discretizzazione è troppo grossolana per
  distinguere stati diversi, e questo effetto **domina** sulla sparsità. La
  curse of dimensionality è reale in principio, ma a questo budget di episodi
  non è ancora il fattore limitante.
- **Lezione**: le ipotesi vanno verificate sui dati. Function approximation
  resta comunque la via giusta per spazi molto più grandi, ma non è ciò che
  questo esperimento dimostra in queste condizioni.

### sensitivity_epsilon.png — ⚠️ risultato che smentisce l'ipotesi
- **Ipotesi iniziale**: reward sparso → serve *più* esplorazione.
- **Osservato**: i valori **bassi** (ε=0.05–0.1) danno i risultati migliori
  (≈-320); ε=0.4 è nettamente il peggiore (≈-440).
- **Interpretazione**: con ε **costante** (non decrescente), un valore alto
  disturba continuamente la policy greedy e impedisce di sfruttare lo swing-up
  appreso. Servirebbe esplorazione alta solo all'inizio (annealing), non
  costante. Questo spiega perché "più esplorazione" qui peggiora.

### prediction_return.png e prediction_value.png (dopo il fix #1)
- **prediction_return.png**: return della policy, identico per MC e TD(0)
  (stessa policy) — non distingue gli stimatori.
- **prediction_value.png**: stima V(stato di reset) nel tempo. Su Acrobot il
  **bias di TD è ancora più netto** che su CartPole: MC converge subito a
  ~-92 (vicino al valore vero), mentre TD(0) parte da 0 e scende lentamente,
  a 3000 episodi è solo a ~-57. Lo stato di riferimento è ~90 passi
  dall'obiettivo e TD propaga il valore un passo alla volta → bootstrap lento,
  stima distorta verso l'inizializzazione (0). MC usa il return completo e non
  ha questo problema.
- **Nota anchor**: lo stato di riferimento è lo stato di reset discretizzato.
  Un "bin centrale" sarebbe impossibile su Acrobot (cos e sin non possono
  essere entrambi ~0) → V resterebbe piatta a 0.

## Modello mentale (perché qui è tutto più difficile)

Rispetto a CartPole, immagina la differenza dal punto di vista dell'agente:

- **Il segnale**: su CartPole ricevo `+1` ad ogni step — so *subito* se sto
  andando bene. Su Acrobot ricevo `-1` ad ogni step *qualunque cosa faccia*,
  finché per caso non raggiungo il goal. Per centinaia di step non ho **nessuna
  informazione** su quali azioni siano migliori: tutte danno `-1`.
- **Cosa cambia dopo un episodio "fallito"** (che non raggiunge il goal): quasi
  nulla di utile — ho solo imparato che tanti stati valgono `≈ -500`. Il vero
  apprendimento parte solo dopo il *primo* episodio che, per esplorazione,
  inciampa nel goal e ottiene un return meno negativo.
- **Cosa è memorizzato**: una Q-table potenzialmente enorme (46 656 stati con 6
  bin), di cui in pratica ne visito solo una frazione. Molte celle restano a 0
  per sempre.
- **Perché è rumoroso**: con reward sparso e tanti stati, la propagazione del
  valore all'indietro è lenta e dipende molto dal caso (quali episodi hanno
  trovato il goal). Da qui le bande di deviazione standard larghe.

Questa è la ragione profonda per cui *lo stesso codice* che su CartPole vola,
su Acrobot arranca: non è un bug, è la natura del problema (reward sparso +
spazio grande). È esattamente ciò che la consegna chiede di **discutere**.

## Debugging e trappole comuni

| Sintomo | Causa probabile | Dove guardare |
|---|---|---|
| Il value-plot di prediction è **piatto a 0** | Lo stato di riferimento non viene mai visitato | `run_acrobot.py`: l'anchor deve essere lo **stato di reset**, non il "bin centrale" (che su Acrobot è impossibile: cos e sin non possono essere entrambi ~0) |
| Tutte le curve `bin_resolution` sembrano uguali/mescolate | Chiusura Python: la lambda cattura la variabile del ciclo per riferimento | `run_acrobot.py`: il trucco `def _make(b=bins)` fissa il valore corrente |
| Memoria/tempo esplodono aumentando i bin | Atteso: `(bin)^6` cresce esponenzialmente | è il punto dell'esperimento, non un bug |
| L'agente resta a ~−500 (mai migliora) | Esplorazione insufficiente per trovare il goal, **oppure** troncamento a 500 troppo corto | `configs.py`/`run_acrobot.py`: `epsilon`, `max_steps` |
| Return "migliore" con meno bin | **Non per forza un bug**: a questo budget bin fini vincono; verifica prima di dubitare | vedi "bin_resolution.png" sopra |

**Come si verifica l'anchor**: dopo `env.reset()`, la tupla dello stato di reset
deve avere `cos θ ≈ 1` (link in basso) → bin alto su quelle dimensioni, non
centrale. Se stampi l'anchor e vedi tutti indici centrali, è il bug del bin
centrale.

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
