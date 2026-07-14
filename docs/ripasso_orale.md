# Prima dell'orale - ripasso rapido

> Domande probabili con risposte sintetiche. Pensato per l'ultimo ripasso.
> Include i punti "scomodi" (risultati che smentiscono le ipotesi, limiti,
> scelte discutibili): all'orale è meglio arrivarci preparati e onesti.

---

## 1. Domande sui fondamenti

**Control vs prediction?**
Control impara/migliora la policy (Q-Learning, SARSA). Prediction stima `V(s)`
di una policy **fissa** (MC, TD(0)). Nel codice: gli algoritmi di prediction
ricevono `policy_fn` come input, quelli di control no.

**Differenza SARSA / Q-Learning?**
Il target. Q-Learning (off-policy): `R + γ max_a Q(S',a)`. SARSA (on-policy):
`R + γ Q(S',A')` con `A'` l'azione effettivamente scelta da ε-greedy.
Q-Learning valuta la greedy ignorando l'esplorazione; SARSA valuta la policy che
segue davvero. Esempio classico: CliffWalking (S&B Fig. 6.5).

**MC vs TD(0), bias e varianza?**
MC usa il return completo `G`: non distorto, alta varianza, aggiorna a fine
episodio. TD(0) usa il bootstrap `R + γV(S')`: distorto (usa una stima), bassa
varianza, aggiorna online. Il TD error `δ` è la "sorpresa".

**Cos'è il bootstrap?**
Aggiornare una stima usando altre stime (TD sì, MC no). È la causa del *bias* di
TD ma anche della sua bassa varianza e capacità di apprendere online.

**Perché ε-greedy?**
È il metodo di esplorazione più semplice (S&B §2.1) e l'unico richiesto. Con
prob. ε azione casuale, altrimenti greedy.

---

## 2. Domande sulla discretizzazione

**Perché discretizzi?**
I metodi tabular indicizzano `Q[s]`: serve uno spazio di stati finito e
hashable. CartPole/Acrobot hanno stato continuo - binning per ogni dimensione - 
tupla di indici. Concetto: *state aggregation* (S&B §9.4).

**Qual è il trade-off dei bin?**
Pochi bin - si perde informazione (policy grossolana). Molti bin - lo spazio
esplode come `(bin)^dim` (curse of dimensionality), servono più episodi.

**Sorpresa su Acrobot (preparati, te lo chiederanno):**
L'ipotesi era "più bin = peggio". I dati mostrano il contrario a 10k episodi:
**10 bin è il migliore, 4 il peggiore**. Motivo: Acrobot ha bisogno di
*precisione angolare*; con 4 bin la griglia è troppo grezza e questo effetto
domina sulla sparsità. La curse è reale *in principio*, ma non è il collo di
bottiglia a questo budget. **Lezione**: le ipotesi vanno verificate sui dati.

**Discretizzazione di CartPole - cosa cambia e con quale effetto?**
Ho ristretto i range di clipping di posizione e angolo dai bound
dell'osservazione (`±4.8`/`±0.418`) ai limiti di **terminazione** (`±2.4`/`±0.21`):
metà dei bin era sprecata su stati impossibili. Effetto **a doppio taglio**:
alza il tetto (SARSA ~225, curve ancora in salita) ma rallenta la convergenza e
aumenta la varianza. Non è una miglioria netta, ed è giusto dirlo.

---

## 3. Domande sul DQN

**Perché DQN e non tabular su Pong?**
Lo spazio dei pixel è astronomico: la Q-table è impossibile. Si approssima `Q`
con una rete: `Q(s,·;θ)`.

**A cosa serve l'experience replay?**
Le transizioni consecutive sono correlate; il gradient descent assume campioni
i.i.d. Il buffer + il campionamento casuale rompono la correlazione e permettono
di riusare le esperienze. **Nel nostro ablation: senza replay il DQN non impara
affatto** (resta a ~−20.5) - è il componente più critico.

**A cosa serve la target network?**
A dare un target stabile: usare la stessa rete per predizione e target crea un
"moving target". La copia `θ⁻` viene sincronizzata ogni 1000 step. Nel nostro
ablation: senza target net impara comunque, ma più lentamente e con più varianza.

**Double DQN - cosa risolve?**
Il maximization bias (S&B §6.7): il `max` sovrastima su stime rumorose. Double
DQN separa selezione (rete online) e valutazione (rete target) dell'azione.
Nel nostro regime breve (200k step) **non supera** il DQN base (−14.2 vs −12.1):
il vantaggio emerge con più training, ed è normale.

**In cosa il target del DQN è uguale al Q-Learning?**
È identico: `y = R + γ max_a' Q(S',a')`. La differenza è solo *come* si
avvicina Q a `y`: gradient descent con MSE (DQN) invece di update tabellare
(`Q[s][a] += α(y − Q[s][a])`).

**Perché solo 200k step?**
Budget. Il paper Nature usa 50M. La consegna chiede di mostrare apprendimento
vs random, non alta performance. Per questo Pong si ferma a ~−12, non a punteggi
positivi.

---

## 4. Domande metodologiche

**Come garantisci la riproducibilità?**
Ogni run fissa il seed (numpy + torch + env). Ogni risultato è media su 3 seed
`{42,123,456}` con banda ±1σ. Gli array grezzi sono salvati in `.npz` e le
tabelle del report sono generate da quei dati (nessun numero a mano).

**Il confronto MC vs TD(0): perché due grafici?**
Il grafico del *return* è identico per costruzione (stessa policy) e non mostra
la differenza. Il grafico della *stima V(s)* sì: MC converge subito con
alta varianza, TD sale lentamente ed è distorta all'inizio. Su Acrobot il bias
di TD è ancora più evidente (a 3000 episodi TD è a −57, MC a −92).
Limite: è una misura *locale* (un solo stato di riferimento).

**Failure case: quali e perché obbligatori?**
La consegna chiede di analizzare i risultati negativi. Due casi: α=1.0 (update
troppo aggressivo - bloccato ~65) e ε=0 (nessuna esplorazione - bloccato ~40).

**Cosa è tuo e cosa dal corso?**
I 4 algoritmi tabular sono *adattati* dalle esercitazioni. Il DQN, il
discretizzatore, tutta l'infrastruttura e gli script sono *nuovi*. Nessun SB3 nel
core. Dettaglio in [`origine_codice.md`](origine_codice.md).

---

## 5. Mappa consegna - dove l'ho soddisfatta

| Requisito | Dove |
|---|---|
| 4 algoritmi tabular da zero | `algorithms/tabular/` |
| Discretizzazione | `envs/discretizer.py` |
| Curve, sensitivity, SARSA-vs-QL, MC-vs-TD, failure | Fase 2, `run_cartpole.py` |
| Acrobot + difficoltà esplorazione/discretizzazione | Fase 3, `run_acrobot.py` |
| DQN da zero + ablation (replay/target) | Fase 4, `dqn.py`, `run_pong.py` |
| Double DQN (bonus) | flag `use_double_dqn` |
| Media 3 seed + std | tutto, via `runner.py` |
| Report con failure analysis | `report/` (20 pagine) |

---

## 6. Le tre cose da non dimenticare

1. **Onestà sui risultati**: due esperimenti Acrobot smentiscono le ipotesi e
   questa scelta è un trade-off - dirlo è un punto di forza, non di debolezza.
2. **Generalità del codice**: gli stessi algoritmi girano su CartPole e Acrobot
   senza modifiche - è il senso della separazione `rl_lab/` vs `scripts/`.
3. **Il filo teorico**: tutto ruota attorno allo stesso target di Bellman, dal
   Q-Learning tabellare al DQN.
