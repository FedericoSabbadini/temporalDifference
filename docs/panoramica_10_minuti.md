# Il progetto in 10 minuti

> Lettura veloce per riottenere il quadro d'insieme senza aprire il codice.
> Se hai solo 10 minuti prima di riprendere il progetto (o prima dell'orale),
> leggi questo.

## Cos'è

Progetto finale del corso di **Reinforcement Learning** (UniBS). Implementa da
zero i principali algoritmi RL e li confronta su **tre ambienti** di difficoltà
crescente. L'obiettivo dichiarato dalla consegna è la **comprensione e l'analisi
sperimentale critica**, non la performance assoluta.

## I tre ambienti e cosa dimostrano

| Ambiente | Metodo | Cosa insegna |
|---|---|---|
| **CartPole** | 4 algoritmi tabular | I fondamenti: control vs prediction, on/off-policy, bias/varianza, failure cases |
| **Acrobot** | stessi tabular | I *limiti* del tabular: curse of dimensionality, reward sparso |
| **Pong** | DQN da zero (PyTorch) | Deep RL: replay + target network, ablation, Double DQN |

## I quattro algoritmi tabular

Due imparano la policy (**control**), due stimano i valori di una policy fissa
(**prediction**):

- **Q-Learning** (control, off-policy): target `R + γ max Q(S',·)`
- **SARSA** (control, on-policy): target `R + γ Q(S',A')`
- **Monte Carlo** (prediction): usa il return completo `G` (no bias, alta varianza)
- **TD(0)** (prediction): usa il bootstrap `R + γV(S')` (bias, bassa varianza)

Poiché gli ambienti hanno **stato continuo**, prima si **discretizza** (binning)
per poter usare una Q-table.

## Il DQN in una frase

Sostituisce la Q-table con una **rete neurale** `Q(s,·;θ)`; per stabilizzare
l'addestramento usa **experience replay** (mini-batch casuali dal buffer) e una
**target network** (copia lenta della rete per calcolare il target). Il bonus
**Double DQN** riduce la sovrastima del `max`.

## I risultati in una riga ciascuno

- **CartPole**: entrambi apprendono; con la discretizzazione fine (fix #2) SARSA
  supera Q-Learning ma le curve sono più rumorose e ancora in salita a 5000 ep.
- **Acrobot**: apprendimento parziale (~−345), lontano dall'ottimo. **Due
  sorprese**: bin più fini vanno *meglio* e ε basso va *meglio* — contro le
  ipotesi iniziali, mantenute e analizzate.
- **Pong**: l'**experience replay è essenziale** (senza, non impara affatto); la
  target network accelera. Full DQN −12.1, Double DQN −14.2 (random ≈ −20.4).

## Le due correzioni chiave (fix)

- **Fix #1**: il confronto MC vs TD(0) ora traccia la **stima V(s)** nel tempo
  (non solo il return, che è identico tra i due) → mostra davvero bias/varianza.
- **Fix #2**: i bin di CartPole ristretti ai limiti di *terminazione* → più
  risoluzione utile, ma convergenza più lenta (trade-off, non miglioria netta).

## Com'è organizzato il codice

```
rl_lab/          il "motore": algoritmi + ambienti + utilità (codice riutilizzabile)
scripts/         un file per ambiente: orchestrano gli esperimenti
results/         array grezzi .npz + checkpoint DQN
figures/         grafici generati
report/          report LaTeX (16→20 pagine, in italiano)
docs/            queste guide
```

Regola mentale: **`rl_lab/` è puro** (ambiente in ingresso, risultati in
uscita); **`scripts/` orchestra**; **il plotting è separato** dagli algoritmi.

## Cosa è farina del sacco e cosa no

- I **4 algoritmi tabular** sono *adattati* dalle esercitazioni del corso.
- Il **DQN, il discretizzatore, tutta l'infrastruttura e gli script** sono
  *nuovi*, scritti da zero.
- Nessun codice **Stable-Baselines3** nel core (vietato dalla consegna).

## Dove andare adesso

- Vuoi il percorso guidato di 1 ora? → [`walkthrough_1_ora.md`](walkthrough_1_ora.md)
- Ripasso mirato per l'orale? → [`ripasso_orale.md`](ripasso_orale.md)
- Studiare una fase col codice a fianco? → guide `guida_fase1…4`
- Indice completo e ordine di studio → [`README.md`](README.md)
