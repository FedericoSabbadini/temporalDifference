# Il progetto in 1 ora - walkthrough guidato

> Percorso di studio da fare **con il codice aperto a fianco**. Ogni tappa dice
> quale guida leggere, quali file aprire e cosa cercarci. Segui l'ordine:
> è costruito per accumulare comprensione gradualmente.
>
> **Prerequisiti**: aver letto [`panoramica_10_minuti.md`](panoramica_10_minuti.md).
> Tieni aperti in un'altra scheda il [glossario](glossario.md) e i
> [simboli/parametri](simboli_e_parametri.md).

Tempo totale ≈ 60 min. I minuti sono indicativi.

---

## Tappa 0 - Orientamento (5 min)

Apri la **radice del repo** e guarda la struttura di cartelle. Fissa la regola:
`rl_lab/` = motore riutilizzabile, `scripts/` = orchestratori, il resto = output.

> Domanda-guida: *"Se volessi lanciare tutti gli esperimenti di un ambiente,
> quale singolo file eseguirei?"* - uno `scripts/run_<env>.py`.

---

## Tappa 1 - Infrastruttura (12 min) - Fase 1

**Leggi**: [`guida_fase1_introduzione.md`](guida_fase1_introduzione.md).
**Apri in parallelo**, in quest'ordine:

1. `rl_lab/configs.py` - nota `CARTPOLE`, `ACROBOT`, `SEEDS`. Sono *dati*, non
   logica: qui vivono bin e range di clipping.
2. `rl_lab/envs/discretizer.py` - la classe `DiscretizedEnv`. Segui il metodo
   `observation()`: continuo - clip - scala - indice di bin - **tupla**.
   *È il ponte tra stato continuo e Q-table.*
3. `rl_lab/policies.py` - `greedy_action` (con tie-breaking) e
   `epsilon_greedy_action`. Due funzioni, tutta l'esplorazione del tabular.
4. `rl_lab/experiments/runner.py` - `run_multi_seed`: il ciclo sui 3 seed.
   Nota che prende `result[1]` (i return) da qualunque algoritmo.
5. `rl_lab/utils/{seeding,plotting}.py` - dai solo un'occhiata: seed combinato,
   e `plot_learning_curves` che fa media ± banda lungo `axis=0`.

> Modello mentale da fissare: *un esperimento = "crea env discretizzato - esegui
> algoritmo per 3 seed - media e plotta".* Tutto il resto è dettaglio.

---

## Tappa 2 - I 4 algoritmi tabular (15 min) - Fase 2

**Leggi**: [`guida_fase2_cartpole.md`](guida_fase2_cartpole.md), sezioni
"Collegamento con la teoria" e "Flusso del codice".
**Apri in parallelo** (sono corti, ~70 righe ciascuno - leggili *tutti*):

1. `algorithms/tabular/q_learning.py` - trova la riga dell'update:
   `td_target = reward + gamma * np.max(Q[next_state]) * (not done)`. È la
   Bellman di ottimalità. Il `(not done)` azzera il futuro allo stato terminale.
2. `algorithms/tabular/sarsa.py` - stesso schema, ma il target usa `Q[S'][A']`
   e `A'` è scelto **prima** dell'update. Confronta le due funzioni fianco a
   fianco: la differenza è *una riga*. Quella riga è tutta la distanza tra
   on-policy e off-policy.
3. `algorithms/tabular/td0.py` - l'update `V += α·δ`, con `δ` il TD error.
4. `algorithms/tabular/mc_prediction.py` - il ciclo *all'indietro*
   `G = γ·G + r` e la media incrementale. Nota il parametro `track_state`
   (serve al confronto sulle stime V).

**Poi apri** `scripts/run_cartpole.py` e scorri le 4 funzioni `exp_*`: sono
l'incarnazione del modello mentale della Tappa 1, ripetuto 4 volte.

> Domanda-guida: *"Perché MC riceve `policy_fn` e Q-Learning no?"* - perché MC è
> **prediction** (valuta una policy data), Q-Learning è **control** (la impara).

---

## Tappa 3 - I limiti del tabular (8 min) - Fase 3

**Leggi**: [`guida_fase3_acrobot.md`](guida_fase3_acrobot.md), in particolare
"Curse of dimensionality" e le due sezioni "risultato che smentisce l'ipotesi".
**Apri**: `scripts/run_acrobot.py`, funzione `exp_bin_resolution()`.

Punto chiave: **il codice è lo stesso della Fase 2** - cambia solo l'ambiente
(config `ACROBOT`) e il set di esperimenti. Questo dimostra la generalità degli
algoritmi. Le due sorprese (bin fini meglio, ε basso meglio) sono l'esempio
principe di "verifica le ipotesi sui dati".

---

## Tappa 4 - Il DQN (15 min) - Fase 4

**Leggi**: [`guida_fase4_pong.md`](guida_fase4_pong.md), sezioni "Le due
innovazioni chiave" e "Dentro `train_dqn()`".
**Apri in parallelo**:

1. `algorithms/deep/networks.py` - `NatureCNN`: 3 conv + 2 FC. Output = un Q per
   azione. (Se conosci le CNN dal corso di DL, qui non c'è nulla di nuovo.)
2. `algorithms/deep/replay_buffer.py` - `push` e `sample`. È un `deque`. Il
   `sample` casuale è *tutto* il valore dell'experience replay.
3. `algorithms/deep/dqn.py` - il cuore. Segui `train_dqn` dall'alto:
   scelta ε-greedy - `env.step` - `buffer.push` - (se pronto) `_update_network`
 - (ogni N step) sync target. Poi leggi `_update_network`: è **un** passo di
   Q-Learning fatto con backprop invece che con l'update tabellare.
   Cerca il ramo `if use_double_dqn`: sono le ~3 righe del bonus.
4. `envs/atari_wrappers.py` - il preprocessing (grayscale, 84×84, stack 4).

**Poi apri** `scripts/run_pong.py`: le 5 chiamate `run_config` con i flag di
ablazione diversi. Nota `random_baseline()` come riferimento.

> Ponte concettuale: `_update_network` ↔ l'update di `q_learning.py`. Stesso
> target `r + γ max Q(s',·)`, ma minimizzato con MSE + Adam invece che
> `Q[s][a] += α(target − Q[s][a])`.

---

## Tappa 5 - Dai numeri al report (5 min)

**Apri**: `scripts/make_report_tables.py` (legge i `.npz` e genera le tabelle
LaTeX - nessun numero scritto a mano) e sfoglia `report/sections/results.tex`.

Chiudi il cerchio del **flusso completo**:

```
configs.py - make_env (discretizer) - algoritmo - runner (3 seed)
 - results/*.npz - plotting - figures/*.png
 - make_report_tables.py - report/tables/*.tex - report/main.tex - PDF
```

---

## Cosa dovresti saper rispondere a fine ora

1. Perché serve la discretizzazione, e qual è il suo trade-off?
2. Qual è l'*unica* riga che distingue SARSA da Q-Learning?
3. Perché MC e TD(0) hanno lo stesso return ma stime V diverse?
4. Cosa fa, concretamente, l'experience replay e perché senza il DQN non impara?
5. In cosa il target del DQN è identico a quello del Q-Learning tabular?

Se rispondi a queste 5, hai ricostruito l'ossatura del progetto.
Per il ripasso mirato all'orale - [`ripasso_orale.md`](ripasso_orale.md).
