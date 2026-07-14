# Glossario - termini RL usati nel progetto

> Ogni voce collega il **concetto teorico** (che già conosci da Sutton & Barto)
> al **punto preciso del codice** in cui compare. Non è un glossario generico:
> serve a ritrovare *dove vive* ogni concetto in questo repository.

Legenda: = file principale dove il concetto è implementato.

---

## Concetti fondamentali (MDP)

**Stato (state, `s`)**
La situazione osservata dall'agente. In CartPole/Acrobot è un vettore continuo
(es. `[cart_pos, cart_vel, angle, ang_vel]`); in Pong sono 4 frame 84×84.
Nel codice tabular lo stato diventa una **tupla di indici di bin** dopo la
discretizzazione. `envs/discretizer.py` (`DiscretizedEnv.observation`).

**Azione (action, `a`)**
Ciò che l'agente sceglie di fare. CartPole: 2 (sinistra/destra); Acrobot: 3
(torque -1/0/+1); Pong: 6. Nel codice: `env.action_space.n`. La scelta avviene
in `policies.py` (tabular) o dentro il loop di `algorithms/deep/dqn.py` (DQN).

**Reward (`r`)**
Segnale scalare immediato. CartPole: +1 per step (denso); Acrobot: -1 per step
(sparso); Pong: +1/-1 per punto. Compare come `reward` nel valore di ritorno di
`env.step(action)` in ogni algoritmo.

**Return (`G`)**
Somma scontata dei reward futuri: `G_t = R_{t+1} + γ R_{t+2} + …`. È ciò che
l'agente massimizza. Calcolato all'indietro in `mc_prediction.py`
(`G = gamma * G + r`).

**Discount factor (`γ`, gamma)**
Peso dei reward futuri (0 = miope, 1 = lungo termine). Iperparametro, default
`0.99`. `configs.py` (non fisso lì, passato agli algoritmi dagli script).

**Policy (`π`)**
La strategia dell'agente: da stato ad azione. In questo progetto è sempre
ε-greedy (control) o una policy fissa da valutare (prediction). `policies.py`.

**MDP (Markov Decision Process)**
Il modello formale (stati, azioni, transizioni, reward). Non c'è una classe
"MDP" nel codice: è l'ambiente Gymnasium stesso a fornire l'interfaccia MDP
(`reset`, `step`).

---

## Funzioni di valore

**State-value (`V(s)`)**
Return atteso partendo da `s` seguendo `π`. Stimata dagli algoritmi di
**prediction**. `mc_prediction.py`, `td0.py` (dict `V`).

**Action-value / Q-function (`Q(s,a)`)**
Return atteso partendo da `s`, azione `a`, poi `π`. Stimata dagli algoritmi di
**control**. `q_learning.py`, `sarsa.py` (`defaultdict` `Q`). Nel DQN è la
rete `Q(s,·;θ)`. `networks.py`.

**Q-table**
La tabella `stato - valori-per-azione`. Implementata come
`defaultdict(lambda: np.zeros(n_actions))`: alloca solo gli stati visitati.
`q_learning.py`, `sarsa.py`.

**Bellman (equazione / update)**
La regola ricorsiva dietro ogni update. Q-Learning usa la Bellman *ottimalità*
(`max`), SARSA la Bellman di *valutazione* (azione effettiva). Vedi le formule
in `q_learning.py` / `sarsa.py` (commento sopra la funzione).

**TD error (`δ`)**
`δ = R + γ V(S') - V(S)` (o con Q). Misura la "sorpresa". `td0.py`
(`td_error = td_target - V[state]`).

**Bootstrap**
Usare una stima corrente per aggiornarne un'altra (TD sì, MC no). È la ragione
del *bias* di TD. `td0.py`, e nel target del DQN `dqn.py`.

---

## Control vs Prediction

**Control**
Imparare/migliorare la policy. Q-Learning, SARSA. Restituiscono `(Q, returns,
lengths)`. `q_learning.py`, `sarsa.py`.

**Prediction**
Stimare `V(s)` per una policy **fissa** (non la migliorano). Monte Carlo, TD(0).
Ricevono `policy_fn` come input. `mc_prediction.py`, `td0.py`.

**On-policy / Off-policy**
SARSA è on-policy (valuta la policy che segue, incl. l'esplorazione);
Q-Learning è off-policy (valuta la greedy tramite `max`, indipendentemente da
come esplora). Differenza visibile nel target: `Q[S'][A']` vs `max Q[S']`.

**Monte Carlo (MC)**
Aggiorna con il **return completo** `G` a fine episodio: non distorto, alta
varianza. `mc_prediction.py`.

**TD(0)**
Aggiorna ad ogni step con il **bootstrap** `R + γV(S')`: distorto, bassa
varianza, online. `td0.py`.

**First-visit MC**
Conta solo la prima occorrenza di uno stato in un episodio. `mc_prediction.py`
(`if first_visit and s in visited: continue`).

---

## Esplorazione

**Exploration / Exploitation**
Il dilemma tra provare azioni nuove e sfruttare le migliori note. `policies.py`.

**ε-greedy**
Con prob. `ε` azione casuale, altrimenti greedy. `policies.py`
(`epsilon_greedy_action`). Nel DQN l'ε **decresce** linearmente. `dqn.py`.

**Greedy policy / tie-breaking**
`π(s) = argmax_a Q(s,a)`, con scelta casuale tra i massimi (evita bias verso
l'azione 0). `policies.py` (`greedy_action`, `np.flatnonzero`).

**ε-decay (annealing)**
Ridurre ε nel tempo (esplora molto all'inizio, poco dopo). Presente solo nel
DQN. `dqn.py` (`epsilon = max(epsilon_end, …)`).

---

## Discretizzazione (tabular su stato continuo)

**Discretizzazione / binning**
Convertire uno stato continuo in indici discreti dividendo ogni dimensione in
bin. `envs/discretizer.py`.

**State aggregation (S&B §9.4)**
Il concetto teorico dietro il binning: stati vicini condividono lo stesso
valore stimato. `envs/discretizer.py`.

**Curse of dimensionality**
Lo spazio degli stati cresce come `(bin)^dim`. È il tema dell'esperimento
`exp_bin_resolution`. `scripts/run_acrobot.py`.

**Clipping range**
Intervallo `[clip_low, clip_high]` a cui si limita ogni dimensione prima di
binnare. Per CartPole ristretto ai limiti di *terminazione*.
`configs.py`.

---

## Deep RL (DQN)

**DQN (Deep Q-Network)**
Q-Learning con la Q-function approssimata da una rete neurale. `dqn.py`.

**Experience Replay / Replay buffer**
Buffer di transizioni da cui si campionano mini-batch casuali: rompe la
correlazione temporale. `replay_buffer.py`.

**Target network (`θ⁻`)**
Copia della rete aggiornata di rado, usata per calcolare il target stabile.
`dqn.py` (`target_net`, `load_state_dict` ogni `target_update_freq`).

**Online network (`θ`)**
La rete addestrata ad ogni step (sceglie le azioni). `dqn.py` (`online_net`).

**Moving target problem**
L'instabilità che nasce se il target usa la stessa rete che si aggiorna;
risolto dalla target network. Discusso in `dqn.py` (commenti).

**Double DQN (van Hasselt 2016)**
Disaccoppia selezione (online) e valutazione (target) dell'azione, riducendo il
maximization bias. `dqn.py` (`use_double_dqn`).

**Maximization bias (S&B §6.7)**
La tendenza del `max` a sovrastimare i valori su stime rumorose. Motiva Double
DQN/Double Q-Learning.

**Nature CNN**
L'architettura convoluzionale del paper Mnih 2015. `networks.py` (`NatureCNN`).

**Frame stacking / frame skip**
Impilare 4 frame (per percepire il moto) e ripetere l'azione per 4 frame
(efficienza). `envs/atari_wrappers.py`.

**Ablation study**
Rimuovere un componente per misurarne il contributo (qui: ±replay, ±target).
`scripts/run_pong.py`.

---

## Metodologia sperimentale

**Seed / riproducibilità**
Fissare i generatori casuali per rendere ripetibile una run. `utils/seeding.py`.

**Multi-seed averaging**
Ogni risultato = media su 3 seed `{42,123,456}` con banda ±1σ (requisito
consegna). `experiments/runner.py`, `utils/plotting.py`.

**Learning curve**
Return vs episodi, smussata con media mobile. `utils/plotting.py`
(`plot_learning_curves`, `moving_average`).

**Sensitivity analysis**
Come cambia l'apprendimento variando un iperparametro (ε/α/γ). 
`utils/plotting.py` (`plot_sensitivity`), `scripts/run_cartpole.py` (`_sweep`).

**Failure case**
Configurazione che fallisce di proposito (α=1.0, ε=0), da analizzare non
nascondere. `scripts/run_cartpole.py` (`exp_failure_case`).

---

## Termini che *non* compaiono (e perché)

Alcuni termini che potresti cercare **non** sono usati, per scelta:
- **Policy gradient / Actor-Critic**: il progetto usa solo metodi *value-based*
  (Q-Learning, SARSA, DQN). Nessun metodo a gradiente di policy è richiesto.
- **Prioritized Experience Replay, Dueling DQN**: citati come possibili
  estensioni future, non implementati.
- **Reward shaping**: non usato (Acrobot resta a reward sparso di proposito).
