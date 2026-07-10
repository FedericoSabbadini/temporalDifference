# Simboli, parametri e strutture dati

> Tabella di riferimento per ritrovare al volo cosa significa ogni variabile,
> iperparametro, tensore o struttura dati. La colonna **"Se lo cambio…"** aiuta
> a costruire l'intuizione (cosa succede se lo tocco).

---

## 1. Iperparametri tabular (Q-Learning, SARSA, MC, TD)

📄 usati in `algorithms/tabular/*.py`, passati dagli script.

| Simbolo | Nome codice | Significato | Default | Se lo cambio… |
|---|---|---|---|---|
| `α` | `alpha` | Learning rate: quanto ogni update sposta Q/V verso il target | 0.1 | Troppo alto (1.0) → instabile/bloccato; troppo basso (0.01) → lentissimo |
| `γ` | `gamma` | Discount factor: peso del futuro | 0.99 | Basso → miope, return finale più basso; 1.0 → valorizza il lungo termine |
| `ε` | `epsilon` | Prob. di azione casuale (esplorazione) | 0.1 | 0 → nessuna esplorazione (fallisce); alto → policy rumorosa |
| — | `episodes` | Numero di episodi di training | 5000 (CartPole), 10000 (Acrobot) | Meno → non converge; più → più costo |
| — | `max_steps` | Cap di step per episodio | 1000 (CartPole), 500 (Acrobot) | Deve combaciare col limite dell'env |

**Valori di ritorno degli algoritmi** (importante per il runner):
- Control (`q_learning`, `sarsa`): `(Q, episode_returns, episode_lengths)`
- Prediction (`mc_prediction`, `td0_prediction`): `(V, episode_returns, value_history)`
- Il runner usa sempre l'**indice [1]** = `episode_returns`. 📄 `experiments/runner.py`.

---

## 2. Discretizzazione

📄 `envs/discretizer.py` + `configs.py`.

| Simbolo | Nome codice | Significato | Esempio |
|---|---|---|---|
| — | `n_bins` | Bin per dimensione | CartPole `[10,10,10,10]`; Acrobot `[6,6,6,6,6,6]` |
| — | `clip_low` / `clip_high` | Range a cui si limita ogni dimensione prima di binnare | CartPole pos `±2.4`, angle `±0.21` (fix #2) |
| — | `bin_widths` | `(high-low)/n_bins`, larghezza di ogni bin | calcolato in `__init__` |
| `s` (discreto) | tupla output | Indici di bin, chiave della Q-table | `(5, 4, 6, 5)` |

**Perché CartPole usa `±2.4`/`±0.21` e non i bound dell'osservazione (`±4.8`/`±0.418`)?**
Perché l'episodio *termina* oltre quei valori: i bin più larghi sprecherebbero
metà risoluzione su stati impossibili (fix #2). 📄 `configs.py` (commento).

---

## 3. Iperparametri DQN

📄 `algorithms/deep/dqn.py` (`train_dqn`), valori impostati in `scripts/run_pong.py`.

| Nome codice | Significato | Default (Pong) | Nota / vs paper Nature |
|---|---|---|---|
| `total_steps` | Step ambiente totali (loop step-based) | 200 000 | Paper: 50M. Qui basta mostrare apprendimento |
| `gamma` | Discount | 0.99 | = paper |
| `lr` | Learning rate di Adam | 1e-4 | Paper usa RMSProp 2.5e-4 |
| `buffer_capacity` | Dimensione replay buffer | 100 000 | Paper: 1M (non ci sta in 32 GB RAM) |
| `batch_size` | Dimensione mini-batch | 32 | = paper |
| `target_update_freq` | Ogni quanti step `θ⁻ ← θ` | 1000 | Paper: 10 000 |
| `epsilon_start/end` | ε iniziale/finale | 1.0 → 0.02 | Decadimento lineare |
| `epsilon_decay_steps` | Durata del decay | 100 000 | ~50% del training |
| `learning_starts` | Step di riempimento buffer prima di allenare | 10 000 | Warm-up |
| `use_target_network` | Flag ablazione | `True` | `False` = target = online (moving target) |
| `use_replay` | Flag ablazione | `True` | `False` = buffer di dimensione batch (online) |
| `use_double_dqn` | Flag bonus | `False` | `True` = selezione online, valutazione target |
| `device` | `"cuda"` o `"cpu"` | auto | GPU se disponibile |

**Valore di ritorno di `train_dqn`**: `(episode_returns, episode_lengths, online_net)`.

---

## 4. Tensori e forme (DQN)

📄 `dqn.py` (`_obs_to_tensor`, `_batch_to_tensor`, `_update_network`).
`B` = batch size (32), `n` = numero azioni (6 per Pong).

| Variabile | Forma | Contenuto |
|---|---|---|
| osservazione Atari | `(4, 84, 84)` | 4 frame grayscale, **channel-first** (no transpose!) |
| `states_t`, `next_states_t` | `(B, 4, 84, 84)` | batch normalizzato a `[0,1]` float32 |
| `q_values = online_net(states_t)` | `(B, n)` | Q per **tutte** le azioni |
| `q_value` (dopo `gather`) | `(B,)` | Q dell'azione effettivamente presa |
| `next_q` | `(B,)` | max (o Double) Q dello stato successivo |
| `td_target` | `(B,)` | `r + γ·next_q·(1-done)` |
| `loss` | scalare | `MSE(q_value, td_target)` |

**Trappola nota**: i frame arrivano già `(4,84,84)` da `FrameStackObservation`,
quindi **non** vanno trasposti. Un transpose sbagliato produce input malformati
alla CNN (vedi guida Fase 4, sezione Debugging).

---

## 5. Strutture dati principali

| Struttura | Tipo | Dove | Perché così |
|---|---|---|---|
| Q-table | `defaultdict(lambda: np.zeros(n_actions))` | tabular | Alloca solo stati visitati; stato assente → Q=0 |
| V-table | `defaultdict(float)` | prediction | idem per i valori di stato |
| Replay buffer | `collections.deque(maxlen=capacity)` | DQN | FIFO: quando pieno scarta la transizione più vecchia |
| Transizione | tupla `(s, a, r, s', done)` | buffer | Frame come `uint8` per risparmiare RAM |
| Risultati multi-seed | `np.ndarray (n_seeds, n_episodes)` | runner/plotting | media/std lungo `axis=0` |
| Risultati Pong | array **NaN-padded** | `run_pong.py` | seed diversi → numeri di episodi diversi |

---

## 6. Parametri sperimentali globali

| Simbolo | Valore | Dove |
|---|---|---|
| `SEEDS` | `[42, 123, 456]` | `configs.py` — i 3 seed di ogni esperimento |
| finestra smoothing | `window=20` (tabular), `10` (Pong), `1` (value plot) | `plotting.py` / script |
| baseline random Pong | ≈ `-20.4` | `run_pong.py` (`random_baseline`) |

---

## 7. Grandezze osservate (risultati reali, media su 3 seed)

> Numeri effettivi, non attesi. Fonte: `results/*.npz`, tabelle in `report/tables/`.

| Ambiente | Metrica | Valore |
|---|---|---|
| CartPole control | Q-Learning finale | 173.8 ± 39.0 |
| CartPole control | SARSA finale | 225.2 ± 61.4 |
| Acrobot control | Q-Learning finale | −344.7 ± 10.2 |
| Acrobot control | SARSA finale | −354.5 ± 30.9 |
| Pong | Full DQN | −12.1 ± 0.9 |
| Pong | Double DQN | −14.2 ± 0.8 |
| Pong | No target net | −16.3 ± 2.8 |
| Pong | No replay / No both | ≈ −20.5 (nessun apprendimento) |

Per l'interpretazione critica di questi numeri vedi le sezioni "Come leggere i
risultati" nelle guide di fase e la Sez. Risultati del report.
