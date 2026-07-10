# Mappa del codice — classi, funzioni e file

> Glossario navigabile del *codice* (non della teoria: per quella vedi
> [`glossario.md`](glossario.md)). Per ogni file: cosa contiene, le funzioni/classi
> chiave con la loro firma, e chi le chiama. Serve a orientarsi rapidamente.

Convenzione firme: `nome(argomenti) → valore_di_ritorno`.

---

## `rl_lab/` — il motore

### `configs.py`
Parametri come dict Python. Nessuna logica.
- `CARTPOLE`, `ACROBOT`, `PONG` — dict con `env_name`, `n_bins`, `clip_low/high`.
- `SEEDS = [42, 123, 456]` — i 3 seed usati ovunque.

### `policies.py` — selezione dell'azione (tabular)
- `greedy_action(Q, state) → action` — `argmax_a Q(s,a)` con tie-breaking casuale
  (`np.flatnonzero` + `np.random.choice`). Usata da SARSA/Q-Learning e come
  policy da valutare in prediction.
- `epsilon_greedy_action(Q, state, epsilon, n_actions) → action` — con prob. ε
  azione casuale, altrimenti `greedy_action`.

### `envs/discretizer.py` — stato continuo → discreto
- `class DiscretizedEnv(gym.ObservationWrapper)`
  - `__init__(env, n_bins, clip_low, clip_high)` — precalcola `bin_widths`.
  - `observation(obs) → tuple` — clip → scala → indice di bin → **tupla**
    (chiave della Q-table). È il metodo chiamato automaticamente da Gymnasium
    ad ogni `reset`/`step`.

### `envs/atari_wrappers.py` — preprocessing Atari
- `make_atari_env(env_name, render_mode=None) → env` — applica
  `AtariPreprocessing` (grayscale, 84×84, frame-skip 4) + `FrameStackObservation`
  (stack 4). Registra gli env ALE.

### `algorithms/tabular/` — i 4 algoritmi
Tutti prendono l'`env` e restituiscono tuple con `episode_returns` all'**indice [1]**.

- `q_learning(env, episodes, alpha, gamma, epsilon, max_steps) → (Q, returns, lengths)`
  — control off-policy. Update con `max` sul next-state.
- `sarsa(env, episodes, alpha, gamma, epsilon, max_steps) → (Q, returns, lengths)`
  — control on-policy. Sceglie `A'` prima dell'update.
- `mc_prediction(env, policy_fn, episodes, gamma, first_visit, max_steps, track_state) → (V, returns, value_history)`
  — prediction Monte Carlo. `value_history` = stima di `V(track_state)` per il fix #1.
- `td0_prediction(env, policy_fn, episodes, alpha, gamma, max_steps, track_state) → (V, returns, value_history)`
  — prediction TD(0). Stessa struttura di `mc_prediction`.

### `algorithms/deep/` — il DQN
- `networks.py`
  - `class MLP(nn.Module)` — rete per osservazioni-vettore (debug su CartPole).
  - `class NatureCNN(nn.Module)` — CNN del paper Nature per Atari; `forward` →
    Q per tutte le azioni.
- `replay_buffer.py`
  - `class ReplayBuffer` — `push(s,a,r,s',done)`, `sample(batch_size) → tuple di
    array`, `__len__`. FIFO su `deque`.
- `dqn.py` (il cuore, tutto nuovo)
  - `_obs_to_tensor(obs, device)` / `_batch_to_tensor(batch, device)` — numpy →
    tensore normalizzato (nessun transpose: i frame sono già channel-first).
  - `_update_network(online, target, buffer, optimizer, batch_size, gamma, device, use_double_dqn) → loss`
    — **un** gradient step. Contiene il ramo Double DQN.
  - `train_dqn(env, total_steps, …, use_target_network, use_replay, use_double_dqn, …) → (returns, lengths, online_net)`
    — il training loop step-based completo (ε-decay, replay, target sync).

### `experiments/runner.py`
- `run_multi_seed(make_env_fn, train_fn, seeds, **train_kwargs) → (returns_array, all_results)`
  — esegue `train_fn` una volta per seed (con `set_seed`), impila i return in un
  array `(n_seed, n_episodi)`. `all_results` conserva le tuple complete (serve a
  estrarre `value_history` per il fix #1).

### `utils/`
- `seeding.py` → `set_seed(seed)` — seed di numpy + torch (CPU/CUDA). Chiamato da
  `run_multi_seed` e dagli script.
- `plotting.py`
  - `moving_average(x, window) → array` — media mobile (`np.convolve`).
  - `plot_learning_curves(results_dict, title, save_path, window, ylabel)` — mean
    + banda ±σ per ogni algoritmo.
  - `plot_sensitivity(results_by_param, param_name, algo_name, …)` — una curva per
    valore dell'iperparametro (senza banda, per leggibilità).

---

## `scripts/` — gli orchestratori

### `run_cartpole.py`
- `make_cartpole() → env` — CartPole discretizzato.
- `exp_control_comparison()` — Q-Learning vs SARSA → `control_comparison.png/.npz`.
- `_sweep(algo_fn, algo_name, param_name, param_values, defaults)` — helper per gli sweep.
- `exp_sensitivity()` — sweep ε/α/γ per QL e SARSA (6 figure).
- `exp_prediction_comparison()` — MC vs TD(0): grafici `_return` e `_value` (fix #1).
- `exp_failure_case()` — α=1.0 e ε=0.

### `run_acrobot.py`
- `make_acrobot() → env`.
- `exp_control_comparison()`, `exp_bin_resolution()` (l'esperimento originale),
  `exp_sensitivity_epsilon()`, `exp_prediction()`.

### `run_pong.py`
- `make_pong() → env`.
- `run_config(label, seeds, **dqn_kwargs) → array` — allena una configurazione su
  3 seed, salva i checkpoint, fa il padding NaN dei return.
- `random_baseline(seeds, n_episodes) → mean_return` — riferimento ≈ −20.4.
- `main()` — le 5 configurazioni + i grafici di ablazione.

### `make_report_tables.py`
- `_valid_rows(arr)`, `_final_perf(arr, tail_frac)` — helper (gestiscono il padding NaN).
- `table_control(...)`, `table_pong(...)` — generano il LaTeX.
- `main()` — scrive `report/tables/*.tex` dai `.npz`.

---

## Chi chiama chi (catena tipica)

```
scripts/run_cartpole.py :: exp_control_comparison()
   └── run_multi_seed(make_cartpole, q_learning, SEEDS, …)   [runner.py]
         ├── set_seed(seed)                                  [seeding.py]
         ├── make_cartpole() → DiscretizedEnv                [discretizer.py]
         └── q_learning(env, …)                              [q_learning.py]
               └── epsilon_greedy_action(Q, s, ε, n)         [policies.py]
   └── plot_learning_curves({...})                           [plotting.py]
```

Per il DQN la catena è analoga: `run_pong.py :: run_config` → `train_dqn`
(→ `ReplayBuffer`, `NatureCNN`, `_update_network`).

---

## File non-codice di riferimento

| File | Contenuto |
|---|---|
| `pyproject.toml` / `uv.lock` | dipendenze (vedi [`librerie.md`](librerie.md)) |
| `results/*/*.npz` | array grezzi (fonte di verità dei numeri) |
| `results/pong/*/*.pt` | checkpoint dei modelli DQN |
| `figures/*/*.png` | grafici generati |
| `report/` | report LaTeX + tabelle generate |
