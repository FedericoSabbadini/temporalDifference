# Fase 4 — Pong / DQN (Env 3, Deep RL)

## Obiettivo della fase

Implementare **DQN da zero in PyTorch** e applicarlo all'ambiente Atari Pong.
La consegna richiede:

1. DQN con **Experience Replay** + **Target Network** (implementazione completa)
2. **Ablation study**: confronto con/senza target network, con/senza replay
3. **NON è richiesta alta performance** — basta mostrare apprendimento vs random
4. **Bonus**: Double DQN (+3 punti, cap a 25)

Questa fase vale 4 punti (Deep RL) + fino a 3 punti bonus. È la parte più
tecnicamente nuova del progetto — il primo DQN scritto da zero.

## Collegamento con la teoria

### Perché DQN e non tabular? (Sutton & Barto §9, Mnih et al. 2015)

Pong ha osservazioni di 210×160×3 pixel. Anche discretizzando brutalmente,
lo spazio degli stati è astronomico. La Q-table non è un'opzione.

**DQN** (Deep Q-Network) sostituisce la Q-table con una **rete neurale**:
invece di Q[s][a] come entry di un dict, si ha Q(s, a; θ) dove θ sono i pesi
della rete. La rete prende lo stato (frame) in input e restituisce Q(s, a)
per tutte le azioni in output.

L'update diventa un **problema di regressione supervisionato**:
- **Target**: y = R + γ max_a' Q(S', a'; θ⁻)   (come Q-Learning, ma con la rete)
- **Loss**: L(θ) = (Q(S, A; θ) - y)²
- **Ottimizzazione**: gradient descent (Adam)

### Le due innovazioni chiave (Nature DQN, Mnih et al. 2015)

Il Q-Learning tabular converge perché la Q-table è aggiornata indipendentemente
per ogni (s, a). Con una rete neurale, un update su uno stato **influenza tutti
gli altri** (i pesi sono condivisi). Questo causa due problemi:

#### 1. Experience Replay

**Problema**: le transizioni (S, A, R, S') arrivano in sequenza temporale e sono
**correlate** (stati consecutivi sono simili). Il gradient descent assume
campioni i.i.d. → correlazione temporale rende l'apprendimento instabile.

**Soluzione**: salvare le transizioni in un **buffer** e campionare **mini-batch
casuali** per il training. Questo rompe le correlazioni temporali e permette
di riutilizzare ogni transizione più volte (efficienza dei dati).

È come avere un "quaderno di appunti" da cui si ripescano esperienze passate
in ordine casuale, invece di studiare solo l'ultima esperienza.

#### 2. Target Network

**Problema**: il target y = R + γ max_a' Q(S', a'; θ) usa la **stessa rete** che
stiamo aggiornando. Se aggiorniamo θ, il target cambia → "moving target problem".
È come sparare a un bersaglio che si muove ad ogni colpo.

**Soluzione**: usare una **copia separata** della rete (θ⁻) per calcolare il target.
Questa copia viene aggiornata solo periodicamente (ogni N step si fa θ⁻ ← θ).
Il target diventa stabile tra un aggiornamento e l'altro.

### Double DQN (van Hasselt et al. 2016) — bonus

Q-Learning standard soffre di **maximization bias** (Sutton & Barto §6.7):
il `max` nel target tende a sovrastimare i Q-values perché usa la stessa rete
sia per **scegliere** la best action sia per **valutarne** il Q-value.

Double DQN separa i due ruoli:
- La **rete online** (θ) sceglie l'azione: a* = argmax_a Q(S', a; θ)
- La **rete target** (θ⁻) valuta il Q: y = R + γ Q(S', a*; θ⁻)

In codice, sono ~3 righe di differenza rispetto al DQN standard.
Questo è lo stesso principio del **Double Q-Learning** tabular (Sutton & Barto §6.7),
applicato con reti neurali.

### Preprocessing Atari (Mnih et al. 2015, §Methods)

I frame raw di Pong (210×160×3, RGB) sono troppo grandi e ridondanti. Il
preprocessing standard:

1. **Grayscale** (3 canali → 1): colore irrilevante per Pong
2. **Resize 84×84**: riduce la dimensionalità
3. **Frame skip 4**: ripete ogni azione per 4 frame (efficienza + velocità)
4. **Stack 4 frame**: la rete vede 4 frame consecutivi → può percepire il **movimento**
   della palla (un singolo frame è ambiguo sulla direzione)
5. **Max over last 2 frames**: gestisce il flickering dei giochi Atari

## File coinvolti

| File | Ruolo |
|---|---|
| `rl_lab/algorithms/deep/networks.py` | `NatureCNN` (architettura convoluzionale) e `MLP` (per debug su CartPole) |
| `rl_lab/algorithms/deep/replay_buffer.py` | Buffer circolare con campionamento uniforme |
| `rl_lab/algorithms/deep/dqn.py` | Training loop completo: ε-decay, update, target sync, ablation flags |
| `rl_lab/envs/atari_wrappers.py` | Preprocessing Atari (grayscale, resize, frame stack) |
| `scripts/run_pong.py` | Orchestratore: 5 configurazioni × 3 seed + random baseline |

## Flusso del codice

```
run_pong.py
│
├── random_baseline()
│   50 episodi con azioni casuali → return medio (≈ -21)
│   Serve come riferimento: DQN deve fare meglio di questo
│
├── run_config("Full DQN", use_target=True, use_replay=True)
│   Per ogni seed:
│     1. make_pong() → crea env con preprocessing
│     2. train_dqn() → training loop (200k step)
│        - Epsilon decay lineare: 1.0 → 0.02 in 100k step
│        - Ogni step: salva transizione nel buffer
│        - Se buffer abbastanza pieno: campiona batch, gradient step
│        - Ogni 1000 step: copia pesi nella target net
│     3. Salva modello (.pt)
│   Ripete per seed 42, 123, 456
│
├── run_config("No target net", ...)      ← ablazione
├── run_config("No replay", ...)          ← ablazione
├── run_config("No target + no replay")   ← ablazione
├── run_config("Double DQN", ...)         ← bonus
│
└── Genera plot:
    ablation_replay.png     — con vs senza replay
    ablation_target.png     — con vs senza target net
    full_comparison.png     — tutte e 5 le configurazioni
```

### Dentro `train_dqn()` — il training loop step by step

```python
while step < total_steps:
    state = env.reset()

    while not done:
        # 1. Scegli azione (ε-greedy sulla rete online)
        if random < epsilon:
            action = random
        else:
            action = argmax Q(state; θ)    # forward pass sulla rete

        # 2. Esegui azione nell'ambiente
        next_state, reward, done = env.step(action)

        # 3. Salva transizione nel replay buffer
        buffer.push(state, action, reward, next_state, done)

        # 4. Se il buffer ha abbastanza transizioni:
        #    campiona mini-batch casuale e fai un gradient step
        if len(buffer) >= learning_starts:
            batch = buffer.sample(32)
            loss = MSE(Q(s,a;θ) - [r + γ max Q(s',a';θ⁻)])
            optimizer.step()

        # 5. Periodicamente: aggiorna la target network
        if step % 1000 == 0:
            θ⁻ ← θ    (hard update = copia completa)

        step += 1
```

## Scelte implementative

- **200.000 step totali**: compromesso tra tempo (≈50 min su RTX 5060) e
  capacità di mostrare apprendimento. Il paper originale usa 50M step, ma
  la consegna dice che bastano segnali di apprendimento.

- **Buffer 100.000 transizioni**: con frame uint8 (84×84×4 byte per transizione ≈ 28KB),
  100k transizioni ≈ 2.7 GB. Sta comodamente in 32 GB di RAM.
  Il paper originale usa 1M, ma con 32 GB di RAM non ci sta tutto.

- **learning_starts = 10.000**: si riempie il buffer con 10k transizioni casuali
  prima di iniziare a fare gradient step. Questo dà un campione iniziale diversificato.

- **ε-decay lineare** da 1.0 a 0.02 in 100k step: all'inizio esplorazione totale,
  poi gradualmente si sfrutta la rete. Il paper originale fa decay in 1M step;
  qui è proporzionalmente adattato ai 200k step totali.

- **Adam con lr = 1e-4**: ottimizzatore standard per deep learning. Il paper
  originale usa RMSProp, ma Adam è il default moderno.

- **Hard update ogni 1000 step** (non soft/Polyak update): il paper originale
  fa hard update ogni 10k step. Qui 1000 è proporzionale ai 200k step totali.

- **NatureCNN automatica vs MLP**: `train_dqn` sceglie la rete in base alla
  forma dell'osservazione. Se `obs.ndim == 1` → MLP (per debug su CartPole).
  Se `obs.ndim >= 3` → NatureCNN (per Atari).

- **senza replay: batch_size=1 equivalente**: quando `use_replay=False`,
  il buffer ha capacità batch_size e si "campiona" la transizione appena inserita.
  È equivalente a fare online learning (un update per transizione).

## Cosa è ripreso e cosa è nuovo

| Componente | Origine |
|---|---|
| `networks.py` — `NatureCNN` | **Nuovo**, ma l'architettura replica il paper (Mnih et al. 2015, Table 1). I numeri (32@8×8/4, 64@4×4/2, 64@3×3/1, FC 512) sono esattamente quelli del paper. |
| `networks.py` — `MLP` | **Nuovo.** Rete semplice per debug su CartPole prima di passare ad Atari. |
| `replay_buffer.py` | **Nuovo.** Implementazione minimale con `deque`. Il concetto è dal paper; lo stile (store uint8, sample con numpy) è standard. |
| `dqn.py` — `train_dqn` | **Nuovo.** È la parte più complessa e completamente originale del progetto. Il loop è scritto da zero seguendo il paper. |
| `dqn.py` — Double DQN | **Nuovo.** ~3 righe di codice aggiuntive: `best_actions = online_net(next_states).argmax()`, poi `target_net.gather(best_actions)`. |
| `atari_wrappers.py` | **Adattato** dal `Tutorial_Gymnasium.json` del professore (che mostra come creare env Atari). I wrapper `AtariPreprocessing` e `FrameStack` sono di Gymnasium (non SB3). |
| `run_pong.py` | **Nuovo.** Design dell'ablation study e gestione multi-seed con padding. |

### Nota importante: cosa NON viene da SB3

La consegna vieta SB3 per Env 1 e 2 ma lo consente come **baseline opzionale** per
Atari. Tutto il codice DQN in questo progetto è scritto da zero:
- Il training loop (`train_dqn`) non usa SB3
- La rete (`NatureCNN`) non usa modelli SB3
- Il replay buffer è implementato a mano
- Solo i **wrapper di preprocessing** (AtariPreprocessing, FrameStack) vengono da
  Gymnasium (non da SB3)

## Come leggere i risultati

### ablation_replay.png
- **Full DQN** (con replay) vs **No replay**
- Senza replay il training è instabile: le transizioni sono correlate, la rete
  "dimentica" le esperienze passate. Ci si aspetta curve molto più rumorose e
  apprendimento peggiore senza replay.

### ablation_target.png
- **Full DQN** (con target net) vs **No target net**
- Senza target net il target è un "moving target" che cambia ad ogni gradient step.
  Ci si aspetta più instabilità e oscillazioni senza target net.

### full_comparison.png
- Tutte e 5 le configurazioni su un grafico + indicazione del baseline random.
- **Ordine atteso** (dal migliore al peggiore):
  1. Double DQN (o Full DQN)
  2. Full DQN
  3. No target net (instabile ma impara qualcosa)
  4. No replay (molto instabile)
  5. No target + no replay (il peggiore)
- Il **random baseline** (≈ -21) è la linea di riferimento. Qualsiasi
  configurazione che fa meglio di -21 "ha imparato qualcosa".

### Nota sulle performance attese

Con solo 200k step, non ci si aspetta che DQN impari a giocare bene a Pong.
Il paper originale usa 50M step. L'obiettivo è mostrare un **trend positivo**:
le curve dovrebbero salire rispetto al random, anche se non raggiungono +21
(vittoria perfetta). Un return di -15 / -10 dopo 200k step è già un buon segnale.

## Limiti e semplificazioni

- **200k step è pochissimo** per Atari. Il paper usa 50M. Il risultato sarà
  parziale, ma sufficiente per mostrare apprendimento. Se c'è tempo, aumentare
  a 500k–1M step migliorerebbe visibilmente i risultati.

- **Hard update** della target net (copia completa) è meno smooth del **soft
  update** (Polyak: θ⁻ ← τθ + (1-τ)θ⁻). Il soft update è più stabile ma
  il paper originale usa hard update.

- **Buffer in RAM** con `deque` di tuple: funziona ma non è il più efficiente.
  Un buffer basato su array pre-allocati sarebbe più veloce. Per 100k transizioni
  la differenza è trascurabile.

- **Nessun reward clipping**: il paper originale clippa i reward a [-1, 1].
  Pong ha già reward in {-1, 0, +1} quindi non serve, ma per altri giochi
  Atari sarebbe necessario.

- **Nessun gradient clipping**: il paper originale clippa i gradienti. Per
  200k step probabilmente non è critico, ma potrebbe aiutare la stabilità.

- **`terminal_on_life_loss=False`**: il paper originale tratta la perdita di
  una vita come terminazione dell'episodio. Qui non lo facciamo per semplicità.
  Potrebbe rallentare l'apprendimento perché l'agente non "sente" la penalità
  della perdita di una vita immediatamente.
