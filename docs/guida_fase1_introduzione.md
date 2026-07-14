# Fase 1 - Introduzione pratica e infrastruttura

> ### Percorso di lettura
> - **Prerequisiti**: [`panoramica_10_minuti.md`](panoramica_10_minuti.md). Teoria: nozione di MDP, policy, ε-greedy (già dal corso).
> - **Guida precedente**: nessuna - è la prima. **Successiva**: [Fase 2](guida_fase2_cartpole.md).
> - **Tempo stimato**: ~12 minuti (guida) + sfogliare i file.
> - **Apri in parallelo** (in quest'ordine): `rl_lab/configs.py` - `rl_lab/envs/discretizer.py` - `rl_lab/policies.py` - `rl_lab/experiments/runner.py` - `rl_lab/utils/{seeding,plotting}.py`.
> - **Requisito consegna coperto**: nessuno *direttamente*. Questa fase costruisce l'infrastruttura che *abilita* tutti i requisiti delle fasi 2-4 (discretizzazione, riproducibilità multi-seed, plotting con bande std).

## Obiettivo della fase

Costruire le fondamenta del progetto: l'infrastruttura che permette di eseguire
algoritmi tabular RL su ambienti Gymnasium con stato continuo.
Questa fase non produce risultati sperimentali: prepara il terreno per le fasi 2-4.

I componenti principali sono:
- **Discretizzazione** dello spazio degli stati (da continuo a discreto)
- **Policy di selezione delle azioni** (greedy, ε-greedy)
- **Runner multi-seed** per esperimenti riproducibili
- **Utilità** per plotting e seeding

## Collegamento con la teoria

| Concetto teorico (Sutton & Barto) | Dove nel codice |
|---|---|
| **MDP con stati continui** - la Q-table richiede stati finiti e hashable | `envs/discretizer.py` |
| **Policy ε-greedy** - bilancia exploration (azione casuale con prob. ε) ed exploitation (azione greedy) | `policies.py` |
| **Greedy policy con tie-breaking** - π(s) = argmax_a Q(s,a), con scelta casuale tra i massimi | `policies.py` |
| **Riproducibilità** - fissare il seed per numpy, torch e l'env | `utils/seeding.py` |
| **Media su più seed + deviazione standard** - requisito sperimentale della consegna | `experiments/runner.py` |
| **Learning curve** - reward vs episodi, smoothed con moving average | `utils/plotting.py` |

### Perché serve la discretizzazione?

I metodi tabular (Q-Learning, SARSA, MC, TD) usano una **Q-table** o **V-table**
indicizzata per stato. Se lo stato è un vettore continuo ℝ^n (come in CartPole o
Acrobot), servono infiniti stati - impossibile. La soluzione è **binning**: dividere
ogni dimensione in intervalli uniformi e mappare ogni osservazione continua in un
indice di bin. Il risultato è una **tupla di interi** usabile come chiave di un dict.

Questo corrisponde al concetto di **state aggregation** (Sutton & Barto §9.4):
raggruppare stati simili sotto lo stesso valore stimato. Il trade-off è:
- **Troppi pochi bin** - si perde informazione, policy subottimale
- **Troppi bin** - spazio degli stati esplode (curse of dimensionality), servono
  troppe esperienze per convergere

Questo trade-off è esattamente il tema dell'esperimento `exp_bin_resolution` in Fase 3.

### Perché ε-greedy e non una policy più sofisticata?

L'ε-greedy è il metodo di esplorazione più semplice e quello trattato nel corso.
Con probabilità ε si sceglie un'azione casuale (exploration), altrimenti si sceglie
l'azione con Q massimo (exploitation). È sufficiente per gli ambienti del progetto.
Metodi più avanzati (UCB, softmax, etc.) non sono richiesti dalla consegna.

## File coinvolti

| File | Ruolo |
|---|---|
| `rl_lab/configs.py` | Configurazioni: nome dell'env, numero di bin, range di clipping per dimensione |
| `rl_lab/policies.py` | Funzioni `greedy_action` e `epsilon_greedy_action` |
| `rl_lab/envs/discretizer.py` | Wrapper Gymnasium che converte osservazioni continue in tuple discrete |
| `rl_lab/utils/seeding.py` | `set_seed(seed)` per numpy + torch |
| `rl_lab/utils/plotting.py` | `plot_learning_curves` (media + banda std), `plot_sensitivity`, `moving_average` |
| `rl_lab/experiments/runner.py` | `run_multi_seed`: esegue un algoritmo N volte con seed diversi |

## Flusso del codice

```
1. configs.py definisce i parametri dell'ambiente (es. CARTPOLE, ACROBOT)
       |
2. make_env() crea l'env Gymnasium e lo wrappa con DiscretizedEnv
       |
3. DiscretizedEnv.observation() converte ogni osservazione:
   [cart_pos, cart_vel, angle, ang_vel] - (bin_0, bin_1, bin_2, bin_3)
       |
4. L'algoritmo (q_learning, sarsa, ...) usa la tupla come chiave del dict Q
       |
5. runner.py ripete il tutto per 3 seed diversi
       |
6. plotting.py calcola media e std e genera i grafici
```

## Scelte implementative

- **Dict Python come Q-table** (`defaultdict(lambda: np.zeros(n_actions))` = { (1,2,3,4): array([0.,0.])} for state=(1,2,3,4) q(s,a)=[0,0]):
  scelta del corso. Efficiente per spazi sparsi - alloca solo gli stati visitati.
  Un array NumPy 2D sarebbe più veloce ma richiederebbe numerare tutti gli stati
  a priori, complicando il codice.

- **Clip + bin uniforme** per la discretizzazione: il metodo più semplice.
  Alternative come tile coding (Sutton & Barto §9.5.4) darebbero risultati migliori
  ma sono più complessi e non richiesti.

- **Osservazione come tupla** (non come array): le tuple sono hashable, gli array
  NumPy no. Serve una tupla per usarla come chiave di un dict.

- **Configurazioni in Python** (non YAML): evita una dipendenza esterna (pyyaml).
  I dict Python sono ugualmente leggibili per un progetto di questa scala.

## Cosa è ripreso e cosa è nuovo

| Componente | Origine |
|---|---|
| `policies.py` - `greedy_action`, `epsilon_greedy_action` | **Adattato** dallo stile dell'Esercitazione_RL del corso. La logica è la stessa (ε-greedy con tie-breaking casuale), adattata per funzionare con `defaultdict` Q-table. |
| `envs/discretizer.py` | **Nuovo.** Nelle esercitazioni del corso gli ambienti erano già discreti (gridworld). Il wrapper è necessario perché CartPole e Acrobot hanno stati continui. |
| `utils/seeding.py` | **Nuovo**, ma il pattern (`np.random.seed + torch.manual_seed`) è standard. |
| `utils/plotting.py` | **Adattato** dalla funzione `moving_average` usata negli esercizi del corso. `plot_learning_curves` con bande std è nuovo. |
| `experiments/runner.py` | **Adattato** dal pattern `run_bias_experiment` dell'Esercitazione_RL (100 run con seed diversi). Generalizzato per funzionare con qualsiasi algoritmo. |
| `configs.py` | **Nuovo.** Centralizza i parametri che prima erano sparsi nei notebook. |

## Come leggere i risultati

Questa fase non produce grafici. Il suo output è verificabile importando il package:

```python
from rl_lab.configs import CARTPOLE
from rl_lab.envs.discretizer import DiscretizedEnv
# Se l'import funziona - l'infrastruttura è pronta.
```

## Modello mentale (cosa succede davvero)

Immagina un singolo esperimento come una catena di montaggio:

1. `configs.py` fornisce i **parametri** (nome env, bin, range). Sono dati inerti.
2. Una `make_env()` (definita negli script) crea l'ambiente Gymnasium e lo avvolge
   in `DiscretizedEnv`. Da qui in poi, ogni `env.step()` restituisce uno **stato
   già discretizzato** (una tupla), non più un vettore continuo.
3. `run_multi_seed` esegue l'algoritmo **3 volte** (un seed alla volta), ogni volta
   con un ambiente fresco. Raccoglie i return in una matrice `(3, n_episodi)`.
4. `plotting.py` collassa quella matrice in **media ± banda** e disegna.

Cosa "vive" tra un pezzo e l'altro:
- Lo **stato** è effimero: nasce nell'env, viene discretizzato, usato come chiave,
  e dimenticato.
- La **Q-table** (nelle fasi 2-4) è ciò che *persiste e cresce* durante una run.
- Il **seed** è ciò che rende la catena ripetibile: stessi seed - stessi numeri.

Questa fase non "impara" nulla: costruisce solo la catena di montaggio. L'unico
output verificabile è che il package si importi (vedi sotto).

## Debugging e trappole comuni

| Sintomo | Causa probabile | Dove guardare |
|---|---|---|
| `TypeError: unhashable type: 'numpy.ndarray'` usando lo stato come chiave | Hai dimenticato di convertire l'osservazione in **tupla** | `discretizer.py` deve restituire `tuple(bins)`, non l'array |
| L'agente non impara nulla anche con codice corretto | Range di clipping sbagliato: gli stati utili finiscono tutti nello stesso bin | `configs.py` (`clip_low/high`); verifica che coprano il range *visitato*, non oltre |
| Risultati diversi ad ogni esecuzione | `set_seed` non chiamato prima della run | `runner.py` chiama `set_seed(seed)` prima di ogni env |
| Le bande std sembrano assenti | Un solo seed, o array 1-D | `plotting.py` fa `reshape(1,-1)`; servono ≥2 seed per una banda vera |
| Grafico vuoto / crash matplotlib | Backend GUI su macchina headless | `plotting.py` forza `matplotlib.use("Agg")` |

**Come si ispeziona a mano**: per capire cosa produce il discretizzatore,
`env.reset()` e stampa la tupla; deve avere lunghezza = numero di dimensioni e
ogni indice in `[0, n_bins-1]`.

## Limiti e semplificazioni

- **Binning uniforme** perde informazione rispetto a metodi adattivi (tile coding,
  radial basis functions). È sufficiente per CartPole; per Acrobot il binning
  grossolano diventa un punto di analisi (Fase 3).
- **ε costante** durante il training: non c'è decay. Per i tabular methods del
  progetto è sufficiente. Il DQN (Fase 4) usa invece un ε con decay lineare.
- **Nessun logging avanzato** (MLflow, W&B): non necessario alla scala del progetto.
