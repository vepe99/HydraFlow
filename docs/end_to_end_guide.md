# HydraBFlow — End-to-End Pipeline Guide

This is a hands-on walkthrough for standing up a complete Simulation-Based Inference (SBI)
pipeline with HydraBFlow. We thread one concrete example — a trivial 2-parameter **Gaussian**
simulator — through every section so the commands are runnable, not abstract.

By the end you will know how to:

1. [Write/change the simulator](#2-changing-the-simulator)
2. [Drive everything through `conf/`](#3-using-conf)
3. [Add a SummaryNetwork that isn't shipped](#4-adding-a-summarynetwork-that-isnt-shipped)
4. [Add an InferenceNetwork that isn't shipped](#5-adding-an-inferencenetwork-that-isnt-shipped)
5. [Set up hyperparameter tuning](#6-hyperparameter-tuning)
6. [Change which values the tuner studies](#7-changing-the-values-studied)

> **Mental model.** You write a *simulator* and pick/configure *SBI components*. Everything
> else — config composition, training loop, checkpointing, diagnostics, traceability — is fixed
> infrastructure you never touch. The two extension styles are: **self-registering** components
> (simulators) that need *no* infrastructure edits, and **factory-backed** components (summary /
> inference networks) where you add one branch to a factory plus a typed config field.

---

## 0. Prerequisites & install

```bash
uv sync                      # resolves bayesflow, keras, jax, hydra, optuna, ...
uv run python -c "import hydrabflow"   # smoke test; pins KERAS_BACKEND=jax on import
```

The JAX backend is pinned for you in [src/hydrabflow/utils/backend.py](../src/hydrabflow/utils/backend.py)
(imported first by [src/hydrabflow/__init__.py](../src/hydrabflow/__init__.py)) via
`os.environ.setdefault("KERAS_BACKEND", "jax")`. Override with `KERAS_BACKEND=tensorflow uv run ...`
if you must, but JAX is the supported default.

There are two equivalent ways to launch each stage — pick one and stay consistent:

| Stage          | Console script             | Script file                                      |
| -------------- | -------------------------- | ------------------------------------------------ |
| simulate       | `hydrabflow-simulate`       | `uv run python scripts/simulate.py`              |
| train          | `hydrabflow-train`          | `uv run python scripts/train.py`                 |
| evaluate       | `hydrabflow-evaluate`       | `uv run python scripts/evaluate.py`              |
| evaluate-real  | `hydrabflow-evaluate-real`  | `uv run python scripts/evaluate_real.py`         |
| tune           | `hydrabflow-tune`           | `uv run python scripts/tune.py`                  |

Both are thin [Hydra](https://hydra.cc) apps over `hydrabflow.pipeline.<stage>`. This guide uses the
`uv run python scripts/...` form.

---

## 1. The five stages at a glance

```
simulate ─▶ train ─▶ evaluate          (simulated test set, has ground truth)
                 └──▶ evaluate_real     (your observed data, no ground truth)
                 └──▶ tune              (Optuna search over hyperparameters)
```

Every run writes a fully traceable, timestamped output directory (set in
[conf/config.yaml](../conf/config.yaml)):

```
outputs/${simulator.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}/
├── .hydra/                  # full resolved config (Hydra writes this automatically)
├── approximator.keras       # (train) the trained model
├── preprocessing_state.npz  # (train) fitted preprocessing — reused at eval time
├── loss.png                 # (train)
├── posterior.npz            # (evaluate / evaluate_real)
├── metrics.json + *.png     # (evaluate) RMSE/calibration + diagnostic plots
└── best_trials.json         # (tune)
```

A run is valid only if it can be reconstructed from its folder — that is why `evaluate` and
`evaluate_real` reload the model **and** the fitted preprocessing from a `model_dir` you point at a
completed `train` run.

---

## 2. Changing the simulator

The simulator is the **only** Python you must write. The shipped
[SkeletonSimulator](../src/hydrabflow/simulators/skeleton.py) declares a 2-parameter /
single-observable interface so the whole pipeline composes, but every forward call raises
`NotImplementedError`. We will replace it with a real one.

### 2a. Write the simulator class

A simulator subclasses [BaseSimulator](../src/hydrabflow/simulators/base.py) and implements four
things. The shape contract (leading axis = number of simulations `n`):

- `sample_prior(n, rng)` → `{param_name: (n, 1)}`
- `simulate(params, rng)` → `{observable_key: (n, *event_shape)}`

Create **`src/hydrabflow/simulators/gaussian.py`**:

```python
"""A tiny worked-example simulator: infer the mean of a 2-D Gaussian from a set of draws."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from hydrabflow.simulators.base import BaseSimulator
from hydrabflow.simulators.registry import register_simulator


@register_simulator("gaussian")          # <-- self-registers under this name
class GaussianSimulator(BaseSimulator):
    @property
    def parameter_names(self) -> list[str]:
        # These become the inference targets (adapter.inference_variables).
        return ["mu1", "mu2"]

    @property
    def observable_keys(self) -> list[str]:
        # One key = single observable, fed to the summary network.
        return ["x"]

    def sample_prior(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        # `self.params` is the free-form mapping from conf/simulator/gaussian.yaml.
        lo, hi = float(self.params.get("prior_low", -3.0)), float(self.params.get("prior_high", 3.0))
        return {
            "mu1": rng.uniform(lo, hi, size=(n, 1)),
            "mu2": rng.uniform(lo, hi, size=(n, 1)),
        }

    def simulate(
        self, params: Mapping[str, np.ndarray], rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        set_size = int(self.params.get("set_size", 10))
        noise = float(self.params.get("noise", 1.0))
        mu = np.concatenate([params["mu1"], params["mu2"]], axis=-1)   # (n, 2)
        # Set-shaped observable (n, set_size, 2): a cloud of draws around mu.
        # SetTransformer (the default summary net) expects exactly this 3-D shape.
        eps = rng.normal(0.0, noise, size=(mu.shape[0], set_size, 2))
        return {"x": mu[:, None, :] + eps}
```

> The base class already provides `sample(n, rng)`, which merges `sample_prior` + `simulate` into
> one dataset chunk — infrastructure calls it, you normally don't override it.

### 2b. Make it self-register

Components only register when their module is imported. Add the import to
[src/hydrabflow/simulators/\_\_init\_\_.py](../src/hydrabflow/simulators/__init__.py):

```python
from hydrabflow.simulators import skeleton   # noqa: F401  (existing)
from hydrabflow.simulators import gaussian    # noqa: F401  (self-registers "gaussian")  <-- add
```

Skip this and you get a clear error from `get_simulator`:
`Unknown simulator 'gaussian'. Registered: [...]. Did you import the module that defines it?`

### 2c. Add the simulator config

Create **`conf/simulator/gaussian.yaml`** (mirror
[conf/simulator/skeleton.yaml](../conf/simulator/skeleton.yaml)):

```yaml
defaults:
  - base_simulator

name: gaussian          # must match @register_simulator("gaussian")
params:                 # passed verbatim to the constructor as self.params
  prior_low: -3.0
  prior_high: 3.0
  set_size: 10
  noise: 1.0
```

`params` is a free-form mapping — declare whatever your forward model needs (particle counts,
integrator settings, prior bounds) without ever touching the schema.

### 2d. Wire the adapter to your parameter / observable names

The adapter maps raw dataset keys to BayesFlow roles. Edit
[conf/adapter/default.yaml](../conf/adapter/default.yaml):

```yaml
defaults:
  - base_adapter

inference_variables: [mu1, mu2]   # == GaussianSimulator.parameter_names  (MANDATORY)
summary_variables: [x]            # == GaussianSimulator.observable_keys
inference_conditions: []          # direct, non-summarized conditions (rarely needed)
drop: []                          # keys to discard
```

> `inference_variables` ships as `???` (Hydra's "mandatory value"). Until you set it — in this file
> or on the CLI (`adapter.inference_variables=[mu1,mu2]`) — every stage fails fast with
> `MissingMandatoryValue`. This is intentional: the targets are simulator-specific.

### 2e. Shape contract cheat-sheet

| Summary network          | Expected observable shape        |
| ------------------------ | -------------------------------- |
| `set_transformer`        | `(n, set_size, features)` (unordered set) |
| `deep_set`               | `(n, set_size, features)`        |
| `time_series_transformer`| `(n, time_steps, features)` (ordered) |

Our Gaussian emits `(n, 10, 2)`, which matches the default `set_transformer`.

### 2f. Generate the dataset

```bash
uv run python scripts/simulate.py simulator=gaussian
```

This samples the prior, runs the forward model in `chunk_size` batches, and writes one aggregated
`.npz` to `data/gaussian/training_data_10000.npz` (path = `${data.data_dir}/${data.dataset_name}`,
see [conf/data/default.yaml](../conf/data/default.yaml)).

---

## 3. Using `conf/`

HydraBFlow is **Hydra-native**: there is no argparse. The root file
[conf/config.yaml](../conf/config.yaml) composes exactly one choice from every config *group*:

```yaml
defaults:
  - base_config            # the structured (dataclass) schema — validates everything
  - simulator: skeleton    # <- pick a file from conf/simulator/
  - model: default         # <- conf/model/default.yaml (itself composes sub-groups)
  - data: default
  - training: default
  - preprocessing: default
  - augmentation: default
  - adapter: default
  - inference: default
  - eval: default
  - tuning: default
  - _self_                 # last, so values in config.yaml win
```

Each group is a folder under `conf/`; each `.yaml` in it is one selectable option. Groups can nest:
[conf/model/default.yaml](../conf/model/default.yaml) further composes
`summary_network: set_transformer` and `inference_network: flow_matching`.

Every group YAML starts with `defaults: [- base_<group>]`. That `base_<group>` is the **typed
dataclass schema** from [src/hydrabflow/config/schema.py](../src/hydrabflow/config/schema.py),
registered in Hydra's `ConfigStore`. It is what catches typos and wrong types before training
starts — set a string where an int is expected and composition fails immediately.

### Three ways to set a value

**(a) Edit the group YAML** — the permanent default for your project (what we did for the adapter).

**(b) Override a single value on the CLI** — dotted path, great for experiments:

```bash
uv run python scripts/train.py simulator=gaussian \
    model.summary_network.summary_dim=64 \
    training.n_epochs=100 training.batch_size=256
```

**(c) Swap a whole group** — `group=option`, e.g. switch the inference network:

```bash
uv run python scripts/train.py simulator=gaussian model/inference_network=diffusion
```

Note the slash: nested groups use `model/inference_network=...`; plain keys use dots.

### Run the rest of the pipeline

```bash
# Train (writes outputs/gaussian/default/<timestamp>/)
uv run python scripts/train.py simulator=gaussian

# Make a held-out test set (different name + seed so it isn't the training data)
uv run python scripts/simulate.py simulator=gaussian \
    data.dataset_name=test_data_10000.npz seed=123

# Evaluate: point model_dir at the completed train run
uv run python scripts/evaluate.py simulator=gaussian \
    model_dir=outputs/gaussian/default/<timestamp>
```

`evaluate` looks for `${data.data_dir}/${eval.test_dataset_name}` (default
`test_data_${data.n_simulations}.npz`, see [conf/eval/default.yaml](../conf/eval/default.yaml)),
reloads the model + fitted `preprocessing_state.npz` from `model_dir`, samples the posterior, and
writes `metrics.json` plus recovery / calibration-ecdf / z-score-contraction plots.

For real observations, set `data.real_data_path` and run `scripts/evaluate_real.py` with the same
`model_dir` — no ground truth, no resimulation, just posterior pair plots.

---

## 4. Adding a SummaryNetwork that isn't shipped

Summary and inference networks are **factory-backed**, not registry-backed: the builder is an
explicit `if/elif` on `cfg.type`. Adding one is a focused 3-step change — one factory branch, one
(optional) schema edit, one YAML.

Worked example: a recurrent **GRU** summary network for ordered/time-series observables.

### Step 1 — add a branch to the factory

Edit [`build_summary_network`](../src/hydrabflow/networks/factory.py):

```python
    if t == "deep_set":
        return bf.networks.DeepSet(
            summary_dim=int(cfg.summary_dim),
            dropout=float(cfg.dropout),
        )
    # --- new branch ---
    if t == "gru":
        return bf.networks.RecurrentNetwork(          # check the exact class in your bayesflow
            summary_dim=int(cfg.summary_dim),
            cell_type="gru",
            num_layers=int(cfg.num_blocks),           # reuse an existing field...
            hidden_dim=int(cfg.embed_dim),
        )
    raise ValueError(
        f"Unknown summary_network.type '{t}'. "
        "Expected one of: set_transformer, time_series_transformer, deep_set, gru."
    )
```

> Check your installed BayesFlow for the exact summary-network class and kwargs
> (`python -c "import bayesflow as bf; print(dir(bf.networks))"`). The pattern is what matters:
> read scalar fields off `cfg` and pass them to the constructor.

### Step 2 — (only if you need new hyperparameters) extend the schema

`SummaryNetworkConfig` in [src/hydrabflow/config/schema.py](../src/hydrabflow/config/schema.py)
already exposes `summary_dim, num_blocks, num_heads, embed_dim, mlp_depth, mlp_width, dropout`. If
your network needs a knob none of those cover, add a typed field with a default:

```python
@dataclass
class SummaryNetworkConfig:
    type: str = "set_transformer"
    summary_dim: int = 32
    num_blocks: int = 2
    num_heads: int = 4
    embed_dim: int = 64
    mlp_depth: int = 2
    mlp_width: int = 128
    dropout: float = 0.05
    bidirectional: bool = False     # <-- new optional field, consumed by the "gru" branch
```

Reusing existing fields (as the GRU branch above does) means you can skip this step entirely.

### Step 3 — add a selectable config file

Create **`conf/model/summary_network/gru.yaml`** (mirror
[set_transformer.yaml](../conf/model/summary_network/set_transformer.yaml)):

```yaml
defaults:
  - base_summary_network

type: gru          # must match the factory branch
summary_dim: 32
num_blocks: 2      # -> num_layers
embed_dim: 64      # -> hidden_dim
dropout: 0.05
```

### Use it

```bash
uv run python scripts/train.py simulator=gaussian model/summary_network=gru
```

---

## 5. Adding an InferenceNetwork that isn't shipped

Identical 3-step pattern, against the inference side. Shipped types are `flow_matching` and
`diffusion`. Example: add a **coupling-flow** posterior network.

### Step 1 — branch in the factory

Edit [`build_inference_network`](../src/hydrabflow/networks/factory.py):

```python
    if t == "diffusion":
        return bf.networks.DiffusionModel(
            subnet_kwargs={"widths": widths, "time_embedding_dim": int(cfg.time_embedding_dim)},
        )
    # --- new branch ---
    if t == "coupling_flow":
        return bf.networks.CouplingFlow(
            subnet_kwargs={"widths": widths, "dropout": float(cfg.dropout)},
        )
    raise ValueError(
        f"Unknown inference_network.type '{t}'. "
        "Expected one of: flow_matching, diffusion, coupling_flow."
    )
```

`widths` is already derived in the factory as `[mlp_width] * mlp_depth`.

### Step 2 — extend `InferenceNetworkConfig` if needed

Fields available out of the box: `mlp_depth, mlp_width, dropout, time_embedding_dim`. Add a typed
field with a default only if your network needs something new (same pattern as §4 step 2).

### Step 3 — selectable config

Create **`conf/model/inference_network/coupling_flow.yaml`** (mirror
[flow_matching.yaml](../conf/model/inference_network/flow_matching.yaml)):

```yaml
defaults:
  - base_inference_network

type: coupling_flow
mlp_depth: 4
mlp_width: 128
dropout: 0.05
```

### Use it

```bash
uv run python scripts/train.py simulator=gaussian model/inference_network=coupling_flow
```

---

## 6. Hyperparameter tuning

Tuning is an [Optuna](https://optuna.org) study driven entirely by
[conf/tuning/default.yaml](../conf/tuning/default.yaml). It is **multi-objective by default**:
minimize both validation RMSE and calibration error.

```bash
uv run python scripts/tune.py simulator=gaussian
```

What happens (see [src/hydrabflow/pipeline/tune.py](../src/hydrabflow/pipeline/tune.py)):

1. The dataset is loaded and preprocessed **once** (train/val split reused across all trials).
2. Each trial deep-copies the config and applies its sampled hyperparameters with
   `OmegaConf.update(cfg, path, value)` for every entry in `search_space`.
3. A fresh workflow trains for a short budget (`tuning.n_epochs`, default 10), then scores the
   validation split with `bayesflow.diagnostics.metrics` (`root_mean_squared_error`,
   `calibration_error`).
4. The objective returns `(rmse_mean, cal_mean)` for multi-objective, or just `rmse_mean` if
   `directions` has one entry.

Knobs in the config:

```yaml
study_name: hydrabflow_study
storage_dir: ${data.data_dir}/tuning      # sqlite db lives here: <storage_dir>/<study_name>.db
n_trials: 50
directions: [minimize, minimize]          # two objectives -> Pareto front
n_epochs: 10                              # short budget PER trial (not the full training run)
```

The study is persisted to SQLite and opened with `load_if_exists=True`, so **re-running the same
command resumes** and adds more trials. Results land in `best_trials.json` in the run dir — the
Pareto-optimal trials (or the single best) with their parameter values.

One-off overrides for a quick smoke run:

```bash
uv run python scripts/tune.py simulator=gaussian tuning.n_trials=20 tuning.n_epochs=5
```

---

## 7. Changing the values studied

The search space is a mapping from a **dotted config path** (anything in the composed config) to a
**sampling spec**. Edit the `search_space` block of
[conf/tuning/default.yaml](../conf/tuning/default.yaml):

```yaml
search_space:
  model.summary_network.summary_dim:      # any config path the objective can update
    type: int
    low: 16
    high: 64
  model.summary_network.embed_dim:
    type: int
    low: 32
    high: 128
    step: 16                              # int/float support `step`
  model.inference_network.mlp_width:
    type: int
    low: 32
    high: 256
    step: 16
  training.learning_rate:
    type: float
    low: 1e-4
    high: 1e-2
    log: true                            # log-uniform sampling
```

### Spec reference (consumed by `_suggest` in `tune.py`)

| `type`        | Keys used                          | Optuna call             |
| ------------- | ---------------------------------- | ----------------------- |
| `int`         | `low`, `high`, `step` (default 1)  | `suggest_int`           |
| `float`       | `low`, `high`, `step?`, `log?`     | `suggest_float`         |
| `categorical` | `choices: [...]`                   | `suggest_categorical`   |

Example categorical entry:

```yaml
  model.inference_network.type:
    type: categorical
    choices: [flow_matching, diffusion]
```

### Common edits

- **Add a parameter:** add an entry whose key is any valid config path, e.g.
  `training.batch_size`, `augmentation.params.noise_std`, or a new summary-net field you created
  in §4 (`model.summary_network.bidirectional`). `OmegaConf.update(..., force_add=True)` means even
  paths not in the defaults can be injected.
- **Remove a parameter:** delete its block — it then keeps the config/CLI default.
- **Switch to single-objective:** set `directions: [minimize]`. The objective then returns only the
  RMSE scalar and `best_trials.json` reports `study.best_trial`.
- **Tune a different metric trade-off:** the metrics themselves (RMSE + calibration error) are
  computed in `_objective`; edit that function if you want different objectives.

> CLI overrides also work for top-level tuning fields (`tuning.n_trials=...`,
> `tuning.directions=[minimize]`), but the nested `search_space` mapping is best edited in the YAML.

---

## 8. End-to-end recap (Gaussian example)

```bash
# 0. install
uv sync

# 1. (one-time code) add src/hydrabflow/simulators/gaussian.py,
#    import it in simulators/__init__.py,
#    add conf/simulator/gaussian.yaml,
#    set adapter.inference_variables=[mu1,mu2] in conf/adapter/default.yaml

# 2. generate training data
uv run python scripts/simulate.py simulator=gaussian

# 3. train
uv run python scripts/train.py simulator=gaussian
#    -> outputs/gaussian/default/<TS>/  (note this timestamp dir)

# 4. held-out test set + evaluation
uv run python scripts/simulate.py simulator=gaussian \
    data.dataset_name=test_data_10000.npz seed=123
uv run python scripts/evaluate.py simulator=gaussian \
    model_dir=outputs/gaussian/default/<TS>

# 5. real data (optional)
uv run python scripts/evaluate_real.py simulator=gaussian \
    model_dir=outputs/gaussian/default/<TS> \
    data.real_data_path=path/to/observed.npz

# 6. hyperparameter tuning
uv run python scripts/tune.py simulator=gaussian
```

Inspect any run interactively with the Marimo notebook:

```bash
uv run marimo edit notebooks/explore.py
```

That's the whole loop. To move to a new problem you only repeat §2 (a new simulator + its config +
the adapter keys); the networks, tuning, and diagnostics infrastructure carry over unchanged.
