# BrainPy Wendling Model

BrainPy implementation of the Wendling–Chauvel neural mass model with:
- single-node simulation
- N-node network / whole-brain style simulation (shared local params)
- PSD-based fitting with nevergrad

New here? Read `full_guide.md` first.

## Install (conda)

```bash
conda activate brainpy_model
pip install -r requirements.txt
```

## TL;DR examples

### 1) Single node

```python
from wendling_sim import simulate
from wendling_sim.model.params import get_type_params

sim_cfg = {"dt_ms": 0.1, "duration_ms": 5000, "jit": True, "integrator": "rk4"}
model_cfg = get_type_params("Type4")
noise_cfg = {"seed": 42}  # deterministic

res = simulate(sim_cfg=sim_cfg, model_cfg=model_cfg, noise_cfg=noise_cfg)
print(res.lfp.shape)  # (T, 1)
```

### 2) Network from a generator (20 nodes)

```python
from wendling_sim import simulate

sim_cfg = {"dt_ms": 0.1, "duration_ms": 2000, "jit": True, "integrator": "rk4"}
network_cfg = {
    "n_nodes": 20,
    "generator": {
        "name": "erdos_renyi",
        "options": {"p": 0.2, "weight_dist": "lognormal", "weight_scale": 0.8, "seed": 123},
    },
    "normalize": "row_sum",
    "remove_self_loops": True,
    "G_net": 0.8,
}

res = simulate(sim_cfg=sim_cfg, model_cfg={}, network_cfg=network_cfg, noise_cfg={"seed": 0})
print(res.lfp.shape)  # (T, 20)
```

### 3) Network from an external connectivity file (`W_path`)

```python
from wendling_sim import simulate

network_cfg = {
    "n_nodes": 86,                 # must match file shape
    "W_path": "data/connectome/W.npy",  # .npy or .csv
    "labels_path": "data/connectome/labels.txt",  # optional
    "normalize": "row_sum",
    "remove_self_loops": True,
    "G_net": 0.6,
}

res = simulate(
    sim_cfg={"dt_ms": 0.1, "duration_ms": 2000, "jit": True, "integrator": "rk4"},
    model_cfg={},
    network_cfg=network_cfg,
    noise_cfg={"seed": 1},
)
```

See `notebooks/exploration/wendling_network_examples.ipynb` for multi-node plots + connectivity heatmaps/graphs.

## Public API

```python
from wendling_sim import simulate, optimize
```

### `simulate(...) -> SimResult`

Signature (simplified):
```python
simulate(sim_cfg, model_cfg, network_cfg=None, stim_cfg=None, noise_cfg=None, monitor_cfg=None)
```

Returns `SimResult` (see `src/wendling_sim/sim/runner.py`):
- `t_s`: time in seconds, shape `(T,)`
- `lfp`: LFP proxy, time-major, shape `(T, N)`
- `states`: optional monitored state arrays, each shape `(T, N)` (only if requested)
- `meta`: dt/duration/seed/n_nodes/G_net/connectivity metadata, etc.

### `optimize(...) -> OptResult`

Signature (simplified):
```python
optimize(opt_cfg, objective_cfg, target_psd, target_freqs=None)
```

Returns `OptResult` (see `src/wendling_sim/optimize/nevergrad_engine.py`):
- `best_params`, `best_loss`, `history`, `meta`

## Config reference (what each dict means)

### `sim_cfg`

Common keys:
- `dt_ms` or `dt_s`: time step
- `duration_ms` or `duration_s`: total simulation time
- `jit`: `True/False` (BrainPy DSRunner JIT)
- `integrator`: `"rk4"` (default) or `"euler"` (mapped to ExponentialEuler currently)
- `progress_bar`: `True/False`
- `debug`: `True/False` (records a few extra states if monitors not specified)

Where it is used: `src/wendling_sim/sim/runner.py` (`run_simulation()`).

### `model_cfg`

Local Wendling parameters (shared across nodes). Typical keys:
- gains: `A, B, G`
- time constants: `a, b, g`
- sigmoid: `e0, v0, r`
- input: `p_mean, p_sigma`
- connectivity scalars: `C, C1..C7`

Where defaults live: `src/wendling_sim/model/params.py`.

### `noise_cfg`

Keys:
- `seed`: random seed
- `p_mean`, `p_sigma`: override the same in `model_cfg`
- `mode`: `"gaussian"` (default in network) or `"uniform"`
- `shared`: `True/False` (shared noise across nodes vs per-node)

Where it is applied: `src/wendling_sim/model/wendling_network.py` and `src/wendling_sim/model/wendling_single.py`.

### `network_cfg`

Required for networks; optional for single-node.

Core keys:
- `n_nodes`: number of nodes
- `G_net`: global coupling gain
- `normalize`: `"row_sum"` (default) | `"max"` | `"none"`
- `remove_self_loops`: `True` (default) / `False`

Connectivity source (choose one):
- **Direct matrix**:
  - `W`: array-like `(N, N)`
- **File**:
  - `W_path`: `.npy` or `.csv`
  - `labels_path`: optional `.txt/.csv/.json`
- **Generator**:
  - `generator`: `{"name": "...", "options": {...}}`
  - names: `erdos_renyi`, `small_world`, `ring_lattice`, `stochastic_block_model`
  - implemented in: `src/wendling_sim/connectivity/generators.py`
- **Custom builder callable**:
  - `builder`: callable called as `builder(n_nodes=..., seed=..., **builder_kwargs)`
  - `builder_kwargs`: dict passed through

Connectivity semantics (important):
- `W[i, j]` means **source j -> target i** (target row, source column).

Where it is loaded/normalized: `src/wendling_sim/connectivity/io.py` (`load_connectivity()`).

### `monitor_cfg`

To record additional state variables:
```python
monitor_cfg = {"variables": ["y0", "y1", "y2", "y3", "y4"]}
```
This increases memory use, especially for whole-brain runs.

## How to load/import connectivities

You can either:

### A) Let `simulate()` load it from disk
```python
network_cfg = {"n_nodes": 86, "W_path": "W.npy", "normalize": "row_sum", "G_net": 0.6}
res = simulate(sim_cfg=sim_cfg, model_cfg=model_cfg, network_cfg=network_cfg, noise_cfg=noise_cfg)
```

### B) Load once (for inspection), then pass `W`
```python
from wendling_sim.connectivity.io import load_connectivity

conn = load_connectivity({"n_nodes": 86, "W_path": "W.npy", "normalize": "row_sum"})
print(conn.W.shape, conn.meta)

res = simulate(sim_cfg=sim_cfg, model_cfg=model_cfg, network_cfg={"n_nodes": 86, "W": conn.W, "G_net": 0.6})
```

## Optimization: targets, losses, and where to modify them

### 1) How targets work (PSD targets)

`optimize()` currently fits **PSD** computed from the simulation LFP.

To create a target from your own data:
```python
import numpy as np
from wendling_sim.features.psd import compute_psd

lfp = np.load("my_lfp.npy")            # shape (T,) or (T, N)
fs = 1000.0 / 0.1                      # if dt_ms=0.1
target = compute_psd(lfp, fs=fs, roi="mean", freq_range=(1, 50))
target_freqs, target_psd = target.freqs, target.psd
```

Target shape rules:
- If you use `roi="mean"` you typically get `psd` shape `(F, 1)` (or `(F,)` depending on input).
- If you use `roi="none"` you get `psd` shape `(F, N)` and your loss must handle that.

### 2) How to call `optimize()`

Minimal example:
```python
from wendling_sim import optimize

opt_cfg = {"budget": 100, "optimizer": "NGOpt", "seed": 0}
objective_cfg = {
    "sim_cfg": {"dt_ms": 0.1, "duration_ms": 5000, "jit": True, "integrator": "rk4"},
    "network_cfg": network_cfg,  # can be omitted for single-node
    "psd_cfg": {"nperseg": 4096, "freq_range": (1, 50), "roi": "mean"},
    "loss_name": "psd_mse",
    "loss_cfg": {},              # kwargs forwarded to the loss
}

result = optimize(opt_cfg=opt_cfg, objective_cfg=objective_cfg, target_psd=target_psd, target_freqs=target_freqs)
print(result.best_params, result.best_loss)
```

### 3) Add your own loss function

Where losses live:
- implementations: `src/wendling_sim/loss/psd_mse.py`
- registry/wiring: `src/wendling_sim/loss/registry.py`

Steps:
1) Create `src/wendling_sim/loss/my_loss.py`:
```python
import numpy as np

def my_loss(psd: np.ndarray, target_psd: np.ndarray, freqs=None, **kwargs) -> float:
    # psd and target_psd can be (F,) or (F, N) depending on roi
    return float(np.mean((psd - target_psd) ** 2))
```
2) Register it in `src/wendling_sim/loss/registry.py`:
```python
from wendling_sim.loss.my_loss import my_loss
register_loss("my_loss", my_loss)
```
3) Use it:
```python
objective_cfg["loss_name"] = "my_loss"
objective_cfg["loss_cfg"] = {"some_kwarg": 123}
```

### 3b) Multi-objective (multiple losses)

`optimize()` ultimately needs **one scalar** to feed nevergrad, but you can define a multi-term objective by
combining multiple loss functions with weights:

```python
objective_cfg = {
    "sim_cfg": {...},
    "psd_cfg": {"roi": "mean", "freq_range": (1, 50)},
    "losses": [
        {"name": "psd_mse", "weight": 1.0, "cfg": {}},
        {"name": "log_psd_mse", "weight": 0.2, "cfg": {}},
        {"name": "weighted_psd_mse", "weight": 0.5, "cfg": {"roi_band": (8, 13), "roi_weight": 10.0}},
    ],
}
```

The optimizer still minimizes the weighted sum; per-term values are saved in `OptResult.meta["loss_details"]`.

### 4) If you need a *non-PSD* objective

`optimize()` is PSD-oriented. For custom objectives (e.g., time-domain targets, ERP, bandpower features),
build your own `objective_fn` and use `NevergradOptimizer` directly:
- `src/wendling_sim/optimize/nevergrad_engine.py` (ask/tell loop)
- `src/wendling_sim/optimize/search_space.py` (bounds)

## Where to change what (map)

| What you want to change | Where |
|---|---|
| Single-node dynamics | `src/wendling_sim/model/wendling_single.py` |
| Network coupling / injection site | `src/wendling_sim/model/wendling_network.py` |
| Connectivity loading/normalization | `src/wendling_sim/connectivity/io.py` |
| Connectivity generators | `src/wendling_sim/connectivity/generators.py` |
| Simulation runner, dt units, integrator choice | `src/wendling_sim/sim/runner.py` |
| PSD computation / ROI options | `src/wendling_sim/features/psd.py` |
| Loss functions | `src/wendling_sim/loss/*.py` + `src/wendling_sim/loss/registry.py` |
| Optimization loop | `src/wendling_sim/api.py` + `src/wendling_sim/optimize/*.py` |

## Scripts

- `scripts/run_simulate.py`: CLI for `simulate()`
- `scripts/run_optimize.py`: CLI for `optimize()`
- `scripts/estimate_runtime.py`: benchmark + linear runtime estimate

---

Last updated: 2026-01-21
