# Full Guide: BrainPy Wendling Model (`wendling_sim`)

This guide aims to take you from **zero to running single-node, network, and whole-brain simulations**, including custom losses and optimization strategies.

The project maintains a minimalist public API. You will primarily use two functions:

```python
from wendling_sim import simulate, optimize
```

**Key Concept:** Everything follows the pipeline: **Config Dict → Simulate → Features → Loss → Optimize**.

---

## 0. Quick Start

### 0.1 Environment

Activate your environment:
```bash
conda activate brainpy_model
```

### 0.2 Single-Node Simulation (Python)

```python
from wendling_sim import simulate
from wendling_sim.model.params import get_type_params

res = simulate(
    sim_cfg={"dt_ms": 0.1, "duration_ms": 5000, "jit": True},
    model_cfg=get_type_params("Type4"),
    noise_cfg={"seed": 0},
)
print(res.lfp.shape)  # (T, 1)
```

### 0.3 CLI Usage

Run a single simulation and plot the result:
```bash
python scripts/run_simulate.py --type Type4 --plot
```

Run a PSD fitting optimization using Nevergrad:
```bash
python scripts/run_optimize.py --target-type Type4 --budget 50
```

---

## 1. Project Architecture (Mental Model)

Think of this repository as a 4-stage pipeline:

1.  **Simulation**: Wendling dynamical system (Single node or N-node network).
2.  **Feature**: Extract features from the simulated LFP (Primary feature: PSD).
3.  **Loss**: Calculate distance between "Simulated Features" and "Target Features" (Supports multiple weighted losses).
4.  **Optimization**: Use Nevergrad to iteratively propose parameters → simulate → feature → loss, minimizing the loss.

**Key Files:**
*   **Public API**: `src/wendling_sim/api.py`
*   **Simulation Runner**: `src/wendling_sim/sim/runner.py` (Returns `SimResult`)
*   **Model**: `src/wendling_sim/model/wendling_single.py`, `wendling_network.py`
*   **Connectivity**: `src/wendling_sim/connectivity/io.py`
*   **Optimization Loop**: `src/wendling_sim/optimize/nevergrad_engine.py`

---

## 2. `simulate()`: Single, Network, & Whole-Brain

The `simulate()` function takes configuration dictionaries and delegates execution to `run_simulation()`.

```python
simulate(sim_cfg, model_cfg, network_cfg=None, stim_cfg=None, noise_cfg=None, monitor_cfg=None) -> SimResult
```

### 2.1 `SimResult`
The result object contains:
*   `t_s`: Time array in seconds (shape `(T,)`).
*   `lfp`: **Time-major** LFP proxy (shape `(T, N)`).
*   `states`: Optional monitored internal states.
*   `meta`: Metadata (dt, duration, seed, connectivity info, etc.).

### 2.2 `sim_cfg`
Controls time and execution settings.
*   `dt_ms` / `duration_ms`: Time step and duration (recommended).
*   `jit`: `True` (recommended) to enable BrainPy JIT compilation.

### 2.3 `model_cfg`
Wendling local parameters (used for both single nodes and networks).
Defaults are defined in `src/wendling_sim/model/params.py`.

You can pass a partial dict to override specific parameters:
```python
model_cfg = {"A": 5.0, "B": 10.0, "G": 15.0}
```

Or load a preset type:
```python
model_cfg = get_type_params("Type4")
```

### 2.4 `network_cfg`: Building Networks & Whole-Brain Models
This is required if `n_nodes > 1`.

**Key Logic**: `W[i, j]` represents the weight from source `j` to target `i`.
Equation: `u_net = G_net * (W @ out)`

You can provide connectivity in 4 ways:

**A) Direct Matrix**
```python
network_cfg = {"n_nodes": 20, "W": my_numpy_matrix, "G_net": 0.8, "normalize": "row_sum"}
```

**B) File Path (`.npy`, `.mat`)**
```python
network_cfg = {
    "n_nodes": 86,
    "W_path": "data/connectome/W.npy",
    "normalize": "row_sum",
    "G_net": 0.6,
}
```

**C) Generator (for Toy Networks)**
Supported: `erdos_renyi`, `small_world`, `ring_lattice`, etc.
```python
network_cfg = {
    "n_nodes": 20,
    "generator": {
        "name": "erdos_renyi",
        "options": {"p": 0.2, "weight_dist": "lognormal", "seed": 123},
    },
    "G_net": 0.8,
}
```

**D) Custom Builder**
Pass a callable function that returns the weight matrix `W`.

---

## 3. PSD Feature & Target Preparation

The project primarily uses Power Spectral Density (PSD) for fitting.

### 3.1 ROI Modes (`roi`)
When computing PSD from LFP `(T, N)`:
*   `roi="none"`: Keep PSD for every node -> Shape `(F, N)`.
*   `roi="mean"`: Average LFP across nodes first, then compute PSD -> Shape `(F, 1)`. (**Recommended for beginners**).
*   `roi="subset"`: Compute for specific nodes only.

### 3.2 Aligning Target Data
Welch's method depends on `fs` and `nperseg`. If your target data was computed with different settings, the frequency bins (`freqs`) won't match the simulation.

**Solution:** Interpolate your target PSD to match the simulation's frequency bins (see `scripts/demo_hcp_optimize.py` for an example).

---

## 4. Loss Functions

The project uses a **Registry** pattern for loss functions, allowing you to select losses via config strings (e.g., `"loss_name": "psd_mse"`).

### 4.1 Built-in Losses
*   `psd_mse`: Mean Squared Error of PSD.
*   `weighted_psd_mse`: MSE with frequency weights.
*   `log_psd_mse`: MSE of log-transformed PSD.

### 4.2 Custom Losses
You can register a custom loss function at runtime in your script:

```python
from wendling_sim.loss.registry import register_loss

def my_custom_loss(psd, target_psd, freqs, **kwargs):
    return float(((psd - target_psd) ** 2).mean())

register_loss("my_loss", my_custom_loss)
```

Then use `"loss_name": "my_loss"` in your configuration.

---

## 5. Optimization

`optimize()` automates the parameter search.

### 5.1 Configuration
*   **`opt_cfg`**: Optimizer settings (budget, algorithm, number of workers).
*   **`objective_cfg`**: Simulation and Loss settings.

### 5.2 Search Space
Define which parameters to optimize and their bounds.

```python
from wendling_sim.optimize.search_space import SearchSpace

search_space = SearchSpace(
    bounds={"A": (1, 10), "B": (5, 40), "G_net": (0, 5)},
    log_scale={"A", "B"},  # Parameters to sample in log-scale
)
```

**Note on `G_net`**: If `G_net` is in the search space, it automatically overrides `network_cfg["G_net"]`. All other parameters are assumed to be model parameters (`model_cfg`).

---

## 6. Config Philosophy

This project uses Python dictionaries (or YAML files loading into dicts) for configuration.
*   Keep experiments reproducible by saving the config used for every run.
*   See `configs/` for default YAML templates.
