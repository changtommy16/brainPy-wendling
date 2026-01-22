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

## 2. API Reference & Configuration Schema

This section details every configuration parameter available in the system.

### 2.1 Simulation Config (`sim_cfg`)
Controls time integration and execution backend.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dt_ms` | `float` | `0.1` | Integration time step in milliseconds. |
| `duration_ms` | `float` | `5000.0` | Total simulation duration in milliseconds. |
| `dt_s` | `float` | - | (Alternative) Time step in seconds. |
| `duration_s` | `float` | - | (Alternative) Duration in seconds. |
| `jit` | `bool` | `True` | Enable BrainPy JIT compilation (highly recommended for speed). |
| `integrator` | `str` | `"rk4"` | Numerical integrator. Options: `"rk4"`, `"euler"`. |
| `debug` | `bool` | `False` | If True, automatically monitors internal variables (`y0`...`y4`). |

### 2.2 Model Parameters (`model_cfg`)
Wendling Neural Mass Model parameters.

| Key | Unit | Default | Description |
| :--- | :--- | :--- | :--- |
| `A` | mV | `3.25` | Excitatory synaptic gain (average PSP amplitude). |
| `B` | mV | `22.0` | Slow inhibitory synaptic gain. |
| `G` | mV | `10.0` | Fast inhibitory synaptic gain. |
| `a` | Hz ($s^{-1}$) | `100.0` | Reciprocal of excitatory time constant. |
| `b` | Hz ($s^{-1}$) | `50.0` | Reciprocal of slow inhibitory time constant. |
| `g` | Hz ($s^{-1}$) | `500.0` | Reciprocal of fast inhibitory time constant. |
| `C` | - | `135.0` | Average number of synaptic contacts (connectivity constant). |
| `C1`..`C7` | - | Derived | Layer-specific connectivity constants (derived from C). |
| `p_mean` | - | `90.0` | Mean external input noise. |
| `p_sigma` | - | `2.0` | Standard deviation of external input noise. |

**Presets:** You can load standard parameters for seizure types using `get_type_params(type_name)`.
*   **Type 1:** Background (1-7 Hz)
*   **Type 2:** Sporadic Spikes
*   **Type 3:** Sustained SWD (3-6 Hz)
*   **Type 4:** Alpha-like (8-13 Hz)
*   **Type 5:** LVFA (10-20 Hz)
*   **Type 6:** Quasi-sinusoidal

### 2.3 Network Configuration (`network_cfg`)
Required when `n_nodes > 1`. Defines the structural connectivity matrix $W$.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_nodes` | `int` | `1` | Number of nodes. |
| `G_net` | `float` | `0.0` | **Global Coupling Strength.** Scalar multiplier for network input. Input to node $i$ is $G_{net} 
cdot 
sum_j W_{ij} 
cdot 	ext{output}_j$. |
| `W` | `array` | `None` | Direct numpy array of shape `(N, N)`. |
| `W_path` | `str` | `None` | Path to `.npy`, `.csv`, or `.mat` file containing $W$. |
| `normalize` | `str` | `"row_sum"` | Normalization method (see below). |
| `remove_self_loops` | `bool` | `True` | If True, sets diagonal elements $W_{ii} = 0$. |

**Normalization Methods:**
*   `"row_sum"`: $W_{ij} = W_{ij} / 
sum_k W_{ik}$. Ensures total input to each node is normalized relative to its in-degree strength. (Row stochastic).
*   `"max"`: $W_{ij} = W_{ij} / 
max(W)$. Scales weights relative to the strongest connection in the network.
*   `"none"`: No normalization is applied.

**File Formats for `W_path`:**
*   `.npy`: Standard Numpy binary.
*   `.csv`: Comma-separated values.
*   `.mat`: MATLAB file. The loader heuristically looks for variables named `'W'`, `'sc'`, `'len'`, `'data'`, or takes the first 2D array found.

### 2.4 Noise Configuration (`noise_cfg`)

| Key | Default | Description |
| :--- | :--- | :--- |
| `seed` | `None` | Random seed for reproducibility. |
| `mode` | `"gaussian"` | Noise distribution. (Currently only gaussian is fully supported in core). |
| `shared` | `False` | (Network only) If True, all nodes receive the exact same noise time series. |
| `p_mean` | - | Overrides `model_cfg['p_mean']` if provided. |
| `p_sigma` | - | Overrides `model_cfg['p_sigma']` if provided. |

### 2.5 Simulation Output (`SimResult`)
The object returned by `simulate()`.

*   `res.t_s`: `(T,)` array. Time in seconds.
*   `res.lfp`: `(T, N)` array. Simulated Local Field Potential (proxy). Time is the first dimension.
*   `res.states`: `Dict[str, (T, N)]` (Optional). Contains raw state variables (e.g., `'y0'`...`'y4'`) if `monitor_cfg` was set.
*   `res.meta`: Dictionary containing full config used for the run (dt, seed, params, connectivity info).

---

## 3. Optimization

`optimize()` automates the parameter search using `nevergrad`.

### 3.1 Optimizer Config (`opt_cfg`)

| Key | Default | Description |
| :--- | :--- | :--- |
| `budget` | `100` | Number of simulation evaluations allowed. |
| `optimizer` | `"NGOpt"` | Nevergrad algorithm name (e.g., `"CMA"`, `"DE"`, `"TwoPointsDE"`). |
| `num_workers` | `1` | Number of parallel workers. |
| `seed` | `0` | Random seed for the optimizer's proposal generation. |

### 3.2 Objective Config (`objective_cfg`)
Defines "how to evaluate one simulation".

Must contain:
1.  `sim_cfg`: (See 2.1)
2.  `network_cfg`: (See 2.3) - *Required for network models.*
3.  `target`: The ground truth data (usually a PSD array or loaded from file).
4.  **Loss Definition:** Either `loss_name` (single) or `losses` (list).

**Example of Multi-Loss:**
```python
"losses": [
    {"name": "psd_mse", "weight": 1.0, "cfg": {"freq_range": (0, 50)}},
    {"name": "log_psd_mse", "weight": 0.5}
]
```

### 3.3 Search Space
Defined using `wendling_sim.optimize.search_space.SearchSpace`.

```python
search_space = SearchSpace(
    bounds={"A": (1, 10), "G_net": (0, 5)},
    log_scale={"A"}  # Optional: Optimizes log10(A) instead of A
)
```

**Rule:** Any parameter in `bounds` is treated as a variable to optimize.
*   If name is `G_net`, it updates the network coupling.
*   Otherwise, it updates `model_cfg`.

---

## 4. Advanced Usage

### 4.1 Custom Loss Functions
You can register a custom loss function at runtime in your script:

```python
from wendling_sim.loss.registry import register_loss

def my_custom_loss(psd, target_psd, freqs, **kwargs):
    # psd: (F,) or (F, N)
    # target_psd: same shape
    return float(((psd - target_psd) ** 2).mean())

register_loss("my_loss", my_custom_loss)
```

Then use `"loss_name": "my_loss"` in your configuration.

### 4.2 Handling Target Data (PSD)
Welch's method depends on `fs` (sampling rate) and `nperseg` (window length). If your target data was computed with different settings, the frequency bins (`freqs`) won't match the simulation.

**Solution:** Interpolate your target PSD to match the simulation's frequency bins.

```python
import numpy as np

# 1. Get simulation frequencies
fs = 1000.0 / sim_cfg["dt_ms"]
sim_freqs = np.fft.rfftfreq(nperseg, 1/fs)

# 2. Interpolate target
target_psd_aligned = np.interp(sim_freqs, target_freqs_raw, target_psd_raw)
```

See `scripts/demo_hcp_optimize.py` for a full implementation of this alignment.

---

## 5. Troubleshooting & FAQ

### Q1: `ModuleNotFoundError: No module named 'wendling_sim'`
**Solution:** The package is not installed in your current environment.
*   **Fix 1 (Recommended):** Install in editable mode: `pip install -e .`
*   **Fix 2:** Add the source directory to your python path manually at the top of your script:
    ```python
    import sys
    sys.path.append("path/to/brainPy_modeling/src")
    ```

### Q2: Optimization returns `inf` or `nan` loss.
**Causes:**
*   **Unstable Parameters:** The optimizer proposed parameters (like extremely high `G_net`) that caused the simulation to blow up (values -> infinity).
*   **Log(0):** Using `log_psd_mse` on a silent signal (zeros).
**Fix:** Restrict your `SearchSpace` bounds to physically plausible ranges.

### Q3: Why does `loss` look tiny (~1e-8) but the fit is bad?
**Cause:** `psd_mse` squares the errors. If power values are `1e-4`, the error is `1e-8`.
**Fix:** Use `log_psd_mse` (Log-MSE) or `weighted_psd_mse` to emphasize the **shape** of the power spectrum rather than raw amplitude.

### Q4: I changed `integrator` to `euler` but results didn't change.
**Reason:** The current Wendling model implementation (`wendling_single.py`) uses a manual Euler step in its `update()` method for performance and simplicity. The BrainPy integrator setting is currently a placeholder for this specific model.

---

## 6. Extending the Codebase

| To Change... | Edit File... |
| :--- | :--- |
| **Model Equations** | `src/wendling_sim/model/wendling_single.py` |
| **Network Coupling** | `src/wendling_sim/model/wendling_network.py` |
| **Connectivity Loading** | `src/wendling_sim/connectivity/io.py` |
| **Loss Functions** | `src/wendling_sim/loss/` |
| **Optimization Logic** | `src/wendling_sim/optimize/nevergrad_engine.py` |

---

## 7. Recommended Workflow

1.  **Exploration:** Use `scripts/run_simulate.py` to test different parameters manually and visualize the output time series.
2.  **Calibration:** Use `scripts/run_optimize.py` to fit a **single node** to your desired power spectrum (e.g., finding parameters that generate Alpha waves).
3.  **Network Modeling:** Once you have good local parameters, define a `network_cfg` with your connectivity matrix and run simulations to study large-scale dynamics (like functional connectivity).
4.  **Whole-Brain Fitting:** Use optimization to find the best global coupling `G_net` that matches empirical FC data.

```