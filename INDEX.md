# Wendling BrainPy Simulator Index

## Quick Start

```bash
# Activate environment
conda activate brainpy_model

# Run forward simulation
python scripts/run_simulate.py

# Run optimization
python scripts/run_optimize.py --config configs/optimize_default.yaml
```

## Where is X?

| Component | Location | Description |
|-----------|----------|-------------|
| **Model (single)** | `src/wendling_sim/model/wendling_single.py` | WendlingSingleNode (BrainPy DynamicalSystem) |
| **Model (network)** | `src/wendling_sim/model/wendling_network.py` | Vectorized WendlingNetwork with coupling |
| **Parameters** | `src/wendling_sim/model/params.py` | Standard params + 6 activity types |
| **Connectivity** | `src/wendling_sim/connectivity/io.py` | Load/generate W, normalization, labels |
| **Runner** | `src/wendling_sim/sim/runner.py` | DSRunner wrapper -> SimResult (time-major LFP) |
| **Noise** | `src/wendling_sim/sim/noise.py` | Deterministic/stochastic noise policy |
| **PSD Features** | `src/wendling_sim/features/psd.py` | Welch PSD + summaries |
| **Loss** | `src/wendling_sim/loss/psd_mse.py` | PSD MSE / weighted MSE |
| **Optimizer** | `src/wendling_sim/optimize/nevergrad_engine.py` | Nevergrad ask/tell loop |
| **Search Space** | `src/wendling_sim/optimize/search_space.py` | Bounds for A,B,G,a,b,g,G_net |

## Public API

Only two functions exposed:

```python
from wendling_sim import simulate, optimize

# Forward simulation
result = simulate(sim_cfg=sim_cfg, model_cfg=model_cfg, network_cfg=network_cfg, noise_cfg=noise_cfg, monitor_cfg=monitor_cfg)

# Parameter optimization
opt_result = optimize(opt_cfg, objective_cfg)
```

## Extension Points

- **Add a new loss function**: create `src/wendling_sim/loss/my_loss.py`, register in `src/wendling_sim/loss/registry.py`, reference by name in config.
- **Add a connectivity generator**: add to `src/wendling_sim/connectivity/generators.py`, reference via `network_cfg={'generator': {'name': ...}}`.
- **Add a new stimulus type**: add to `src/wendling_sim/sim/stim.py` (reserved hook), update config schema.

## Config Files

- `configs/sim_default.yaml`  - Simulation settings
- `configs/optimize_default.yaml`  - Optimization settings
- `configs/target_psd.yaml`  - Target PSD for fitting

## Generated Docs

Run `python tools/gen_docs.py` to regenerate:
- `docs/TREE.md`  - Repo structure
- `docs/API_MAP.md`  - Public API reference
