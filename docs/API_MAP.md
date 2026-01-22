# API Map

*Generated: 2026-01-21 03:06*

## Public API

```python
from wendling_sim import simulate, optimize
```

### `simulate(sim_cfg, model_cfg, network_cfg=None, stim_cfg=None, noise_cfg=None, monitor_cfg=None) -> SimResult`

Run forward simulation (single node or network) with specified configuration.

**Returns:** `SimResult` with fields:
- `t_s`: time array (seconds)
- `lfp`: LFP proxy array (time-major, shape (T, N))
- `states`: monitored state variables (optional)
- `meta`: simulation metadata dict

### `optimize(opt_cfg, objective_cfg, target_psd, target_freqs=None) -> OptResult`

Run nevergrad optimization to find parameters minimizing PSD loss.

**Returns:** `OptResult` with fields:
- `best_params`: optimal parameter dict
- `best_loss`: final loss value
- `history`: optimization history

## Config Schemas

See `configs/` directory for YAML examples.