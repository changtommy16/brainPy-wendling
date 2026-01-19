# BrainPy Wendling Single-Node Model

Single-node Wendling-Chauvel neural mass model implemented in BrainPy for forward simulation.

## Quick Start

```bash
# Run forward simulation with default config
python core_wendling/run_forward.py

# Run with custom config
python core_wendling/run_forward.py --config my_config.yaml

# Run sanity checks
python core_wendling/sanity_check.py

# Plot results
python core_wendling/plot_basic.py results/wendling_forward.npz
```

## File Structure

```
core_wendling/
├── wendling_bp.py      # WendlingSingleNode model class
├── stim.py             # Stimulus generators (baseline, sine)
├── run_forward.py      # Main simulation runner
├── plot_basic.py       # Plotting utilities (time series, PSD)
├── sanity_check.py     # Validation checks
├── config.yaml         # Default configuration
└── README.md           # This file
```

## Model Description

The Wendling-Chauvel model is a 10-ODE neural mass model representing a cortical column:

- **Pyramidal cells** (main output)
- **Excitatory interneurons**
- **Slow inhibitory interneurons** (GABA_A, somatic)
- **Fast inhibitory interneurons** (GABA_A, dendritic)

### State Variables

| Variable | Description |
|----------|-------------|
| y0 | Pyramidal → Excitatory interneurons |
| y1 | EPSP (Excitatory → Pyramidal) |
| y2 | Slow IPSP (SlowInh → Pyramidal) |
| y3 | Fast IPSP (FastInh → Pyramidal) |
| y4 | Pyramidal → FastInh |
| y5-y9 | Derivatives of y0-y4 |

### LFP Output

```
LFP = y1 - y2 - y3
```

This represents the net membrane potential at pyramidal cells.

## Configuration

Edit `config.yaml` to customize:

```yaml
seed: 0
dt_ms: 0.1
duration_ms: 20000

params:
  A: 5.0       # Excitatory gain (mV)
  B: 22.0      # Slow inhibitory gain (mV)
  G: 18.0      # Fast inhibitory gain (mV)
  p_mean: 90.0 # Mean input (Hz)
  p_sigma: 2.0 # Input noise (Hz)

stim:
  kind: "sine"  # "baseline" or "sine"
  f_hz: 40      # ASSR frequency
  amp: 1.0
  onset_ms: 500
  offset_ms: 19500
```

## Stimulus Types

### Baseline
Constant input with on/off window:
```yaml
stim:
  kind: "baseline"
  amp: 10.0
  onset_ms: 500
  offset_ms: 5000
```

### Sinusoidal (ASSR)
For auditory steady-state response experiments:
```yaml
stim:
  kind: "sine"
  f_hz: 40       # 20, 30, or 40 Hz
  amp: 1.0
  onset_ms: 500
  offset_ms: 19500
```

## Python API

```python
from core_wendling.wendling_bp import WendlingSingleNode, get_default_params
from core_wendling.stim import create_stimulus, generate_stimulus_array
from core_wendling.run_forward import run_forward, load_config

# Load config and run
config = load_config('core_wendling/config.yaml')
results = run_forward(config)

# Access outputs
t = results['t']       # Time (ms)
lfp = results['lfp']   # LFP signal
stim = results['stim'] # Stimulus

# Plot
from core_wendling.plot_basic import plot_results
plot_results(results, save_path='my_plot.png')
```

## References

- Wendling et al. (2002) *Epileptic fast activity can be explained by a model of impaired GABAergic dendritic inhibition.* European Journal of Neuroscience, 15(9):1499-1508.
- Köksal Ersöz et al. (2020) *Neural mass modeling of slow-fast dynamics of seizure initiation and abortion.* PLoS Computational Biology, 16(11):e1008430.

## Time Units

- **Config/API**: milliseconds (dt_ms, duration_ms, onset_ms)
- **Internal ODEs**: seconds (a, b, g are in 1/s)
- Conversion is handled automatically.
