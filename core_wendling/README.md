# BrainPy Wendling Single-Node Model

Single-node Wendling-Chauvel neural mass model implemented in BrainPy for forward simulation. This package provides a complete framework for simulating, analyzing, and visualizing neural mass model dynamics representing a cortical column with excitatory and inhibitory populations.

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

## Complete File Structure

```
core_wendling/
├── Core Model Files (Neural Mass Simulation)
│   ├── wendling_bp.py            # WendlingSingleNode model class (10-ODE system)
│   ├── stim.py                   # Stimulus generators (baseline, sine, ASSR)
│   ├── wendling_param_sets.py    # Predefined parameter sets for 6 activity types
│   └── config.yaml               # Default simulation configuration
│
├── Execution & Validation
│   ├── run_forward.py            # Main simulation runner and orchestrator
│   ├── sanity_check.py           # Model validation and integrity checks
│   └── __init__.py               # Package initialization and public API
│
├── Analysis & Visualization
│   ├── plot_basic.py             # Core plotting (time series, PSD, spectrograms)
│   ├── plots.py                  # Additional plotting utilities (template)
│   └── features.py               # Feature extraction pipeline (template)
│
├── Project Infrastructure
│   ├── config.py                 # Project paths and global configuration
│   └── dataset.py                # Dataset processing pipeline (template)
│
├── modeling/                     # ML modeling subdirectory
│   ├── __init__.py
│   ├── train.py                  # Model training pipeline (template)
│   └── predict.py                # Inference pipeline (template)
│
└── README.md                     # This file
```

## Detailed File Descriptions

### Core Model Files

#### `wendling_bp.py` - Neural Mass Model Implementation
**Purpose**: Implements the 10-ODE Wendling-Chauvel neural mass model as a BrainPy DynamicalSystem.

**Key Components**:
- `WendlingSingleNode` class: Main model with 10 state variables (y0-y9)
- `sigmoid()`: Sigmoid activation function for population firing rate
- `get_default_params()`: Returns default parameter dictionary

**State Variables**:
- y0: Pyramidal → Excitatory interneurons membrane potential
- y1: EPSP (Excitatory → Pyramidal)
- y2: Slow IPSP (Slow inhibitory → Pyramidal)
- y3: Fast IPSP (Fast inhibitory → Pyramidal)
- y4: Pyramidal → Fast inhibitory membrane potential
- y5-y9: Time derivatives of y0-y4

**Model Parameters**:
- A, B, G: Synaptic gains (mV) for excitatory, slow inhibitory, fast inhibitory
- a, b, g: Time constants (1/s)
- C1-C7: Connectivity constants
- e0, v0, r: Sigmoid parameters
- p_mean, p_sigma: Mean and noise of input drive (Hz)

**Usage Example**:
```python
from core_wendling.wendling_bp import WendlingSingleNode, get_default_params
params = get_default_params()
params['A'] = 5.0
params['B'] = 25.0
model = WendlingSingleNode(params)
```

#### `stim.py` - Stimulus Generators
**Purpose**: Provides various input signal generators for driving the neural mass model.

**Classes**:
- `BaselineStimulus`: Constant drive with onset/offset window
- `SineStimulus`: Sinusoidal drive for ASSR (Auditory Steady-State Response) experiments

**Functions**:
- `create_stimulus(config)`: Factory function to create stimulus from config dict
- `generate_stimulus_array(stim_obj, t_array)`: Generate full stimulus time series

**Stimulus Injection**: External input u(t) is added to the pyramidal cell equation in units of Hz (firing rate).

**Usage Example**:
```python
from core_wendling.stim import SineStimulus, generate_stimulus_array
stim = SineStimulus(f_hz=40, amp=1.0, onset_ms=500, offset_ms=19500)
stim_array = generate_stimulus_array(stim, t_array)
```

#### `wendling_param_sets.py` - Predefined Parameter Sets
**Purpose**: Provides 6 validated parameter sets from the original Wendling et al. (2002) paper.

**Available Parameter Sets**:
1. **Type1**: Background activity (1-7 Hz slow waves)
2. **Type2**: Sporadic spikes (1-5 Hz)
3. **Type3**: Sustained spike-wave discharge (3-6 Hz)
4. **Type4**: Alpha-like rhythm (8-13 Hz)
5. **Type5**: Low-voltage fast activity / LVFA (10-20 Hz)
6. **Type6**: Quasi-sinusoidal activity (9-13 Hz)

**Functions**:
- `list_type_params()`: Returns list of available type names
- `get_type_params(type_name)`: Returns parameter dict for specified type

**Usage Example**:
```python
from core_wendling.wendling_param_sets import get_type_params, WENDLING_TYPE_PARAMS
params = get_type_params("Type3")  # Spike-wave discharge
print(WENDLING_TYPE_PARAMS["Type3"]["description"])
```

#### `config.yaml` - Default Configuration
**Purpose**: YAML configuration file for simulation parameters.

**Configuration Sections**:
- `seed`: Random seed for reproducibility
- `dt_ms`: Time step in milliseconds
- `duration_ms`: Total simulation duration
- `params`: Model parameters (A, B, G, p_mean, p_sigma)
- `stim`: Stimulus configuration (kind, frequency, amplitude, timing)
- `output`: Output file paths and formats

### Execution & Validation

#### `run_forward.py` - Main Simulation Runner
**Purpose**: Orchestrates the complete simulation pipeline from config loading to result saving.

**Main Functions**:
- `load_config(config_path)`: Loads YAML configuration
- `run_forward(config)`: Runs complete forward simulation
- Command-line interface with argparse

**Workflow**:
1. Load configuration from YAML
2. Build WendlingSingleNode model
3. Generate stimulus array
4. Create BrainPy DSRunner
5. Run simulation
6. Compute LFP (lfp = y1 - y2 - y3)
7. Save results to .npz file

**Usage**:
```bash
python core_wendling/run_forward.py --config config.yaml
```

**Python API**:
```python
from core_wendling.run_forward import run_forward, load_config
config = load_config('config.yaml')
results = run_forward(config)  # Returns dict with t, lfp, stim, states
```

#### `sanity_check.py` - Model Validation
**Purpose**: Validates model implementation and parameter ranges.

**Checks Performed**:
- Model initialization with default parameters
- State variable bounds and stability
- LFP computation correctness
- Stimulus injection validation
- Parameter sensitivity tests
- Comparison with expected frequency ranges

**Usage**:
```bash
python core_wendling/sanity_check.py
```

#### `__init__.py` - Package Initialization
**Purpose**: Defines the public API of the core_wendling package.

**Exports**:
- Model: `WendlingSingleNode`, `get_default_params`, `sigmoid`
- Stimulus: `BaselineStimulus`, `SineStimulus`, `create_stimulus`, `generate_stimulus_array`
- Runner: `run_forward`, `load_config`

**Usage**:
```python
from core_wendling import WendlingSingleNode, run_forward, SineStimulus
```

### Analysis & Visualization

#### `plot_basic.py` - Core Plotting Utilities
**Purpose**: Provides comprehensive visualization of simulation results.

**Plotting Functions**:
- Time series plots (LFP, stimulus, state variables)
- Power Spectral Density (PSD) analysis
- Spectrograms (time-frequency analysis)
- Multi-panel figure generation

**Usage**:
```bash
python core_wendling/plot_basic.py results/wendling_forward.npz
```

**Python API**:
```python
from core_wendling.plot_basic import plot_results
results = np.load('results.npz')
plot_results(results, save_path='figure.png')
```

#### `plots.py` - Additional Plotting (Template)
**Purpose**: Template for additional custom plotting utilities.

**Current Status**: Placeholder template with typer CLI interface.

**Intended Usage**: Extend with domain-specific plots (phase portraits, bifurcation diagrams, etc.)

#### `features.py` - Feature Extraction (Template)
**Purpose**: Template for extracting features from simulation results.

**Current Status**: Placeholder template with typer CLI interface.

**Intended Usage**: Extract features for ML pipelines (spectral features, complexity measures, etc.)

### Project Infrastructure

#### `config.py` - Global Configuration
**Purpose**: Defines project-wide paths and settings.

**Key Paths Defined**:
- `PROJ_ROOT`: Project root directory
- `DATA_DIR`: Data directory with subdirectories:
  - `RAW_DATA_DIR`: Raw input data
  - `INTERIM_DATA_DIR`: Intermediate processing
  - `PROCESSED_DATA_DIR`: Final processed data
  - `EXTERNAL_DATA_DIR`: External datasets
- `MODELS_DIR`: Saved models directory
- `REPORTS_DIR`: Reports and figures directory
- `FIGURES_DIR`: Generated figures

**Logger Configuration**: Integrates loguru with tqdm for progress bars.

**Usage**:
```python
from core_wendling.config import DATA_DIR, FIGURES_DIR
input_file = DATA_DIR / "raw" / "data.csv"
```

#### `dataset.py` - Dataset Processing (Template)
**Purpose**: Template for dataset loading and preprocessing.

**Current Status**: Placeholder template with typer CLI interface.

**Intended Usage**: Process raw EEG/LFP data for model fitting or comparison.

### Modeling Subdirectory

#### `modeling/train.py` - Training Pipeline (Template)
**Purpose**: Template for training machine learning models on simulation data.

**Current Status**: Placeholder with typer CLI interface.

**Intended Usage**: Train surrogate models, parameter inference models, or classifiers.

#### `modeling/predict.py` - Inference Pipeline (Template)
**Purpose**: Template for running inference with trained models.

**Current Status**: Placeholder with typer CLI interface.

**Intended Usage**: Apply trained models to new data for predictions.

## How Files Work Together

### Typical Workflow

```
1. Configuration
   config.yaml → load_config() → config dict

2. Model Setup
   config dict → WendlingSingleNode() → model instance
   config dict → create_stimulus() → stimulus object

3. Simulation
   model + stimulus → DSRunner.run() → state trajectories

4. Analysis
   state trajectories → compute LFP → results dict

5. Visualization
   results dict → plot_basic.py → figures

6. Validation
   sanity_check.py → verify model behavior
```

### Data Flow Diagram

```
┌─────────────┐
│ config.yaml │
└──────┬──────┘
       │
       ├──────────────────┐
       │                  │
       v                  v
┌──────────────┐   ┌─────────────┐
│ Model Params │   │ Stim Config │
└──────┬───────┘   └──────┬──────┘
       │                  │
       v                  v
┌──────────────┐   ┌─────────────┐
│ WendlingSingleNode│   │ SineStimulus│
└──────┬───────┘   └──────┬──────┘
       │                  │
       └────────┬─────────┘
                v
         ┌─────────────┐
         │  DSRunner   │
         └──────┬──────┘
                v
         ┌─────────────┐
         │  Run Loop   │
         └──────┬──────┘
                v
      ┌──────────────────┐
      │ State Trajectories│
      │   (y0, y1, ...)   │
      └──────┬───────────┘
             v
      ┌─────────────┐
      │ Compute LFP │
      │ y1-y2-y3    │
      └──────┬──────┘
             v
      ┌─────────────┐
      │ Results Dict│
      │ (t, lfp, stim)│
      └──────┬──────┘
             │
             ├──────────┬──────────┐
             v          v          v
      ┌─────────┐ ┌────────┐ ┌─────────┐
      │Save .npz│ │ Plots  │ │Features │
      └─────────┘ └────────┘ └─────────┘
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

## Comprehensive Examples

### Example 1: Basic Simulation with Default Config

```python
from core_wendling.run_forward import run_forward, load_config

# Load default configuration
config = load_config('core_wendling/config.yaml')

# Run simulation
results = run_forward(config)

# Access outputs
t = results['t']           # Time array (ms)
lfp = results['lfp']       # LFP signal (mV)
stim = results['stim']     # Stimulus array (Hz)
states = results['states'] # All state variables (y0-y9)

print(f"Simulation duration: {t[-1]} ms")
print(f"LFP shape: {lfp.shape}")
print(f"Mean LFP: {lfp.mean():.2f} mV")
```

### Example 2: Using Predefined Parameter Sets

```python
from core_wendling.wendling_bp import WendlingSingleNode, get_default_params
from core_wendling.wendling_param_sets import get_type_params, list_type_params
from core_wendling.run_forward import load_config, run_forward
import numpy as np

# List all available parameter types
print("Available parameter types:")
for type_name in list_type_params():
    print(f"  - {type_name}")

# Load config and replace with Type3 parameters (spike-wave discharge)
config = load_config('core_wendling/config.yaml')
type3_params = get_type_params("Type3")
config['params'].update(type3_params)

# Run simulation
results = run_forward(config)

# Analyze frequency content
from scipy import signal
fs = 1000.0 / config['dt_ms']  # Sampling frequency
f, psd = signal.welch(results['lfp'], fs=fs, nperseg=2048)

# Find dominant frequency
dominant_freq = f[np.argmax(psd)]
print(f"Dominant frequency: {dominant_freq:.2f} Hz")
print(f"Expected range: 3-6 Hz (spike-wave discharge)")
```

### Example 3: Custom Stimulus Configuration

```python
from core_wendling.stim import SineStimulus, BaselineStimulus, generate_stimulus_array
from core_wendling.wendling_bp import WendlingSingleNode, get_default_params
import brainpy as bp
import numpy as np

# Setup model with custom parameters
params = get_default_params()
params['A'] = 5.0
params['B'] = 22.0
params['G'] = 18.0
model = WendlingSingleNode(params)

# Create custom 40 Hz ASSR stimulus
dt_ms = 0.1
duration_ms = 10000
t_array = np.arange(0, duration_ms, dt_ms)

stim = SineStimulus(
    f_hz=40,           # 40 Hz auditory steady-state
    amp=2.0,           # Increased amplitude
    onset_ms=1000,     # Start at 1 second
    offset_ms=9000,    # End at 9 seconds
    dt_ms=dt_ms
)
stim_array = generate_stimulus_array(stim, t_array)

# Run simulation
runner = bp.DSRunner(
    model,
    monitors=['y0', 'y1', 'y2', 'y3', 'y4'],
    dt=dt_ms / 1000.0  # Convert to seconds
)
runner.run(duration_ms / 1000.0, inputs={'u': stim_array})

# Compute LFP
lfp = runner.mon['y1'] - runner.mon['y2'] - runner.mon['y3']
print(f"LFP computed, shape: {lfp.shape}")
```

### Example 4: Parameter Sweep

```python
from core_wendling.run_forward import load_config, run_forward
import numpy as np
import matplotlib.pyplot as plt

# Load base configuration
config = load_config('core_wendling/config.yaml')
config['duration_ms'] = 5000  # Shorter for speed

# Sweep slow inhibitory gain (B parameter)
B_values = np.linspace(5, 50, 10)
lfp_results = []

for B in B_values:
    config['params']['B'] = float(B)
    results = run_forward(config)
    lfp_results.append(results['lfp'])
    print(f"Completed B={B:.1f}")

# Visualize parameter sweep
fig, axes = plt.subplots(10, 1, figsize=(12, 15))
for i, (B, lfp) in enumerate(zip(B_values, lfp_results)):
    axes[i].plot(results['t'][:5000], lfp[:5000])
    axes[i].set_ylabel(f'B={B:.1f}')
    axes[i].set_xlim(0, 500)
plt.tight_layout()
plt.savefig('parameter_sweep_B.png')
print("Parameter sweep complete, saved to parameter_sweep_B.png")
```

### Example 5: Direct Model Usage (Low-Level API)

```python
import brainpy as bp
import brainpy.math as bm
import numpy as np
from core_wendling.wendling_bp import WendlingSingleNode, get_default_params

# Setup
bm.set_dt(0.1)  # 0.1 ms time step
params = get_default_params()
model = WendlingSingleNode(params)

# Initialize
model.reset()

# Manual simulation loop
dt_ms = 0.1
duration_ms = 1000
n_steps = int(duration_ms / dt_ms)

# Pre-allocate arrays
y1_trace = np.zeros(n_steps)
y2_trace = np.zeros(n_steps)
y3_trace = np.zeros(n_steps)

# Simulate
for step in range(n_steps):
    # Apply constant input
    u_input = 10.0 if 100 < step < 900 else 0.0

    # Update model
    model.step_run(0, dt_ms/1000.0, u_input)

    # Record state
    y1_trace[step] = model.y1.value
    y2_trace[step] = model.y2.value
    y3_trace[step] = model.y3.value

# Compute LFP
lfp = y1_trace - y2_trace - y3_trace
t = np.arange(n_steps) * dt_ms

print(f"Manual simulation complete: {n_steps} steps")
print(f"LFP range: [{lfp.min():.2f}, {lfp.max():.2f}] mV")
```

### Example 6: Analyzing All 6 Activity Types

```python
from core_wendling.wendling_param_sets import WENDLING_TYPE_PARAMS, get_type_params
from core_wendling.run_forward import load_config, run_forward
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# Base configuration
config = load_config('core_wendling/config.yaml')
config['duration_ms'] = 10000
config['stim']['kind'] = 'baseline'
config['stim']['amp'] = 0.0

# Run all 6 types
fig, axes = plt.subplots(6, 2, figsize=(14, 12))

for idx, type_name in enumerate(WENDLING_TYPE_PARAMS.keys()):
    # Get parameters and info
    type_info = WENDLING_TYPE_PARAMS[type_name]
    params = get_type_params(type_name)

    # Update config
    config['params'].update(params)

    # Run simulation
    results = run_forward(config)
    t = results['t']
    lfp = results['lfp']

    # Compute PSD
    fs = 1000.0 / config['dt_ms']
    f, psd = signal.welch(lfp, fs=fs, nperseg=2048)

    # Plot time series
    axes[idx, 0].plot(t[5000:10000], lfp[5000:10000])
    axes[idx, 0].set_title(f"{type_info['name']}")
    axes[idx, 0].set_ylabel('LFP (mV)')

    # Plot PSD
    axes[idx, 1].semilogy(f, psd)
    axes[idx, 1].set_xlim(0, 50)
    axes[idx, 1].set_ylabel('PSD')
    axes[idx, 1].axvspan(type_info['expected']['freq_range'][0],
                         type_info['expected']['freq_range'][1],
                         alpha=0.3, color='red')

axes[-1, 0].set_xlabel('Time (ms)')
axes[-1, 1].set_xlabel('Frequency (Hz)')
plt.tight_layout()
plt.savefig('all_activity_types.png', dpi=150)
print("Generated all_activity_types.png")
```

### Example 7: Saving and Loading Results

```python
import numpy as np
from core_wendling.run_forward import run_forward, load_config

# Run and save
config = load_config('core_wendling/config.yaml')
results = run_forward(config)

# Save to npz (done automatically by run_forward, but can also do manually)
np.savez_compressed(
    'my_results.npz',
    t=results['t'],
    lfp=results['lfp'],
    stim=results['stim'],
    y0=results['states'][:, 0],
    y1=results['states'][:, 1],
    y2=results['states'][:, 2],
    y3=results['states'][:, 3],
    y4=results['states'][:, 4],
    config=config  # Save config too
)

# Load and analyze
loaded = np.load('my_results.npz', allow_pickle=True)
print("Loaded keys:", list(loaded.keys()))
print(f"LFP shape: {loaded['lfp'].shape}")
print(f"Time range: {loaded['t'][0]} to {loaded['t'][-1]} ms")

# Visualize loaded results
from core_wendling.plot_basic import plot_results
plot_results(loaded, save_path='loaded_results.png')
```

## References

- Wendling et al. (2002) *Epileptic fast activity can be explained by a model of impaired GABAergic dendritic inhibition.* European Journal of Neuroscience, 15(9):1499-1508.
- Köksal Ersöz et al. (2020) *Neural mass modeling of slow-fast dynamics of seizure initiation and abortion.* PLoS Computational Biology, 16(11):e1008430.

## Time Units

- **Config/API**: milliseconds (dt_ms, duration_ms, onset_ms)
- **Internal ODEs**: seconds (a, b, g are in 1/s)
- Conversion is handled automatically.

## Quick Reference

### Command-Line Interface

```bash
# Forward simulation
python core_wendling/run_forward.py                          # Use default config
python core_wendling/run_forward.py --config my_config.yaml  # Use custom config

# Validation
python core_wendling/sanity_check.py                         # Run model validation

# Visualization
python core_wendling/plot_basic.py results/output.npz        # Plot results from file
```

### Key Imports

```python
# Core model
from core_wendling import WendlingSingleNode, get_default_params, sigmoid

# Stimulus generation
from core_wendling import SineStimulus, BaselineStimulus
from core_wendling import create_stimulus, generate_stimulus_array

# Simulation runner
from core_wendling import run_forward, load_config

# Parameter sets
from core_wendling.wendling_param_sets import get_type_params, list_type_params

# Plotting
from core_wendling.plot_basic import plot_results

# Paths
from core_wendling.config import PROJ_ROOT, DATA_DIR, FIGURES_DIR
```

### Important State Variables

| Variable | Description | Typical Range |
|----------|-------------|---------------|
| y0 | Pyramidal → Excitatory | -10 to 10 mV |
| y1 | EPSP (Excitatory → Pyramidal) | -10 to 10 mV |
| y2 | Slow IPSP | -30 to 0 mV |
| y3 | Fast IPSP | -30 to 0 mV |
| y4 | Pyramidal → Fast inhibitory | -10 to 10 mV |
| LFP | y1 - y2 - y3 | -30 to 30 mV |

### Key Parameters

| Parameter | Description | Default | Range | Effect |
|-----------|-------------|---------|-------|--------|
| A | Excitatory synaptic gain | 5.0 mV | 3-8 | Higher → more excitability |
| B | Slow inhibitory gain | 22.0 mV | 5-50 | Higher → more slow inhibition |
| G | Fast inhibitory gain | 18.0 mV | 0-30 | Higher → more fast inhibition |
| a | Excitatory time constant | 100 s⁻¹ | 50-200 | Higher → faster dynamics |
| b | Slow inhibitory time constant | 50 s⁻¹ | 20-100 | Higher → faster dynamics |
| g | Fast inhibitory time constant | 500 s⁻¹ | 200-1000 | Higher → faster dynamics |
| p_mean | Mean input drive | 90 Hz | 50-200 | Baseline excitation level |
| p_sigma | Input noise std | 2.0 Hz | 0-10 | Stochasticity |

### Activity Types (wendling_param_sets.py)

| Type | Name | Frequency | B value | G value | Description |
|------|------|-----------|---------|---------|-------------|
| Type1 | Background | 1-7 Hz | 50 | 15 | Slow waves |
| Type2 | Sporadic spikes | 1-5 Hz | 40 | 15 | Intermittent spikes |
| Type3 | Sustained SWD | 3-6 Hz | 25 | 15 | Spike-wave discharge |
| Type4 | Alpha-like | 8-13 Hz | 10 | 15 | Alpha rhythm |
| Type5 | LVFA | 10-20 Hz | 5 | 25 | Low-voltage fast |
| Type6 | Quasi-sinusoidal | 9-13 Hz | 15 | 0 | Regular oscillation |

### File Categories

**Core Implementation Files** (Must understand):
- `wendling_bp.py` - Neural mass model (10 ODEs)
- `stim.py` - Stimulus generators
- `run_forward.py` - Main simulation runner
- `config.yaml` - Configuration

**Analysis & Validation** (Frequently used):
- `sanity_check.py` - Validation tests
- `plot_basic.py` - Visualization
- `wendling_param_sets.py` - Predefined parameters

**Infrastructure** (Utility/templates):
- `config.py` - Project paths
- `__init__.py` - Package API
- `dataset.py`, `features.py`, `plots.py` - Templates
- `modeling/train.py`, `modeling/predict.py` - ML templates

### Typical Research Workflows

**1. Explore Activity Types**
```bash
# Run all 6 predefined types and visualize
python -c "from examples import run_all_types; run_all_types()"
```

**2. ASSR Experiment**
```yaml
# Edit config.yaml
stim:
  kind: "sine"
  f_hz: 40
  amp: 1.0
```

**3. Parameter Fitting**
```python
# Use run_forward() in optimization loop
from core_wendling import run_forward, load_config
config = load_config('config.yaml')
# ... optimization code using run_forward(config)
```

**4. Validation Against Data**
```python
# Compare model LFP with empirical data
results = run_forward(config)
model_lfp = results['lfp']
# ... compare with real_lfp using correlation, PSD, etc.
```

## Summary

The `core_wendling` folder implements a complete BrainPy-based simulation framework for the Wendling-Chauvel neural mass model. The architecture is modular and extensible:

- **Core model** (`wendling_bp.py`) implements the 10-ODE system
- **Stimulus library** (`stim.py`) provides flexible input generation
- **High-level runner** (`run_forward.py`) orchestrates simulations
- **Validation suite** (`sanity_check.py`) ensures correctness
- **Parameter library** (`wendling_param_sets.py`) provides validated parameter sets
- **Analysis tools** (`plot_basic.py`) visualize results
- **Template files** provide scaffolding for extensions

The framework supports both command-line and programmatic usage, making it suitable for interactive exploration, batch processing, parameter sweeps, and integration into larger pipelines.
