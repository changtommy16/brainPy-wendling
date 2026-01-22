# Repository Tree

*Generated: 2026-01-21 03:06*

```
brainPy_modeling/
|- configs/
|  |- optimize_default.yaml
|  `- sim_default.yaml
|- data/
|  |- external/
|  |- interim/
|  |- processed/
|  `- raw/
|- docs/
|  |- API_MAP.md
|  |- TREE.md
|  |- WENDLING_Subnetwork_WholeBrain_SPEC_TODO_v0.3.md
|  `- mkdocs.yml
|- notebooks/
|  |- brainPy_practice/
|  |- demos/
|  |- exploration/
|- references/
|- reports/
|  |- figures/
|- results/
|  |- opt_test_20260120_225935/
|  |- random_opt_20260120_230618/
|  |- random_opt_20260120_232430/
|  |- random_opt_20260120_232526/
|  |- random_opt_20260120_233041/
|  `- random_opt_20260120_233404/
|- scripts/  # Executable scripts for running simulations and demos.
|  |- __init__.py  # Executable scripts for running simulations and demos.
|  |- run_optimize.py  # Run nevergrad optimization.
|  |- run_simulate.py  # Run forward simulation.
|  `- run_six_types.py  # Run all 6 Wendling activity types and plot results.
|- src/
|  `- wendling_sim/  # Wendling BrainPy Simulator + Nevergrad Optimization
|     |- connectivity/  # Connectivity handling for Wendling networks.
|     |  |- __init__.py  # Connectivity handling for Wendling networks.
|     |  |- generators.py  # Connectivity generators for Wendling networks.
|     |  `- io.py  # Connectivity I/O and normalization utilities.
|     |- features/  # Feature extraction (PSD, etc.).
|     |  |- __init__.py  # Feature extraction (PSD, etc.).
|     |  `- psd.py  # PSD feature extraction using Welch method.
|     |- loss/  # Loss functions for optimization.
|     |  |- __init__.py  # Loss functions for optimization.
|     |  |- psd_mse.py  # PSD-based loss functions for optimization.
|     |  `- registry.py  # Loss function registry for dynamic loss selection.
|     |- model/  # Model definitions for Wendling neural mass model.
|     |  |- __init__.py  # Model definitions for Wendling neural mass model.
|     |  |- params.py  # Standard parameters and type-specific parameters for Wendling model.
|     |  |- wendling_network.py  # WendlingNetwork: Vectorized Wendling-Chauvel model for N nodes.
|     |  `- wendling_single.py  # WendlingSingleNode: BrainPy DynamicalSystem for single-node Wendling model.
|     |- optimize/  # Nevergrad optimization engine.
|     |  |- __init__.py  # Nevergrad optimization engine.
|     |  |- nevergrad_engine.py  # Nevergrad optimization engine with ask/tell interface.
|     |  `- search_space.py  # Search space definition for nevergrad optimization.
|     |- sim/  # Simulation runner and noise utilities.
|     |  |- __init__.py  # Simulation runner and noise utilities.
|     |  |- noise.py  # Noise generation for deterministic/stochastic simulation modes.
|     |  `- runner.py  # DSRunner wrapper for Wendling simulation.
|     |- utils/  # Utilities for I/O and logging.
|     |  |- __init__.py  # Utilities for I/O and logging.
|     |  |- io.py  # I/O utilities for saving/loading simulation artifacts.
|     |  `- logging.py  # Logging utilities for optimization runs.
|     |- __init__.py  # Wendling BrainPy Simulator + Nevergrad Optimization
|     `- api.py  # Public API for Wendling simulation and optimization.
|- tests/  # Tests for wendling_sim package.
|  |- __init__.py  # Tests for wendling_sim package.
|  |- test_data.py
|  |- test_optimization.py  # Test optimization by fitting to a known PSD target.
|  `- test_optimization_random.py  # Random PSD optimization test (no stimulus).
|- tools/
|  `- gen_docs.py  # Generate documentation files from source code.
|- INDEX.md
|- README.md
`- requirements.txt
```