# BrainPy Wendling Model

A high-performance implementation of the Wendling Neural Mass Model using [BrainPy](https://github.com/brainpy/BrainPy) and [JAX](https://github.com/google/jax). This project supports single-node simulations, network models, and whole-brain modeling with GPU acceleration and gradient-free optimization.

## Quick Start

### Prerequisites (Windows Users)
**Windows users must use [WSL2 (Ubuntu)](https://learn.microsoft.com/en-us/windows/wsl/install).**  
This project relies on JAX/BrainPy features that are not fully supported on native Windows. Please install WSL2 and run all commands inside your Ubuntu terminal.

### 1. Installation

We recommend using Conda:

```bash
conda create -n brainpy_model python=3.11 -y
conda activate brainpy_model
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

For detailed environment setup, see [docs/environment.md](docs/environment.md).

### 2. Run a Simulation

Simulate a single node with "Type 4" activity (Seizure-like):

```bash
python scripts/run_simulate.py --type Type4 --plot
```

### 3. Run Optimization

Fit model parameters to a target PSD using Nevergrad:

```bash
python scripts/run_optimize.py --target-type Type4 --budget 50
```

## Documentation

*   [**Full Guide**](docs/guide.md): The comprehensive manual for simulation, network building, and optimization.
*   [**Environment Setup**](docs/environment.md): Detailed package versions and installation instructions.

## Project Structure

*   `src/`: Core source code (`wendling_sim`).
*   `scripts/`: Command-line scripts for running simulations and optimizations.
*   `configs/`: YAML configuration templates.
*   `notebooks/`: Jupyter notebooks for demos and experiments.
*   `docs/`: Documentation.
*   `data/`: Data storage (datasets, raw inputs).

## License

[MIT License](LICENSE) (or see LICENSE file)