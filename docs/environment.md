# Environment Setup for BrainPy Modeling

This guide details how to set up the environment for the BrainPy Wendling Model project.

## Prerequisites

**Operating System:** Linux or macOS.
**Windows Users:** You **must** use [WSL2 (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/install) with an Ubuntu distribution. This project relies on JAX and BrainPy features that are not fully supported or easily configured on native Windows.

1.  Install WSL2 and Ubuntu from the Microsoft Store.
2.  Open your Ubuntu terminal.
3.  Follow the "Quick Start" instructions below *inside* your Ubuntu terminal.

## Quick Start (Conda)

We recommend using Conda to manage the environment.

```bash
# 1. Create the environment
conda create -n brainpy_model python=3.11 -y
conda activate brainpy_model

# 2. Install JAX with CUDA support (for GPU acceleration)
# NOTE: Check https://jax.readthedocs.io/en/latest/installation.html for your specific CUDA version
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3. Install project dependencies
pip install -r requirements.txt
```

## Developer Setup (Editable Mode)

To develop the package (modify `src/`), install it in editable mode:

```bash
pip install -e .
```

This allows you to edit files in `src/wendling_sim/` and have changes reflected immediately without reinstalling.

## Detailed Specification

The project was developed and tested with the following configuration:

*   **Python**: 3.11
*   **BrainPy**: 2.7.6+
*   **JAX**: 0.9.0+ (with CUDA 12 support)
*   **Nevergrad**: 1.0.12+

### Core Dependencies

*   `brainpy`: Neuromorphic computing framework.
*   `jax` / `jaxlib`: High-performance numerical computing (backend for BrainPy).
*   `nevergrad`: Gradient-free optimization library.
*   `numpy`, `scipy`, `matplotlib`: Standard scientific stack.

### Troubleshooting

**GPU not detected?**
Check if JAX sees your GPU:
```python
import jax
print(jax.devices())
```
If it prints `[CpuDevice(id=0)]`, your CUDA installation might be incorrect or mismatched with the `jaxlib` version.

**ImportError: No module named 'wendling_sim'?**
Make sure you ran `pip install -e .` in the project root, or add `src/` to your `PYTHONPATH`.
