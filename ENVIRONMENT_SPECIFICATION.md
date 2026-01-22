# Environment Specification for brainPy_modeling
# Tested Environment: brainpy311 conda environment
# Generated: 2025-01-21

## Environment Details
- **Python**: 3.11.14
- **Platform**: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- **Conda Environment**: brainpy311
- **JAX Backend**: GPU (CUDA available)

## Exact Package Versions

### Core Dependencies
```
brainpy[jax]==2.7.6
jax==0.9.0
jaxlib==0.9.0
jax-cuda12-pjrt==0.9.0
jax-cuda12-plugin==0.9.0
numpy==2.3.5
scipy==1.17.0
matplotlib==3.10.8
```

### Optimization
```
nevergrad==1.0.12
tqdm==4.67.1
```

### Data Handling
```
pandas==2.3.3
h5py==3.15.1
```

### Configuration & Utilities
```
pyyaml>=6.0
python-dotenv>=1.0.0
loguru>=0.7.0
```

### Optional Packages (Not Installed in Current Environment)
```
nilearn          # Not installed (neuroscience, fMRI processing)
pytest           # Not installed (testing)
black            # Not installed (code formatting)
flake8           # Not installed (linting)
mypy             # Not installed (type checking)
jupyter          # Not installed (notebook support)
ipywidgets       # Not installed (interactive widgets)
```

## JAX Configuration
- **Available Devices**: [CudaDevice(id=0)]
- **Default Backend**: gpu
- **GPU Support**: ✅ Available

## Installation Commands

### 1. Create Environment
```bash
conda create -n brainpy311 python=3.11 -y
conda activate brainpy311
```

### 2. Install JAX with GPU Support
```bash
pip install "jax[cuda12]==0.9.0" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# OR install individual packages:
pip install jax==0.9.0 jaxlib==0.9.0 jax-cuda12-pjrt==0.9.0 jax-cuda12-plugin==0.9.0
```

### 3. Install Dependencies
```bash
pip install brainpy[jax]==2.7.6
pip install numpy==2.3.5 scipy==1.17.0 matplotlib==3.10.8
pip install nevergrad==1.0.12 tqdm==4.67.1
pip install pandas==2.3.3 h5py==3.15.1
pip install pyyaml python-dotenv loguru
```

### 4. Install Package
```bash
cd /path/to/brainPy_modeling
pip install -e .
```

## Verification Script

```python
# verify_environment.py
import sys
import platform
import brainpy as bp
import jax
import numpy as np
import scipy
import matplotlib
import nevergrad as ng
import tqdm
import pandas as pd
import h5py

print('=== Environment Verification ===')
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print()

print('=== Core Dependencies ===')
print(f'brainpy: {bp.__version__}')
print(f'jax: {jax.__version__}')
print(f'jaxlib: {jax.lib.__version__}')
print(f'numpy: {np.__version__}')
print(f'scipy: {scipy.__version__}')
print(f'matplotlib: {matplotlib.__version__}')
print()

print('=== Optimization ===')
print(f'nevergrad: {ng.__version__}')
print(f'tqdm: {tqdm.__version__}')
print()

print('=== Data Handling ===')
print(f'pandas: {pd.__version__}')
print(f'h5py: {h5py.__version__}')
print()

print('=== JAX Configuration ===')
print(f'Available devices: {jax.devices()}')
print(f'Default backend: {jax.default_backend()}')
print()

# Test package functionality
try:
    from wendling_sim import simulate
    print('✅ wendling_sim: Successfully imported')
except ImportError as e:
    print(f'❌ wendling_sim: Import failed - {e}')

print('\n=== Environment Ready ===')
```

## Notes

1. **GPU Acceleration**: This environment has GPU support via CUDA
2. **JAX Version**: Using JAX 0.9.0 which is compatible with BrainPy 2.7.6
3. **Package Installation**: Use `requirements_brainpy311.txt` for exact versions
4. **Optional Packages**: Install as needed for development or specific functionality

## Troubleshooting

### Common Issues
1. **CUDA Errors**: Ensure CUDA drivers are up to date
2. **Memory Issues**: Use `export JAX_PLATFORMS=cpu` to force CPU if needed
3. **Import Errors**: Make sure to `pip install -e .` in the project directory

### Performance Tips
1. **GPU Memory**: Use `export XLA_PYTHON_CLIENT_PREALLOCATE=false` for memory management
2. **Compilation**: Enable JIT with `jit=True` in simulation configs
3. **Precision**: Use `export JAX_ENABLE_X64=true` for double precision if needed
