"""
I/O utilities for saving/loading simulation artifacts.
"""

import json
from pathlib import Path
from typing import Union
import numpy as np

from wendling_sim.sim.runner import SimResult


def save_sim_result(result: SimResult, path: Union[str, Path]):
    """Save SimResult to npz file."""
    result.save(str(path))


def load_sim_result(path: Union[str, Path]) -> SimResult:
    """Load SimResult from npz file."""
    return SimResult.load(str(path))


def save_params(params: dict, path: Union[str, Path]):
    """Save parameters to JSON file."""
    path = Path(path)
    with open(path, 'w') as f:
        json.dump(params, f, indent=2)


def load_params(path: Union[str, Path]) -> dict:
    """Load parameters from JSON file."""
    path = Path(path)
    with open(path, 'r') as f:
        return json.load(f)


def save_psd(freqs: np.ndarray, psd: np.ndarray, path: Union[str, Path]):
    """Save PSD to npz file."""
    np.savez(str(path), freqs=freqs, psd=psd)


def load_psd(path: Union[str, Path]) -> tuple:
    """Load PSD from npz file. Returns (freqs, psd)."""
    data = np.load(str(path))
    return data['freqs'], data['psd']
