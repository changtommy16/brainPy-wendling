"""
Noise generation for deterministic/stochastic simulation modes.

Deterministic mode: pre-generate noise sequence with fixed seed.
Stochastic mode: generate noise on-the-fly during simulation.
"""

import numpy as np
from typing import Optional


def create_noise_sequence(
    n_steps: int,
    seed: Optional[int] = None,
    distribution: str = 'normal',
) -> np.ndarray:
    """
    Create pre-generated noise sequence for deterministic evaluation.
    
    Args:
        n_steps: Number of time steps
        seed: Random seed for reproducibility
        distribution: 'normal' (default) or 'uniform'
    
    Returns:
        Noise array of shape (n_steps,)
    """
    rng = np.random.default_rng(seed)
    
    if distribution == 'normal':
        return rng.standard_normal(n_steps)
    elif distribution == 'uniform':
        return rng.uniform(-1.0, 1.0, n_steps)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def get_noise_length(duration_ms: float, dt_ms: float, buffer: float = 1.1) -> int:
    """
    Calculate required noise sequence length with buffer.
    
    Args:
        duration_ms: Simulation duration
        dt_ms: Time step
        buffer: Safety buffer multiplier (default 1.1 = 10% extra)
    
    Returns:
        Required noise sequence length
    """
    return int(np.ceil(duration_ms / dt_ms * buffer))
