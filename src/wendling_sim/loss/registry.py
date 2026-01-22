"""
Loss function registry for dynamic loss selection.
"""

from typing import Dict, Any, Callable
import numpy as np

from wendling_sim.loss.psd_mse import (
    psd_mse_loss,
    weighted_psd_mse_loss,
    log_psd_mse_loss,
)


# Registry of available loss functions
_LOSS_REGISTRY: Dict[str, Callable] = {
    'psd_mse': psd_mse_loss,
    'weighted_psd_mse': weighted_psd_mse_loss,
    'log_psd_mse': log_psd_mse_loss,
}


def register_loss(name: str, fn: Callable):
    """Register a custom loss function."""
    _LOSS_REGISTRY[name] = fn


def list_losses() -> list:
    """List available loss function names."""
    return list(_LOSS_REGISTRY.keys())


def build_loss(name: str, cfg: Dict[str, Any] = None) -> Callable:
    """
    Build a loss function with configuration.
    
    Args:
        name: Loss function name
        cfg: Configuration dict passed to loss function
    
    Returns:
        Configured loss function: f(psd, target_psd, freqs) -> float
    """
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list_losses()}")
    
    base_fn = _LOSS_REGISTRY[name]
    cfg = cfg or {}
    
    def loss_fn(psd: np.ndarray, target_psd: np.ndarray, freqs: np.ndarray) -> float:
        return base_fn(psd, target_psd, freqs=freqs, **cfg)
    
    return loss_fn
