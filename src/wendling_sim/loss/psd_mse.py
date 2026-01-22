"""
PSD-based loss functions for optimization.
"""

import numpy as np
from typing import Optional, Callable


def psd_mse_loss(
    psd: np.ndarray,
    target_psd: np.ndarray,
    freqs: Optional[np.ndarray] = None,
    freq_range: Optional[tuple] = None,
) -> float:
    """
    Mean squared error between PSD curves.
    
    Args:
        psd: Computed PSD array
        target_psd: Target PSD array
        freqs: Frequency array (for freq_range filtering)
        freq_range: Optional (low, high) to restrict comparison
    
    Returns:
        MSE loss value
    """
    if freq_range is not None and freqs is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        psd = psd[mask]
        target_psd = target_psd[mask]
    
    return float(np.mean((psd - target_psd) ** 2))


def weighted_psd_mse_loss(
    psd: np.ndarray,
    target_psd: np.ndarray,
    freqs: np.ndarray,
    weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    roi_band: Optional[tuple] = None,
    roi_weight: float = 10.0,
) -> float:
    """
    Weighted mean squared error between PSD curves.
    
    Args:
        psd: Computed PSD array
        target_psd: Target PSD array
        freqs: Frequency array
        weight_fn: Custom weight function f(freqs) -> weights
        roi_band: Region of interest (low, high) Hz for emphasis
        roi_weight: Weight multiplier for ROI band
    
    Returns:
        Weighted MSE loss value
    """
    # Compute weights
    if weight_fn is not None:
        weights = weight_fn(freqs)
    elif roi_band is not None:
        weights = np.ones_like(freqs)
        roi_mask = (freqs >= roi_band[0]) & (freqs <= roi_band[1])
        weights[roi_mask] = roi_weight
    else:
        weights = np.ones_like(freqs)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    return float(np.sum(weights * (psd - target_psd) ** 2))


def log_psd_mse_loss(
    psd: np.ndarray,
    target_psd: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """
    MSE in log-space (useful for matching spectral shape).
    
    Args:
        psd: Computed PSD array
        target_psd: Target PSD array
        epsilon: Small value to avoid log(0)
    
    Returns:
        Log-space MSE loss
    """
    log_psd = np.log10(psd + epsilon)
    log_target = np.log10(target_psd + epsilon)
    return float(np.mean((log_psd - log_target) ** 2))
