"""
PSD feature extraction using Welch method.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Sequence
import numpy as np
from scipy import signal


@dataclass
class PsdResult:
    """
    PSD computation result.
    
    Attributes:
        freqs: Frequency array (Hz)
        psd: Power spectral density array (F, N)
        peak_freq: Dominant frequency (average across nodes)
        bandpower: Optional dict of bandpower values
    """
    freqs: np.ndarray
    psd: np.ndarray
    peak_freq: float
    bandpower: Optional[Dict[str, np.ndarray]] = None


def compute_psd(
    lfp: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    freq_range: Tuple[float, float] = (1.0, 50.0),
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    detrend: str = 'constant',
    roi: str = 'none',
    roi_nodes: Optional[Sequence[int]] = None,
) -> PsdResult:
    """
    Compute PSD using Welch method (time-major input).
    
    Args:
        lfp: LFP signal array (T,) or (T, N)
        fs: Sampling frequency (Hz)
        nperseg: Samples per segment (default: ~2 seconds)
        noverlap: Overlap samples (default: 50% of nperseg)
        window: Window function ('hann', 'hamming', etc.)
        freq_range: Frequency range for peak detection
        bands: Optional dict of frequency bands for bandpower
        detrend: Detrend method ('constant', 'linear', False)
        roi: 'none' | 'mean' | 'subset'
        roi_nodes: Node indices used when roi='subset'
    
    Returns:
        PsdResult with freqs, psd, peak_freq, bandpower
    """
    lfp_arr = np.asarray(lfp)
    if lfp_arr.ndim == 1:
        lfp_arr = lfp_arr[:, None]

    # ROI aggregation
    roi = (roi or 'none').lower()
    if roi == 'mean':
        lfp_arr = lfp_arr.mean(axis=1, keepdims=True)
    elif roi == 'subset' and roi_nodes is not None:
        lfp_arr = lfp_arr[:, list(roi_nodes)]

    # Default nperseg: ~2 seconds of data
    if nperseg is None:
        nperseg = min(int(2.0 * fs), lfp_arr.shape[0])

    # Default overlap: 50%
    if noverlap is None:
        noverlap = nperseg // 2

    # Compute Welch PSD (time axis = 0)
    freqs, psd = signal.welch(
        lfp_arr,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        axis=0,
    )

    # Find peak frequency (average across nodes)
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if np.any(mask):
        psd_masked = psd[mask]
        peak_idx = np.argmax(psd_masked, axis=0)
        peak_freqs = freqs[mask][peak_idx]
        peak_freq = float(np.mean(peak_freqs))
    else:
        peak_freq = 0.0

    # Compute bandpower if bands specified
    bandpower = None
    if bands:
        bandpower = {}
        for name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if np.any(band_mask):
                bandpower[name] = np.trapz(psd[band_mask], freqs[band_mask], axis=0)
            else:
                bandpower[name] = np.zeros(psd.shape[1], dtype=float)

    return PsdResult(
        freqs=freqs,
        psd=psd,
        peak_freq=peak_freq,
        bandpower=bandpower,
    )


def psd_to_db(psd: np.ndarray, ref: float = 1.0) -> np.ndarray:
    """Convert PSD to decibels."""
    return 10.0 * np.log10(psd / ref + 1e-12)
