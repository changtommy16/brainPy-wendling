"""
Basic Plotting Utilities for Wendling Model

Provides simple visualization functions:
- plot_lfp: Time series plot of LFP signal
- plot_psd: Power spectral density (FFT-based)
- plot_results: Combined time series + PSD plot

Usage:
    from core_wendling.plot_basic import plot_results
    plot_results(results, save_path='./results/wendling_plot.png')
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from pathlib import Path


# =============================================================================
# Time Series Plot
# =============================================================================

def plot_lfp(
    t: np.ndarray,
    lfp: np.ndarray,
    stim: Optional[np.ndarray] = None,
    title: str = "LFP Time Series",
    xlim: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot LFP time series.
    
    Args:
        t: Time array (ms)
        lfp: LFP signal array
        stim: Optional stimulus array to overlay
        title: Plot title
        xlim: X-axis limits (ms)
        ax: Matplotlib axes (creates new if None)
        show: Whether to show plot immediately
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot LFP
    ax.plot(t, lfp, 'b-', linewidth=0.5, label='LFP')
    
    # Plot stimulus if provided
    if stim is not None:
        # Normalize stimulus for visualization
        stim_normalized = stim / (np.max(np.abs(stim)) + 1e-10) * np.std(lfp) * 2
        ax.plot(t, stim_normalized, 'r-', alpha=0.3, linewidth=0.5, label='Stimulus (scaled)')
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('LFP (mV)')
    ax.set_title(title)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    if stim is not None:
        ax.legend(loc='upper right')
    
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


# =============================================================================
# Power Spectral Density
# =============================================================================

def plot_psd(
    lfp: np.ndarray,
    dt_ms: float = 0.1,
    f_range: Tuple[float, float] = (0, 100),
    title: str = "Power Spectral Density",
    ax: Optional[plt.Axes] = None,
    show: bool = True
) -> Tuple[plt.Axes, np.ndarray, np.ndarray]:
    """
    Plot power spectral density using FFT.
    
    Args:
        lfp: LFP signal array
        dt_ms: Time step (ms)
        f_range: Frequency range to display (Hz)
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        show: Whether to show plot immediately
    
    Returns:
        Tuple of (axes, frequencies, power)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Compute FFT
    n = len(lfp)
    fs = 1000.0 / dt_ms  # Sampling frequency (Hz)
    
    # FFT
    fft_vals = np.fft.rfft(lfp)
    freqs = np.fft.rfftfreq(n, d=dt_ms/1000.0)
    
    # Power (normalized)
    power = np.abs(fft_vals)**2 / n
    
    # Filter to frequency range
    mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    
    # Plot
    ax.semilogy(freqs[mask], power[mask], 'b-', linewidth=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (a.u.)')
    ax.set_title(title)
    ax.set_xlim(f_range)
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax, freqs, power


# =============================================================================
# Combined Plot
# =============================================================================

def plot_results(
    results: Dict[str, np.ndarray],
    dt_ms: float = 0.1,
    title: str = "Wendling Model Simulation",
    save_path: Optional[str] = None,
    show: bool = True,
    time_window: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """
    Create combined plot with LFP time series and PSD.
    
    Args:
        results: Dictionary with 't', 'lfp', and optionally 'stim'
        dt_ms: Time step (ms)
        title: Overall figure title
        save_path: Path to save figure (optional)
        show: Whether to show plot
        time_window: Optional time window to display (ms), e.g., (1000, 2000)
    
    Returns:
        Matplotlib figure
    """
    t = results['t']
    lfp = results['lfp']
    stim = results.get('stim', None)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)
    
    # Time series plot
    xlim = time_window if time_window else (t[0], min(t[-1], 5000))  # Show first 5s by default
    plot_lfp(t, lfp, stim=stim, title="LFP Time Series", xlim=xlim, ax=axes[0], show=False)
    
    # PSD plot
    plot_psd(lfp, dt_ms=dt_ms, f_range=(0, 100), title="Power Spectral Density", ax=axes[1], show=False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[plot_basic] Figure saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# Multi-Frequency Comparison (Optional)
# =============================================================================

def plot_assr_comparison(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    dt_ms: float = 0.1,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare PSDs for different ASSR frequencies.
    
    Args:
        results_dict: Dictionary mapping frequency labels to results
                     e.g., {'20 Hz': results_20, '40 Hz': results_40}
        dt_ms: Time step (ms)
        save_path: Path to save figure
        show: Whether to show plot
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results_dict)))
    
    for (label, results), color in zip(results_dict.items(), colors):
        lfp = results['lfp']
        n = len(lfp)
        
        fft_vals = np.fft.rfft(lfp)
        freqs = np.fft.rfftfreq(n, d=dt_ms/1000.0)
        power = np.abs(fft_vals)**2 / n
        
        mask = (freqs >= 0) & (freqs <= 100)
        ax.semilogy(freqs[mask], power[mask], color=color, linewidth=1, label=label)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (a.u.)')
    ax.set_title('ASSR Frequency Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[plot_basic] Figure saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# Quick Plot from File
# =============================================================================

def plot_from_file(
    filepath: str,
    dt_ms: float = 0.1,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Load results from file and create plot.
    
    Args:
        filepath: Path to .npz results file
        dt_ms: Time step (ms)
        save_path: Path to save figure
        show: Whether to show plot
    
    Returns:
        Matplotlib figure
    """
    data = np.load(filepath)
    results = {key: data[key] for key in data.files}
    
    return plot_results(results, dt_ms=dt_ms, save_path=save_path, show=show)


# =============================================================================
# Main (for standalone use)
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot Wendling simulation results')
    parser.add_argument('filepath', type=str, help='Path to .npz results file')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step (ms)')
    parser.add_argument('--save', type=str, default=None, help='Path to save figure')
    
    args = parser.parse_args()
    
    plot_from_file(args.filepath, dt_ms=args.dt, save_path=args.save)
