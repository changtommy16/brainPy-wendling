#!/usr/bin/env python
"""
Run all 6 Wendling activity types and plot results.

Usage:
    conda activate brainpy_model
    python scripts/run_six_types.py
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.math as bm

from wendling_sim import simulate
from wendling_sim.model.params import TYPE_PARAMS, get_type_params
from wendling_sim.features.psd import compute_psd, psd_to_db

# Try GPU
try:
    bm.set_platform('gpu')
    print("[OK] GPU enabled")
except Exception as exc:
    print(f"[WARN] GPU not available: {exc}")
    bm.set_platform('cpu')

# Simulation config
DURATION_MS = 30000
DT_MS = 0.1
FS = 1000.0 / DT_MS

# Setup figure (BrainPy visualize)
fig, gs = bp.visualize.get_figure(6, 2, row_len=2.2, col_len=6)
axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(6)]
fig.suptitle('Wendling Model: All 6 Activity Types (30s simulations)', fontsize=14, y=0.995)

print("\nRunning simulations for all 6 Wendling types...")
print("="*70)

for idx, type_name in enumerate(TYPE_PARAMS.keys()):
    print(f"\n[{idx+1}/6] {type_name}...")

    # Get parameters and info
    type_info = TYPE_PARAMS[type_name]
    params = get_type_params(type_name)
    seed = idx * 42

    # Run simulation using new API
    result = simulate(
        sim_cfg={'dt_ms': DT_MS, 'duration_ms': DURATION_MS, 'jit': True},
        model_cfg=params,
        noise_cfg={'seed': seed, 'p_mean': params['p_mean'], 'p_sigma': params['p_sigma']},
    )
    
    t_ms = result.t_s * 1000.0
    lfp = np.asarray(result.lfp)
    lfp_trace = lfp[:, 0] if lfp.ndim > 1 else lfp

    print(f"  LFP range: [{lfp.min():.2f}, {lfp.max():.2f}] mV")
    print(f"  Seed: {seed}")

    # Compute PSD
    psd_result = compute_psd(lfp_trace, fs=FS, nperseg=8192, freq_range=(1, 50))
    f, psd = psd_result.freqs, psd_result.psd
    dominant_freq = psd_result.peak_freq
    
    print(f"  Dominant frequency: {dominant_freq:.2f} Hz")

    # For Type5, also check peak from 2 Hz and above
    if type_name == 'Type5':
        mask_2hz = (f >= 2) & (f <= 50)
        dominant_freq_2hz = f[mask_2hz][np.argmax(psd[mask_2hz])]
        print(f"  Dominant frequency (>=2Hz): {dominant_freq_2hz:.2f} Hz")

    # Convert PSD to dB
    psd_db = psd_to_db(psd)

    # Plot time series (show middle 5 seconds)
    t_start_idx = int(12500 / DT_MS)
    t_end_idx = int(17500 / DT_MS)
    bp.visualize.line_plot(
        t_ms[t_start_idx:t_end_idx],
        lfp_trace[t_start_idx:t_end_idx],
        ax=axes[idx][0],
        xlabel='Time (ms)',
        ylabel='LFP (mV)',
        title=f"{type_info['name']}",
        linewidth=0.8,
    )
    axes[idx][0].grid(alpha=0.3)

    # Add parameter text
    param_text = f"A={params['A']:.1f}, B={params['B']:.1f}, G={params['G']:.1f}\nSeed={seed}"
    axes[idx][0].text(0.02, 0.98, param_text, transform=axes[idx][0].transAxes,
                      verticalalignment='top', fontsize=7,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Plot PSD in dB
    bp.visualize.line_plot(
        f,
        psd_db,
        ax=axes[idx][1],
        xlabel='Frequency (Hz)',
        ylabel='PSD (dB)',
        linewidth=1.0,
    )
    axes[idx][1].set_xlim(0, 50)
    axes[idx][1].grid(alpha=0.3)

    # Add dominant frequency to PSD plot title
    exp_range = type_info['expected']['freq_range']
    axes[idx][1].set_title(f"Peak: {dominant_freq:.1f} Hz (Expected: {exp_range[0]}-{exp_range[1]} Hz)", fontsize=8)

    # Highlight expected frequency range
    axes[idx][1].axvspan(exp_range[0], exp_range[1], alpha=0.2, color='red', label='Expected range')
    axes[idx][1].axvline(dominant_freq, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                        label=f'Peak: {dominant_freq:.1f} Hz')

    if idx == 0:
        axes[idx][1].legend(fontsize=7, loc='upper right')

# Labels
axes[-1][0].set_xlabel('Time (ms)', fontsize=10)
axes[-1][1].set_xlabel('Frequency (Hz)', fontsize=10)

plt.tight_layout()
plt.savefig('all_activity_types_detailed.png', dpi=200, bbox_inches='tight')
print("\n" + "="*70)
print("Generated all_activity_types_detailed.png")
print(f"Duration: {DURATION_MS/1000:.0f} seconds per simulation")
print(f"PSD resolution: ~{FS/8192:.2f} Hz (nperseg=8192)")
