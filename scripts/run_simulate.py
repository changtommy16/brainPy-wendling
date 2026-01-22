#!/usr/bin/env python
"""
Run forward simulation.

Usage:
    python scripts/run_simulate.py
    python scripts/run_simulate.py --config configs/sim_default.yaml
    python scripts/run_simulate.py --type Type5 --duration 10000
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp

from wendling_sim import simulate
from wendling_sim.model.params import TYPE_PARAMS, get_type_params
from wendling_sim.features.psd import compute_psd


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Run Wendling simulation')
    parser.add_argument('--config', type=str, help='Config YAML file')
    parser.add_argument('--type', type=str, choices=list(TYPE_PARAMS.keys()),
                        help='Activity type (Type1-Type6)')
    parser.add_argument('--duration', type=float, default=5000.0,
                        help='Duration in ms')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, help='Output npz file')
    parser.add_argument('--plot', action='store_true', help='Show plot')
    args = parser.parse_args()
    
    # Load config or use defaults
    if args.config:
        cfg = load_config(args.config)
        sim_cfg = cfg.get('simulation', {})
        model_cfg = cfg.get('model', {})
        noise_cfg = cfg.get('noise', {})
        monitor_cfg = cfg.get('monitor', {})
    else:
        sim_cfg = {'dt_ms': 0.1, 'duration_ms': args.duration, 'jit': True}
        model_cfg = {}
        noise_cfg = {'seed': args.seed, 'p_mean': 90.0, 'p_sigma': 2.0}
        monitor_cfg = {}
    
    # Override with activity type if specified
    if args.type:
        type_params = get_type_params(args.type)
        model_cfg.update(type_params)
        print(f"Using {args.type} parameters: {type_params}")
    
    # Run simulation
    print(f"Running simulation: {sim_cfg.get('duration_ms', args.duration)} ms")
    result = simulate(sim_cfg=sim_cfg, model_cfg=model_cfg, noise_cfg=noise_cfg, monitor_cfg=monitor_cfg)
    
    print("Simulation complete")
    lfp = np.asarray(result.lfp)
    print(f"  LFP shape: {lfp.shape}")
    print(f"  LFP range: [{lfp.min():.2f}, {lfp.max():.2f}] mV")
    
    # Compute PSD
    fs = 1000.0 / result.meta['dt_ms']
    psd_result = compute_psd(lfp[:, 0] if lfp.ndim > 1 else lfp, fs=fs)
    print(f"  Peak frequency: {psd_result.peak_freq:.1f} Hz")
    
    # Save result
    if args.output:
        result.save(args.output)
        print(f"Saved to {args.output}")
    
    # Plot
    if args.plot:
        fig, gs = bp.visualize.get_figure(2, 1, row_len=2.6, col_len=6)
        axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
        
        # Time series (show 2 seconds from middle)
        t_ms = result.t_s * 1000.0
        t_mid = len(t_ms) // 2
        t_window = int(2000 / result.meta['dt_ms'])
        t_slice = slice(t_mid - t_window//2, t_mid + t_window//2)
        
        bp.visualize.line_plot(
            t_ms[t_slice],
            lfp[t_slice, 0] if lfp.ndim > 1 else lfp[t_slice],
            ax=axes[0],
            xlabel='Time (ms)',
            ylabel='LFP (mV)',
            title='LFP Time Series (2s window)',
            linewidth=0.8,
        )
        axes[0].grid(alpha=0.3)
        
        # PSD
        bp.visualize.line_plot(
            psd_result.freqs,
            psd_result.psd,
            ax=axes[1],
            xlabel='Frequency (Hz)',
            ylabel='PSD',
            title='Power Spectral Density',
            linewidth=1.0,
        )
        axes[1].set_yscale('log')
        axes[1].axvline(psd_result.peak_freq, color='r', linestyle='--',
                       label=f'Peak: {psd_result.peak_freq:.1f} Hz')
        axes[1].set_xlim(0, 50)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
