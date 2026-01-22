#!/usr/bin/env python
"""
Run nevergrad optimization.

Usage:
    python scripts/run_optimize.py --target-type Type4 --budget 50
    python scripts/run_optimize.py --config configs/optimize_default.yaml
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import yaml
import json
import numpy as np
from datetime import datetime

from wendling_sim import simulate, optimize
from wendling_sim.model.params import TYPE_PARAMS, get_type_params
from wendling_sim.features.psd import compute_psd
from wendling_sim.optimize.search_space import SearchSpace
from wendling_sim.utils.io import save_psd, save_params


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def generate_target_psd(type_name: str, sim_cfg: dict, psd_cfg: dict) -> tuple:
    """Generate target PSD from known activity type."""
    type_params = get_type_params(type_name)
    
    # Run simulation with type params
    result = simulate(
        sim_cfg=sim_cfg,
        model_cfg=type_params,
        noise_cfg={'mode': 'deterministic', 'seed': 42},
    )
    
    # Compute PSD
    if 'dt_s' in sim_cfg:
        fs = 1.0 / sim_cfg['dt_s']
    else:
        fs = 1000.0 / sim_cfg.get('dt_ms', 0.1)
    lfp = np.asarray(result.lfp)
    psd_result = compute_psd(
        lfp[:, 0] if lfp.ndim > 1 else lfp,
        fs=fs,
        nperseg=psd_cfg.get('nperseg', None),
        freq_range=psd_cfg.get('freq_range', (1.0, 50.0)),
    )
    
    return psd_result.freqs, psd_result.psd


def main():
    parser = argparse.ArgumentParser(description='Run Wendling optimization')
    parser.add_argument('--config', type=str, help='Config YAML file')
    parser.add_argument('--target-type', type=str, choices=list(TYPE_PARAMS.keys()),
                        help='Target activity type')
    parser.add_argument('--target-psd', type=str, help='Target PSD npz file')
    parser.add_argument('--budget', type=int, default=50, help='Optimization budget')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='runs/opt', help='Output directory')
    args = parser.parse_args()
    
    # Load config or use defaults
    if args.config:
        cfg = load_config(args.config)
        opt_cfg = cfg.get('optimization', {})
        objective_cfg = cfg.get('objective', {})
        target_cfg = cfg.get('target', {})
    else:
        opt_cfg = {
            'budget': args.budget,
            'optimizer': 'NGOpt',
            'seed': args.seed,
        }
        objective_cfg = {
            'sim_cfg': {'dt_ms': 0.1, 'duration_ms': 5000.0, 'jit': True},
            'psd_cfg': {'nperseg': 4096, 'freq_range': (1.0, 50.0)},
            'loss_name': 'psd_mse',
        }
        target_cfg = {}
    
    # Get target PSD
    if args.target_psd:
        data = np.load(args.target_psd)
        target_freqs, target_psd = data['freqs'], data['psd']
        print(f"Loaded target PSD from {args.target_psd}")
    elif args.target_type:
        print(f"Generating target PSD from {args.target_type}...")
        target_freqs, target_psd = generate_target_psd(
            args.target_type,
            objective_cfg.get('sim_cfg', {}),
            objective_cfg.get('psd_cfg', {}),
        )
        print("[OK] Target PSD generated")
    else:
        raise ValueError("Must specify --target-type or --target-psd")
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save target PSD
    save_psd(target_freqs, target_psd, output_dir / 'target_psd.npz')
    
    # Run optimization
    print(f"\nStarting optimization (budget={opt_cfg.get('budget', args.budget)})...")
    print("-" * 50)
    
    result = optimize(
        opt_cfg=opt_cfg,
        objective_cfg=objective_cfg,
        target_psd=target_psd,
        target_freqs=target_freqs,
        verbose=True,
    )
    
    print("-" * 50)
    print("\n[OK] Optimization complete")
    print(f"  Best loss: {result.best_loss:.6f}")
    print(f"  Best params: {result.best_params}")
    print(f"  Elapsed: {result.meta.get('elapsed_s', 0):.1f}s")
    
    # Save results
    save_params(result.best_params, output_dir / 'best_params.json')
    
    # Save history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump([{'params': p, 'loss': l} for p, l in result.history], f, indent=2)
    
    print(f"\n[OK] Results saved to {output_dir}")


if __name__ == '__main__':
    main()
