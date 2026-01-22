#!/usr/bin/env python
"""
Test optimization by fitting to a known PSD target.

This test:
1. Generates a target PSD from Type4 (alpha-like rhythm)
2. Starts optimization from different initial parameters
3. Fits to the target PSD using nevergrad
4. Visualizes and saves results

Usage:
    conda activate brainpy_model
    python tests/test_optimization.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.math as bm

from wendling_sim import simulate, optimize
from wendling_sim.model.params import TYPE_PARAMS, get_type_params, STANDARD_PARAMS
from wendling_sim.features.psd import compute_psd, psd_to_db
from wendling_sim.optimize.search_space import SearchSpace


def setup_results_dir():
    """Create timestamped results directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = PROJECT_ROOT / "results" / f"opt_test_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def generate_target_psd(type_name: str, sim_cfg: dict):
    """Generate target PSD from known activity type."""
    params = get_type_params(type_name)

    result = simulate(
        sim_cfg=sim_cfg,
        model_cfg=params,
        noise_cfg={'mode': 'deterministic', 'seed': 42},
    )

    fs = 1000.0 / sim_cfg['dt_ms']
    lfp = np.asarray(result.lfp)
    psd_result = compute_psd(
        lfp[:, 0] if lfp.ndim > 1 else lfp,
        fs=fs,
        nperseg=4096,
        freq_range=(1, 50),
    )

    return psd_result.freqs, psd_result.psd, params


def run_optimization_test(
    target_type: str = 'Type4',
    budget: int = 30,
    duration_ms: float = 5000.0,
):
    """
    Run optimization test fitting to a known activity type.

    Args:
        target_type: Activity type to fit (Type1-Type6)
        budget: Number of optimization evaluations
        duration_ms: Simulation duration

    Returns:
        dict with results
    """
    print(f"\n{'='*60}")
    print(f"Optimization Test: Fitting to {target_type}")
    print(f"{'='*60}")

    # Setup
    results_dir = setup_results_dir()
    sim_cfg = {'dt_ms': 0.1, 'duration_ms': duration_ms, 'jit': True}
    fs = 1000.0 / sim_cfg['dt_ms']

    # Generate target PSD
    print(f"\n[1/4] Generating target PSD from {target_type}...")
    target_freqs, target_psd, true_params = generate_target_psd(target_type, sim_cfg)
    print(f"  True params: A={true_params['A']}, B={true_params['B']}, G={true_params['G']}")

    # Save target PSD
    np.savez(results_dir / 'target_psd.npz', freqs=target_freqs, psd=target_psd)

    # Define search space (narrower for faster convergence in test)
    search_space = SearchSpace(
        bounds={
            'A': (2.0, 8.0),
            'B': (5.0, 30.0),
            'G': (5.0, 30.0),
        },
        log_scale=set(),
        fixed={},  # Fix other params to defaults
    )

    # Run optimization
    print(f"\n[2/4] Running optimization (budget={budget})...")

    opt_cfg = {
        'budget': budget,
        'optimizer': 'NGOpt',
        'seed': 123,
        'search_space': search_space,
    }

    objective_cfg = {
        'sim_cfg': sim_cfg,
        'psd_cfg': {'nperseg': 4096, 'freq_range': (1, 50)},
        'loss_name': 'psd_mse',
    }

    opt_result = optimize(
        opt_cfg=opt_cfg,
        objective_cfg=objective_cfg,
        target_psd=target_psd,
        target_freqs=target_freqs,
        verbose=True,
    )

    print(f"\n  Best loss: {opt_result.best_loss:.6f}")
    print(f"  Best params: {opt_result.best_params}")
    print(f"  Elapsed: {opt_result.meta.get('elapsed_s', 0):.1f}s")

    # Generate PSD with optimized params
    print(f"\n[3/4] Generating PSD with optimized parameters...")
    opt_params = {**STANDARD_PARAMS, **opt_result.best_params}

    opt_sim_result = simulate(
        sim_cfg=sim_cfg,
        model_cfg=opt_params,
        noise_cfg={'mode': 'deterministic', 'seed': 42},
    )

    opt_lfp = np.asarray(opt_sim_result.lfp)
    opt_psd_result = compute_psd(
        opt_lfp[:, 0] if opt_lfp.ndim > 1 else opt_lfp,
        fs=fs,
        nperseg=4096,
        freq_range=(1, 50),
    )

    # Create visualizations
    print(f"\n[4/4] Creating visualizations...")

    # Figure 1: PSD comparison
    fig1, gs1 = bp.visualize.get_figure(2, 2, row_len=3.2, col_len=6)
    axes1 = [[fig1.add_subplot(gs1[i, j]) for j in range(2)] for i in range(2)]

    # Target vs Optimized PSD (log scale)
    bp.visualize.line_plot(
        target_freqs,
        target_psd,
        ax=axes1[0][0],
        xlabel='Frequency (Hz)',
        ylabel='PSD',
        colors='b',
        linewidth=2.0,
        label=f'Target ({target_type})',
    )
    bp.visualize.line_plot(
        opt_psd_result.freqs,
        opt_psd_result.psd,
        ax=axes1[0][0],
        colors='r',
        linewidth=2.0,
        linestyle='--',
        label='Optimized',
    )
    axes1[0][0].set_yscale('log')
    axes1[0][0].set_xlim(0, 50)
    axes1[0][0].set_title('PSD Comparison (log scale)')
    axes1[0][0].grid(alpha=0.3)

    # PSD in dB
    bp.visualize.line_plot(
        target_freqs,
        psd_to_db(target_psd),
        ax=axes1[0][1],
        xlabel='Frequency (Hz)',
        ylabel='PSD (dB)',
        colors='b',
        linewidth=2.0,
        label=f'Target ({target_type})',
    )
    bp.visualize.line_plot(
        opt_psd_result.freqs,
        psd_to_db(opt_psd_result.psd),
        ax=axes1[0][1],
        colors='r',
        linewidth=2.0,
        linestyle='--',
        label='Optimized',
    )
    axes1[0][1].set_xlim(0, 50)
    axes1[0][1].set_title('PSD Comparison (dB scale)')
    axes1[0][1].grid(alpha=0.3)

    # Loss history
    losses = np.asarray([h[1] for h in opt_result.history])
    eval_ids = np.arange(len(losses))
    bp.visualize.line_plot(
        eval_ids,
        losses,
        ax=axes1[1][0],
        xlabel='Evaluation',
        ylabel='Loss',
        title='Optimization Progress',
        colors='b',
        linewidth=1.5,
    )
    axes1[1][0].axhline(opt_result.best_loss, color='r', linestyle='--', label=f'Best: {opt_result.best_loss:.4f}')
    axes1[1][0].legend()
    axes1[1][0].grid(alpha=0.3)

    # Parameter convergence
    param_names = list(opt_result.best_params.keys())
    colors = list(plt.cm.tab10(np.linspace(0, 1, len(param_names))))
    for param, color in zip(param_names, colors):
        values = np.asarray([h[0].get(param, np.nan) for h in opt_result.history])
        bp.visualize.line_plot(
            eval_ids,
            values,
            ax=axes1[1][1],
            xlabel='Evaluation',
            ylabel='Parameter Value',
            title='Parameter Convergence (dashed = true)',
            color=color,
            linewidth=1.0,
            label=param,
        )
        axes1[1][1].axhline(true_params.get(param, np.nan), color=color, linestyle='--', alpha=0.5)
    axes1[1][1].legend(loc='upper right')
    axes1[1][1].grid(alpha=0.3)

    fig1.suptitle(f'Optimization Test: Fitting to {target_type} (budget={budget})', fontsize=14)
    plt.tight_layout()
    fig1.savefig(results_dir / 'optimization_results.png', dpi=150, bbox_inches='tight')
    print("  Saved: optimization_results.png")

    # Figure 2: Time series comparison
    fig2, gs2 = bp.visualize.get_figure(2, 1, row_len=2.6, col_len=7)
    axes2 = [fig2.add_subplot(gs2[i, 0]) for i in range(2)]

    # Generate target time series
    target_result = simulate(
        sim_cfg=sim_cfg,
        model_cfg=true_params,
        noise_cfg={'mode': 'deterministic', 'seed': 42},
    )
    target_lfp = np.asarray(target_result.lfp)
    opt_lfp = np.asarray(opt_sim_result.lfp)

    # Show 2 seconds from middle
    t_ms = target_result.t_s * 1000.0
    t_mid = len(t_ms) // 2
    t_window = int(2000 / sim_cfg['dt_ms'])
    t_slice = slice(t_mid - t_window // 2, t_mid + t_window // 2)

    bp.visualize.line_plot(
        t_ms[t_slice],
        target_lfp[t_slice, 0] if target_lfp.ndim > 1 else target_lfp[t_slice],
        ax=axes2[0],
        xlabel='Time (ms)',
        ylabel='LFP (mV)',
        title=f"Target: {target_type} (A={true_params['A']}, B={true_params['B']}, G={true_params['G']})",
        colors='b',
        linewidth=0.8,
    )
    axes2[0].grid(alpha=0.3)

    best_params = opt_result.best_params
    bp.visualize.line_plot(
        t_ms[t_slice],
        opt_lfp[t_slice, 0] if opt_lfp.ndim > 1 else opt_lfp[t_slice],
        ax=axes2[1],
        xlabel='Time (ms)',
        ylabel='LFP (mV)',
        title=f"Optimized: A={best_params.get('A', '?'):.2f}, B={best_params.get('B', '?'):.2f}, G={best_params.get('G', '?'):.2f}",
        colors='r',
        linewidth=0.8,
    )
    axes2[1].grid(alpha=0.3)

    fig2.suptitle('Time Series Comparison (2s window)', fontsize=14)
    plt.tight_layout()
    fig2.savefig(results_dir / 'time_series_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: time_series_comparison.png")

    # Save results summary
    summary = {
        'target_type': target_type,
        'true_params': true_params,
        'best_params': opt_result.best_params,
        'best_loss': opt_result.best_loss,
        'budget': budget,
        'duration_ms': duration_ms,
        'elapsed_s': opt_result.meta.get('elapsed_s', 0),
        'param_errors': {
            k: abs(opt_result.best_params.get(k, 0) - true_params.get(k, 0))
            for k in opt_result.best_params.keys()
        },
    }

    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print("  Saved: summary.json")

    # Save full history
    history_data = [{'params': p, 'loss': l} for p, l in opt_result.history]
    with open(results_dir / 'history.json', 'w') as f:
        json.dump(history_data, f, indent=2, default=float)
    print("  Saved: history.json")

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")

    # Print summary
    print("\nSummary:")
    print(f"  Target: {target_type}")
    print(f"  True params:  A={true_params['A']:.2f}, B={true_params['B']:.2f}, G={true_params['G']:.2f}")
    print(f"  Found params: A={best_params.get('A', 0):.2f}, B={best_params.get('B', 0):.2f}, G={best_params.get('G', 0):.2f}")
    print(f"  Final loss: {opt_result.best_loss:.6f}")

    return summary, results_dir


if __name__ == '__main__':
    # Try to use GPU
    try:
        bm.set_platform('gpu')
        print("[OK] GPU enabled")
    except Exception as exc:
        print(f"[WARN] GPU not available: {exc}")
        bm.set_platform('cpu')

    # Run test with Type4 (alpha-like rhythm)
    summary, results_dir = run_optimization_test(
        target_type='Type4',
        budget=30,
        duration_ms=5000.0,
    )

    print("\nTest completed successfully!")
