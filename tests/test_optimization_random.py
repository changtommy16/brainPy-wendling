#!/usr/bin/env python
"""
Random PSD optimization test (no stimulus).

This test:
1. Samples a random parameter set from the SearchSpace (A,B,G,a,b,g,G_net)
2. Generates a target PSD from that parameter set (no external stimulus, only noise)
3. Uses nevergrad optimization to fit the PSD starting from the search space
4. Visualizes target vs fitted PSD + time series, with parameters on the figures
5. Saves all artifacts under results/random_opt_YYYYMMDD_HHMMSS/

Usage:
    conda activate brainpy_model
    python tests/test_optimization_random.py
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
from wendling_sim.model.params import STANDARD_PARAMS
from wendling_sim.features.psd import compute_psd, psd_to_db
from wendling_sim.optimize.search_space import SearchSpace


def setup_results_dir(prefix: str = "random_opt") -> Path:
    """Create timestamped results directory under results/."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = PROJECT_ROOT / "results" / f"{prefix}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def generate_random_target_psd(sim_cfg: dict, seed: int = 42):
    """Generate a random target PSD from SearchSpace.default().

    Returns:
        freqs, psd, true_opt_params, true_full_params, search_space, sim_result
    """
    search_space = SearchSpace.default()
    rng = np.random.default_rng(seed)
    true_opt_params = search_space.sample_random(rng=rng)

    true_full_params = dict(STANDARD_PARAMS)
    true_full_params.update(true_opt_params)

    result = simulate(
        sim_cfg=sim_cfg,
        model_cfg=true_full_params,
        noise_cfg={'seed': seed},
    )

    fs = 1000.0 / sim_cfg['dt_ms']
    lfp = np.asarray(result.lfp)
    psd_result = compute_psd(
        lfp[:, 0] if lfp.ndim > 1 else lfp,
        fs=fs,
        nperseg=4096,
        freq_range=(1, 50),
    )

    return psd_result.freqs, psd_result.psd, true_opt_params, true_full_params, search_space, result


def run_random_optimization_test(
    budget: int = 40,
    duration_ms: float = 5000.0,
) -> tuple:
    """Run optimization test fitting to a random PSD (no stimulus)."""
    print(f"\n{'='*60}")
    print("Random PSD Optimization Test (no external stimulus)")
    print(f"{'='*60}")

    results_dir = setup_results_dir()
    sim_cfg = {'dt_ms': 0.1, 'duration_ms': duration_ms, 'jit': True}
    fs = 1000.0 / sim_cfg['dt_ms']

    # Step 1: generate random target
    print("\n[1/4] Generating random target PSD from SearchSpace.default()...")
    target_freqs, target_psd, true_opt_params, true_full_params, search_space, target_sim = (
        generate_random_target_psd(sim_cfg, seed=42)
    )
    print("  True sampled params:")
    for k in sorted(true_opt_params.keys()):
        print(f"    {k} = {true_opt_params[k]:.3f}")

    # Save target PSD and params
    np.savez(results_dir / 'target_psd.npz', freqs=target_freqs, psd=target_psd)
    with open(results_dir / 'true_params.json', 'w') as f:
        json.dump(true_full_params, f, indent=2, default=float)

    # Step 2: run optimization
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
    print("  Best params:")
    for k in sorted(opt_result.best_params.keys()):
        print(f"    {k} = {opt_result.best_params[k]:.3f}")
    print(f"  Elapsed: {opt_result.meta.get('elapsed_s', 0):.1f}s")

    # Step 3: simulate with optimized params (same seed, no stimulus)
    print("\n[3/4] Generating PSD with optimized parameters...")
    best_params = dict(STANDARD_PARAMS)
    best_params.update(opt_result.best_params)

    opt_sim_result = simulate(
        sim_cfg=sim_cfg,
        model_cfg=best_params,
        noise_cfg={'seed': 42},
    )

    opt_lfp = np.asarray(opt_sim_result.lfp)
    opt_psd_result = compute_psd(
        opt_lfp[:, 0] if opt_lfp.ndim > 1 else opt_lfp,
        fs=fs,
        nperseg=4096,
        freq_range=(1, 50),
    )

    # Step 4: visualizations
    print("\n[4/4] Creating visualizations...")

    # Figure 1: PSD comparison + loss + parameter trajectories
    fig1, gs1 = bp.visualize.get_figure(2, 2, row_len=3.2, col_len=6)
    axes1 = [[fig1.add_subplot(gs1[i, j]) for j in range(2)] for i in range(2)]

    # PSD linear / log
    bp.visualize.line_plot(
        target_freqs,
        target_psd,
        ax=axes1[0][0],
        xlabel='Frequency (Hz)',
        ylabel='PSD',
        colors='b',
        linewidth=2.0,
        label='Target (random)',
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
    axes1[0][0].legend()
    axes1[0][0].grid(alpha=0.3)

    bp.visualize.line_plot(
        target_freqs,
        psd_to_db(target_psd),
        ax=axes1[0][1],
        xlabel='Frequency (Hz)',
        ylabel='PSD (dB)',
        colors='b',
        linewidth=2.0,
        label='Target (random)',
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
    axes1[0][1].legend()
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
        title='Optimization Progress (PSD MSE)',
        colors='b',
        linewidth=1.5,
    )
    axes1[1][0].axhline(opt_result.best_loss, color='r', linestyle='--', label=f'Best: {opt_result.best_loss:.4e}')
    axes1[1][0].legend()
    axes1[1][0].grid(alpha=0.3)

    # Parameter convergence
    param_names = sorted(opt_result.best_params.keys())
    colors = list(plt.cm.tab10(np.linspace(0, 1, len(param_names))))
    for p_name, color in zip(param_names, colors):
        values = np.asarray([h[0].get(p_name, np.nan) for h in opt_result.history])
        bp.visualize.line_plot(
            eval_ids,
            values,
            ax=axes1[1][1],
            xlabel='Evaluation',
            ylabel='Parameter value',
            title='Parameter Convergence (dashed = true sampled)',
            color=color,
            linewidth=1.0,
            label=p_name,
        )
        axes1[1][1].axhline(true_opt_params.get(p_name, np.nan), color=color, linestyle='--', alpha=0.5)
    axes1[1][1].legend(loc='upper right')
    axes1[1][1].grid(alpha=0.3)

    fig1.suptitle(f'Random PSD Optimization (budget={budget}, duration={duration_ms/1000:.1f}s)', fontsize=14)
    plt.tight_layout()
    fig1.savefig(results_dir / 'optimization_random_results.png', dpi=150, bbox_inches='tight')
    print("  Saved: optimization_random_results.png")

    # Figure 2: time series comparison (2s window)
    fig2, gs2 = bp.visualize.get_figure(2, 1, row_len=2.6, col_len=7)
    axes2 = [fig2.add_subplot(gs2[i, 0]) for i in range(2)]

    t_ms = target_sim.t_s * 1000.0
    t_mid = len(t_ms) // 2
    t_window = int(2000 / sim_cfg['dt_ms'])
    t_slice = slice(t_mid - t_window // 2, t_mid + t_window // 2)

    target_lfp = np.asarray(target_sim.lfp)
    bp.visualize.line_plot(
        t_ms[t_slice],
        target_lfp[t_slice, 0] if target_lfp.ndim > 1 else target_lfp[t_slice],
        ax=axes2[0],
        xlabel='Time (ms)',
        ylabel='LFP (mV)',
        title='Target (random sampled params) 2s window',
        colors='b',
        linewidth=0.8,
    )
    axes2[0].grid(alpha=0.3)

    opt_lfp_ts = np.asarray(opt_sim_result.lfp)
    bp.visualize.line_plot(
        t_ms[t_slice],
        opt_lfp_ts[t_slice, 0] if opt_lfp_ts.ndim > 1 else opt_lfp_ts[t_slice],
        ax=axes2[1],
        xlabel='Time (ms)',
        ylabel='LFP (mV)',
        title='Optimized 2s window',
        colors='r',
        linewidth=0.8,
    )
    axes2[1].grid(alpha=0.3)

    # Annotate parameters on the figure
    true_text = "True sampled params:\n" + ", ".join(
        f"{k}={true_opt_params[k]:.2f}" for k in sorted(true_opt_params.keys())
    )
    best_text = "Fitted params:\n" + ", ".join(
        f"{k}={opt_result.best_params.get(k, np.nan):.2f}" for k in sorted(true_opt_params.keys())
    )
    axes2[0].text(
        0.01,
        0.97,
        true_text,
        transform=axes2[0].transAxes,
        va='top',
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
    )
    axes2[1].text(
        0.01,
        0.97,
        best_text,
        transform=axes2[1].transAxes,
        va='top',
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
    )

    fig2.suptitle('Time Series Comparison (target vs fitted, no stimulus)', fontsize=14)
    plt.tight_layout()
    fig2.savefig(results_dir / 'time_series_random_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: time_series_random_comparison.png")

    # Save summary + history
    summary = {
        'mode': 'random_psd',
        'budget': budget,
        'duration_ms': duration_ms,
        'best_loss': opt_result.best_loss,
        'elapsed_s': opt_result.meta.get('elapsed_s', 0),
        'true_params': true_opt_params,
        'best_params': opt_result.best_params,
        'param_errors': {
            k: float(abs(opt_result.best_params.get(k, 0) - true_opt_params.get(k, 0)))
            for k in true_opt_params.keys()
        },
    }

    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print("  Saved: summary.json")

    history_data = [{'params': p, 'loss': l} for p, l in opt_result.history]
    with open(results_dir / 'history.json', 'w') as f:
        json.dump(history_data, f, indent=2, default=float)
    print("  Saved: history.json")

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")

    # Print brief summary
    print("\nSummary (random PSD):")
    for k in sorted(true_opt_params.keys()):
        t_val = true_opt_params[k]
        b_val = opt_result.best_params.get(k, np.nan)
        print(f"  {k}: true={t_val:.2f}, fitted={b_val:.2f}, |delta|={abs(b_val - t_val):.2f}")
    print(f"  Final loss: {opt_result.best_loss:.6e}")

    return summary, results_dir


if __name__ == '__main__':
    # Try to use GPU
    try:
        bm.set_platform('gpu')
        print("[OK] GPU enabled")
    except Exception as exc:
        print(f"[WARN] GPU not available: {exc}")
        bm.set_platform('cpu')

    summary, results_dir = run_random_optimization_test(
        budget=40,
        duration_ms=5000.0,
    )

    print("\nRandom PSD optimization test completed!")
