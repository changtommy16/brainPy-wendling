#!/usr/bin/env python
"""
Estimate runtime for whole-brain Wendling simulations and a 6-parameter optimization loop.

Strategy:
- Run a short benchmark simulation (default: 1 s) with a representative network (86 nodes).
- Extrapolate linearly to a target duration (default: 10 minutes) and a target number of trials.
- Optionally estimate a naive optimization wall time: generations * population_size * sim_time_per_trial.

Usage:
    conda run -n brainpy_model python scripts/estimate_runtime.py \
        --duration-benchmark-ms 1000 \
        --duration-target-min 10 \
        --trials 1 \
        --generations 100 \
        --pop-size 1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wendling_sim import simulate  # noqa: E402


def run_benchmark(n_nodes: int, duration_ms: float, dt_ms: float, jit: bool = True, seed: int = 0) -> float:
    """Run one simulation and return wall time (seconds)."""
    network_cfg = {
        "n_nodes": n_nodes,
        "generator": {
            "name": "small_world",
            "options": {"k": 6, "beta": 0.1, "weight_dist": "lognormal", "weight_scale": 0.6, "seed": 123},
        },
        "G_net": 1.0,
        "normalize": "row_sum",
    }
    sim_cfg = {"dt_ms": dt_ms, "duration_ms": duration_ms, "jit": jit, "progress_bar": False}

    t0 = time.perf_counter()
    _ = simulate(sim_cfg=sim_cfg, model_cfg={}, network_cfg=network_cfg, noise_cfg={"seed": seed})
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="Estimate whole-brain Wendling runtime.")
    parser.add_argument("--n-nodes", type=int, default=86, help="Number of nodes for benchmark.")
    parser.add_argument("--duration-benchmark-ms", type=float, default=1000.0, help="Benchmark sim duration (ms).")
    parser.add_argument("--duration-target-min", type=float, default=10.0, help="Target sim duration (minutes).")
    parser.add_argument("--dt-ms", type=float, default=0.1, help="Timestep (ms).")
    parser.add_argument("--jit", action="store_true", help="Enable JIT during benchmark (recommended).")
    parser.add_argument("--trials", type=int, default=1, help="Number of target trials to estimate.")
    parser.add_argument("--generations", type=int, default=100, help="Generations for optimization estimate.")
    parser.add_argument("--pop-size", type=int, default=1, help="Population size per generation for optimization.")
    args = parser.parse_args()

    print(f"[benchmark] n_nodes={args.n_nodes}, duration={args.duration_benchmark_ms} ms, dt={args.dt_ms} ms")
    wall = run_benchmark(
        n_nodes=args.n_nodes,
        duration_ms=args.duration_benchmark_ms,
        dt_ms=args.dt_ms,
        jit=args.jit,
        seed=0,
    )
    print(f"[benchmark] wall time: {wall:.3f} s")

    target_ms = args.duration_target_min * 60_000.0
    scale = target_ms / args.duration_benchmark_ms
    est_single = wall * scale
    est_trials = est_single * args.trials
    est_opt = est_single * args.pop_size * args.generations

    print(f"[estimate] target duration: {args.duration_target_min} min ({target_ms:.0f} ms)")
    print(f"[estimate] single trial (scaled): {est_single/60:.2f} min ({est_single:.1f} s)")
    print(f"[estimate] {args.trials} trial(s): {est_trials/60:.2f} min ({est_trials:.1f} s)")
    print(f"[estimate] optimization (gen={args.generations}, pop={args.pop_size}): {est_opt/3600:.2f} hours")

    print("\nNotes:")
    print("- This is a linear extrapolation; real runs may differ with JIT=True or different hardware.")
    print("- Increase pop-size to reflect your optimizer; decrease dt or duration to trade accuracy vs speed.")
    print("- For more precise estimates, rerun with a longer benchmark duration (e.g., 10s).")


if __name__ == "__main__":
    main()
