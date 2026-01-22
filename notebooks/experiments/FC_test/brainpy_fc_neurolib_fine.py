#!/usr/bin/env python
"""
Fine sweep around G_net to try to reach FC corr >= 0.6 using Neurolib HCP target.
"""

from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wendling_sim import simulate  # noqa: E402
from wendling_sim.connectivity.io import load_connectivity  # noqa: E402

# Reuse helpers from the sweep script
from brainpy_fc_sweep import (
    balloon_bold_from_lfp,
    corr_1d,
    fc_matrix,
    load_neurolib_target_fc,
    upper_tri_flat,
)


def run_fine():
    target_fc, cmat = load_neurolib_target_fc("hcp")
    target_fc_flat = upper_tri_flat(target_fc)

    g_values = [0.06, 0.08, 0.10, 0.12, 0.15]
    seeds = [0, 1, 2, 3, 4]

    sim_cfg = {
        "dt_ms": 0.1,
        "duration_ms": 10000.0,
        "jit": True,
        "integrator": "euler",
    }
    tr_s = 0.72
    burn_s = 2.0

    best = {"psp": (-1.0, None), "bold": (-1.0, None)}

    for g_net in g_values:
        for seed in seeds:
            net_cfg = {"W": cmat, "normalize": "max", "G_net": float(g_net)}
            conn = load_connectivity(net_cfg)
            n_nodes = conn.n_nodes

            model_cfg = {
                "A": 5.0,
                "B": 22.0,
                "G": 18.0,
                "p_mean": 90.0,
                "random_init": True,
                "init_seed": seed,
            }
            noise_cfg = {"seed": seed, "p_sigma": 30.0}

            start = time.time()
            sim_res = simulate(sim_cfg=sim_cfg, model_cfg=model_cfg, network_cfg=net_cfg, noise_cfg=noise_cfg)
            lfp = sim_res.lfp
            burn_samples = int(burn_s / (sim_cfg["dt_ms"] / 1000.0))
            if burn_samples < lfp.shape[0]:
                lfp = lfp[burn_samples:]

            sim_fc = fc_matrix(lfp)
            fc_corr = corr_1d(upper_tri_flat(sim_fc), target_fc_flat)
            elapsed = time.time() - start
            if fc_corr > best["psp"][0]:
                best["psp"] = (fc_corr, {"G_net": g_net, "seed": seed})

            bold = balloon_bold_from_lfp(lfp, dt_ms=sim_cfg["dt_ms"], tr_s=tr_s)
            bold_burn = int(burn_s / tr_s)
            if bold_burn < bold.shape[0]:
                bold = bold[bold_burn:]
            bold_fc = fc_matrix(bold)
            bold_corr = corr_1d(upper_tri_flat(bold_fc), target_fc_flat)
            if bold_corr > best["bold"][0]:
                best["bold"] = (bold_corr, {"G_net": g_net, "seed": seed})

            print(
                f"G_net={g_net:.3f}, seed={seed}: psp={fc_corr:.4f}, bold={bold_corr:.4f} ({elapsed:.1f}s)",
                flush=True,
            )

    print("\nBest PSP:", best["psp"][0], best["psp"][1])
    print("Best BOLD:", best["bold"][0], best["bold"][1])


if __name__ == "__main__":
    run_fine()
