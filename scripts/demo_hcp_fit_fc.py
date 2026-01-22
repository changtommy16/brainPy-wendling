#!/usr/bin/env python
"""
Demo: Fit Wendling network to fMRI functional connectivity (FC) from HCP subject 102816.

Run (no CLI args needed):
    conda run -n brainpy_model python scripts/demo_hcp_fit_fc.py

Flow:
1) Load structural connectivity (DTI_CM.mat) and resting-state fMRI timecourses (TC_rsfMRI_REST1_LR.mat).
2) Build target FC (Pearson corr across ROIs) after z-scoring and dropping initial burn-in.
3) Simulate Wendling network, convert each node's LFP -> pseudo-BOLD via Balloon–Windkessel, compute FC.
4) Optimize model params to minimize FC MSE (upper triangle); optional PSD regularizer for stability.
5) Save plots for FC target vs sim and PSD comparison (mean across ROIs).
"""

from pathlib import Path
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# Prefer GPU if available, otherwise fall back to CPU gracefully.
import brainpy as bp
import brainpy.math as bm
print(f"BrainPy 平台: {bp.math.get_platform()}")
bm.set_platform('gpu')
print(f"BrainPy 平台: {bp.math.get_platform()}")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wendling_sim import simulate  # noqa: E402
from wendling_sim.connectivity.io import load_connectivity  # noqa: E402
from wendling_sim.features.psd import compute_psd  # noqa: E402
from wendling_sim.optimize.search_space import SearchSpace  # noqa: E402
from wendling_sim.optimize.nevergrad_engine import NevergradOptimizer  # noqa: E402

EPS = 1e-12

# Paths
STRUCT_PATH = PROJECT_ROOT / "data" / "demo_HCP_data" / "102816" / "structural" / "DTI_CM.mat"
FMRI_PATH = PROJECT_ROOT / "data" / "demo_HCP_data" / "102816" / "functional" / "TC_rsfMRI_REST1_LR.mat"
OUTDIR = PROJECT_ROOT / "results" / "demo_hcp_fc"

# Sim / opt settings
TR_S = 0.72
SIM_CFG = {"dt_ms": 1.0, "duration_ms": 120000.0, "jit": True, "integrator": "euler"}  # 120s
BURN_S = 10.0
PSD_CFG = {"nperseg": 128, "noverlap": 64, "freq_range": (0.01, 0.5), "roi": "mean"}
OPT_CFG = {"budget": 20, "optimizer": "NGOpt", "num_workers": 1, "seed": 1}

# ---------------------------------------------------------------------------#
# Helpers
# ---------------------------------------------------------------------------#
def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    std = x.std()
    if std > 0:
        x = x / std
    return x


def load_fmri_tc(path: Path) -> np.ndarray:
    mat = sio.loadmat(path)
    tc = mat.get("tc")
    if tc is None:
        raise ValueError(f"'tc' not found in {path}")
    tc = np.asarray(tc, dtype=float)
    if tc.ndim != 2:
        raise ValueError(f"tc must be 2D (n_roi, T). Got {tc.shape}")
    # z-score per ROI
    tc = tc - tc.mean(axis=1, keepdims=True)
    std = tc.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return tc / std


def balloon_bold_from_lfp(
    lfp: np.ndarray,
    dt_ms: float,
    tr_s: float,
    hemo_dt: float = 0.01,
    kappa: float = 0.65,
    gamma: float = 0.41,
    tau_0: float = 0.98,
    alpha: float = 0.32,
    E0: float = 0.34,
    V0: float = 0.02,
    TE: float = 0.04,
) -> np.ndarray:
    """
    Balloon–Windkessel per node. Returns BOLD array shape (T_bold, N).
    """
    lfp = np.asarray(lfp, dtype=float)
    if lfp.ndim == 1:
        lfp = lfp[:, None]
    dt_s = dt_ms / 1000.0
    sub = max(1, int(round(hemo_dt / dt_s)))
    hemo_dt = sub * dt_s
    # downsample neural drive for hemo integration
    u = lfp[::sub]
    n_t, n_nodes = u.shape
    bold = np.zeros((n_t, n_nodes), dtype=float)

    k1 = 7 * E0 * TE
    k2 = 2
    k3 = 2 * E0 - 0.2

    for j in range(n_nodes):
        s = 0.0
        f = 1.0
        v = 1.0
        q = 1.0
        for t_idx in range(n_t):
            ui = u[t_idx, j]
            ui = (ui - u[:, j].mean()) / (u[:, j].std() + EPS)  # z-score per node drive
            ds = ui - kappa * s - gamma * (f - 1.0)
            df = s
            dv = (f - v ** (1.0 / alpha)) / tau_0
            E = 1.0 - (1.0 - E0) ** (1.0 / max(f, EPS))
            dq = (f * E / E0 - v ** (1.0 / alpha) * q / max(v, EPS)) / tau_0

            s += hemo_dt * ds
            f = max(1e-6, f + hemo_dt * df)
            v = max(1e-6, v + hemo_dt * dv)
            q = max(1e-6, q + hemo_dt * dq)

            bold[t_idx, j] = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))

    # downsample to TR
    step = max(1, int(round(tr_s / hemo_dt)))
    bold = bold[::step]
    # detrend/z-score per ROI
    bold = bold - bold.mean(axis=0, keepdims=True)
    std = bold.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    bold = bold / std
    return bold


def fc_matrix(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 2:
        raise ValueError("time series must be 2D (T, N)")
    return np.corrcoef(ts, rowvar=False)


def upper_tri_flat(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    return mat[triu_idx]


def limit_psd_params(sig_len: int, psd_cfg: dict):
    nperseg = min(psd_cfg.get("nperseg", sig_len), sig_len)
    noverlap = psd_cfg.get("noverlap", nperseg // 2)
    noverlap = min(noverlap, max(0, nperseg - 1))
    return nperseg, noverlap


# ---------------------------------------------------------------------------#
# Main
# ---------------------------------------------------------------------------#
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not STRUCT_PATH.exists() or not FMRI_PATH.exists():
        raise FileNotFoundError("HCP demo data not found. Check data/demo_HCP_data/102816/")

    conn = load_connectivity({"W_path": str(STRUCT_PATH), "normalize": "row_sum", "G_net": 1.5})
    print(f"[info] loaded SC (n_nodes={conn.n_nodes})")

    fmri_tc = load_fmri_tc(FMRI_PATH)
    burn = int(BURN_S / TR_S)
    fmri_tc = fmri_tc[:, burn:]
    target_fc = fc_matrix(fmri_tc.T)
    target_fc_flat = upper_tri_flat(target_fc)
    print(f"[info] target FC shape {target_fc.shape}, burn-in {BURN_S}s")

    sim_cfg = dict(SIM_CFG)
    search_space = SearchSpace.default()
    opt_cfg = dict(OPT_CFG)

    def objective_fn(params: dict):
        params_local = dict(params)
        candidate_G_net = params_local.pop("G_net", conn.meta.get("G_net", 0.0))
        net_cfg = {"W_path": str(STRUCT_PATH), "normalize": "row_sum", "G_net": candidate_G_net}

        sim_res = simulate(sim_cfg=sim_cfg, model_cfg=params_local, network_cfg=net_cfg, noise_cfg={"seed": opt_cfg.get("seed", 1)})
        bold = balloon_bold_from_lfp(sim_res.lfp, dt_ms=sim_cfg["dt_ms"], tr_s=TR_S)
        bold = bold[burn:]

        # trim to match fMRI length
        T = min(bold.shape[0], fmri_tc.shape[1])
        if T < 5:
            return float("inf")
        bold = bold[:T]
        tgt = fmri_tc[:, :T].T  # (T, N)

        sim_fc = fc_matrix(bold)
        sim_fc_flat = upper_tri_flat(sim_fc)
        fc_mse = float(np.mean((sim_fc_flat - target_fc_flat) ** 2))

        # PSD regularizer (mean across ROIs)
        nperseg, noverlap = limit_psd_params(T, PSD_CFG)
        sim_psd = compute_psd(bold, fs=1.0 / TR_S, nperseg=nperseg, noverlap=noverlap, roi="mean")
        tgt_psd = compute_psd(tgt, fs=1.0 / TR_S, nperseg=nperseg, noverlap=noverlap, roi="mean")
        psd_loss = float(np.mean((10 * np.log10(sim_psd.psd + EPS) - 10 * np.log10(tgt_psd.psd + EPS)) ** 2))

        return fc_mse + 0.1 * psd_loss

    optimizer = NevergradOptimizer(
        search_space=search_space,
        objective_fn=objective_fn,
        budget=opt_cfg["budget"],
        optimizer_name=opt_cfg["optimizer"],
        num_workers=opt_cfg["num_workers"],
        seed=opt_cfg.get("seed", None),
    )

    print("[info] starting FC optimization...")
    result = optimizer.run(verbose=True)
    best_params = result.best_params
    print(f"[done] best_loss: {result.best_loss:.4f}")
    print(f"[done] best_params: {best_params}")

    # Evaluate best
    best_net = {"W_path": str(STRUCT_PATH), "normalize": "row_sum", "G_net": best_params.get("G_net", 1.5)}
    best_local = dict(best_params)
    best_local.pop("G_net", None)
    sim_best = simulate(sim_cfg=sim_cfg, model_cfg=best_local, network_cfg=best_net, noise_cfg={"seed": 999})
    bold_best = balloon_bold_from_lfp(sim_best.lfp, dt_ms=sim_cfg["dt_ms"], tr_s=TR_S)
    bold_best = bold_best[burn:]
    T_eval = min(bold_best.shape[0], fmri_tc.shape[1])
    bold_best = bold_best[:T_eval]
    tgt_eval = fmri_tc[:, :T_eval].T

    sim_fc_best = fc_matrix(bold_best)
    sim_fc_flat = upper_tri_flat(sim_fc_best)
    ts_corr = np.corrcoef(bold_best.mean(axis=1), tgt_eval.mean(axis=1))[0, 1]
    fc_mse_best = float(np.mean((sim_fc_flat - target_fc_flat) ** 2))
    print(f"[info] eval metrics -> FC MSE: {fc_mse_best:.4f}, mean-signal corr: {ts_corr:.4f}")

    # Plots
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(target_fc, vmin=-1, vmax=1, cmap="coolwarm")
    axes[0].set_title("Target FC (fMRI)")
    im1 = axes[1].imshow(sim_fc_best, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_title("Simulated FC")
    for ax in axes:
        ax.set_xlabel("ROI")
        ax.set_ylabel("ROI")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fc_compare.png", dpi=150)

    nperseg, noverlap = limit_psd_params(T_eval, PSD_CFG)
    sim_psd = compute_psd(bold_best, fs=1.0 / TR_S, nperseg=nperseg, noverlap=noverlap, roi="mean")
    tgt_psd = compute_psd(tgt_eval, fs=1.0 / TR_S, nperseg=nperseg, noverlap=noverlap, roi="mean")
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    ax2.plot(sim_psd.freqs, sim_psd.psd, label="sim mean PSD")
    ax2.plot(tgt_psd.freqs, tgt_psd.psd, label="target mean PSD", color="black")
    ax2.set_xlim(0, 0.5)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUTDIR / "psd_compare.png", dpi=150)

    print(f"[info] saved outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
