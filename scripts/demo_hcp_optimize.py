#!/usr/bin/env python
"""
Demo: Fit Wendling network to functional connectivity (FC) from HCP subject 102816.

Run:
    conda run -n brainpy_model python scripts/demo_hcp_optimize.py

Workflow (no CLI needed):
1) Load structural connectivity (SC) from HCP DTI_CM.mat (102816).
2) Load resting-state fMRI timecourses (TC_rsfMRI_REST1_LR.mat) as target FC.
3) Compute target FC matrix from fMRI timecourses.
4) Simulate Wendling network using direct PSP signals (no BOLD conversion).
5) Optimize parameters to maximize FC-FC correlation (target vs simulated).
6) Re-simulate best params and save FC comparison plots.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# Prefer GPU if available, otherwise fall back to CPU gracefully.
def _set_backend():
    try:
        import jax
        from jax import config

        gpus = []
        try:
            gpus = jax.devices("gpu")
        except Exception:
            gpus = []
        if gpus:
            config.update("jax_platform_name", "gpu")
            os.environ["JAX_PLATFORM_NAME"] = "gpu"
            print("[info] using GPU backend for JAX/BrainPy")
        else:
            os.environ.pop("JAX_PLATFORM_NAME", None)
            print("[info] GPU not found; using CPU backend for JAX/BrainPy")
    except Exception as e:
        os.environ.pop("JAX_PLATFORM_NAME", None)
        print(f"[info] GPU check failed ({e}); using CPU backend")

_set_backend()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wendling_sim import simulate  # noqa: E402
from wendling_sim.connectivity.io import load_connectivity  # noqa: E402
from wendling_sim.features.psd import compute_psd  # noqa: E402
from wendling_sim.loss.registry import register_loss  # noqa: E402
from wendling_sim.optimize.search_space import SearchSpace  # noqa: E402
from wendling_sim.optimize.nevergrad_engine import NevergradOptimizer  # noqa: E402

EPS = 1e-12

# --- Fixed demo settings (edit here if needed) ------------------------------
STRUCT_PATH = PROJECT_ROOT / "data" / "demo_HCP_data" / "102816" / "structural" / "DTI_CM.mat"
FMRI_PATH = PROJECT_ROOT / "data" / "demo_HCP_data" / "102816" / "functional" / "TC_rsfMRI_REST1_LR.mat"
TR_S = 0.72  # assumed TR for HCP rs-fMRI

OUTDIR = PROJECT_ROOT / "results" / "demo_hcp_fc_psp"

SIM_CFG = {"dt_ms": 1.0, "duration_ms": 120000.0, "jit": True, "integrator": "euler"}  # 300s simulation
BURN_S = 10.0  # burn-in time in seconds
TARGET_FC_CORR = 0.4  # target FC-FC correlation based on literature
PLOT = True

# --- Loss helpers -----------------------------------------------------------
def _trapz_safe(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    dx = np.diff(x)
    y_mid = (y[:-1] + y[1:]) * 0.5
    return float(np.sum(y_mid * dx))


def _to_1d(psd: np.ndarray) -> np.ndarray:
    arr = np.asarray(psd)
    if arr.ndim == 2:
        return arr.mean(axis=1)
    return arr


def double_gamma_hrf(t):
    t = np.asarray(t, dtype=float)
    peak1, under, p_u_ratio = 6, 16, 0.35
    import math
    h1 = (t ** peak1) * np.exp(-t) / math.factorial(peak1)
    h2 = (t ** under) * np.exp(-t) / math.factorial(under)
    h = h1 - p_u_ratio * h2
    return h / (np.max(np.abs(h)) + EPS)


def lfp_to_bold_like(
    lfp: np.ndarray,
    dt_ms: float,
    tr_s: float,
    roi_idx: int = None,
    hemo_dt: float = 0.01,
    kappa: float = 0.65,
    gamma: float = 0.41,
    tau_0: float = 0.98,
    alpha: float = 0.32,
    E0: float = 0.34,
    V0: float = 0.02,
    TE: float = 0.04,
):
    """
    Pseudo-BOLD via Balloon–Windkessel model (Friston 2000):
        s' = u - kappa*s - gamma*(f-1)
        f' = s
        v' = (f - v^(1/alpha)) / tau_0
        q' = (f * (1 - (1-E0)^(1/f)) / E0 - v^(1/alpha) * q / v) / tau_0
        BOLD = V0 * (k1*(1-q) + k2*(1-q/v) + k3*(1-v))
    """
    lfp = np.asarray(lfp, dtype=float)
    if lfp.ndim == 2:
        if roi_idx is not None and 0 <= roi_idx < lfp.shape[1]:
            lfp = lfp[:, roi_idx]
        else:
            lfp = lfp.mean(axis=1)
    # neural drive (z-scored)
    u = _zscore(lfp)
    dt_s = dt_ms / 1000.0
    sub = max(1, int(round(hemo_dt / dt_s)))
    hemo_dt = sub * dt_s
    u = u[::sub]

    # hemodynamic states
    s = 0.0
    f = 1.0
    v = 1.0
    q = 1.0

    # constants for BOLD signal
    k1 = 7 * E0 * TE
    k2 = 2
    k3 = 2 * E0 - 0.2

    bold_trace = []
    for ui in u:
        ds = ui - kappa * s - gamma * (f - 1.0)
        df = s
        dv = (f - v ** (1.0 / alpha)) / tau_0
        E = 1.0 - (1.0 - E0) ** (1.0 / max(f, EPS))
        dq = (f * E / E0 - v ** (1.0 / alpha) * q / max(v, EPS)) / tau_0

        s += hemo_dt * ds
        f = max(1e-6, f + hemo_dt * df)
        v = max(1e-6, v + hemo_dt * dv)
        q = max(1e-6, q + hemo_dt * dq)

        bold = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
        bold_trace.append(bold)

    bold_trace = np.asarray(bold_trace, dtype=float)
    # downsample to TR
    step = max(1, int(round(tr_s / hemo_dt)))
    bold_ds = bold_trace[::step]
    # detrend & zscore
    bold_ds = bold_ds - np.mean(bold_ds)
    std = np.std(bold_ds)
    if std > 0:
        bold_ds = bold_ds / std
    return bold_ds, 1.0 / tr_s


def band_peak(psd: np.ndarray, freqs: np.ndarray, band: Tuple[float, float]) -> float:
    psd1d = _to_1d(psd)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    idx = np.argmax(psd1d[mask])
    return float(freqs[mask][idx])


def band_peak_loss(psd: np.ndarray, target_psd: np.ndarray, freqs: np.ndarray, band=(0.04, 0.1)) -> float:
    return (band_peak(psd, freqs, band) - band_peak(target_psd, freqs, band)) ** 2


def psd_slope(psd: np.ndarray, freqs: np.ndarray, band=(0.01, 0.3)) -> float:
    psd1d = _to_1d(psd)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if np.sum(mask) < 2:
        return 0.0
    x = np.log10(freqs[mask] + EPS)
    y = np.log10(psd1d[mask] + EPS)
    coef = np.polyfit(x, y, 1)
    return float(coef[0])


def slope_loss(psd: np.ndarray, target_psd: np.ndarray, freqs: np.ndarray, band=(0.01, 0.3)) -> float:
    return (psd_slope(psd, freqs, band) - psd_slope(target_psd, freqs, band)) ** 2


def psd_db_mse(psd: np.ndarray, target_psd: np.ndarray, freqs=None) -> float:
    psd1d = _to_1d(psd)
    tgt1d = _to_1d(target_psd)
    db = 10.0 * np.log10(psd1d + EPS)
    db_t = 10.0 * np.log10(tgt1d + EPS)
    return float(np.mean((db - db_t) ** 2))


def band_ratio(psd: np.ndarray, freqs: np.ndarray, num_band: Tuple[float, float], den_band: Tuple[float, float]) -> float:
    psd1d = _to_1d(psd)

    def band_power(band):
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(mask):
            return 0.0
        return _trapz_safe(psd1d[mask], freqs[mask])

    num = band_power(num_band)
    den = band_power(den_band)
    return num / (den + EPS)


def ratio_loss(psd: np.ndarray, target_psd: np.ndarray, freqs: np.ndarray, num_band=(0.01, 0.1), den_band=(0.1, 0.25)) -> float:
    return (band_ratio(psd, freqs, num_band, den_band) - band_ratio(target_psd, freqs, num_band, den_band)) ** 2


def register_custom_losses():
    register_loss("low_peak_loss", band_peak_loss)
    register_loss("slope_loss", slope_loss)
    register_loss("psd_db_mse", psd_db_mse)
    register_loss("ratio_loss", ratio_loss)


# --- Utility helpers -------------------------------------------------------
def _limit_psd_params(sig_len: int, psd_cfg: dict):
    cfg_nperseg = psd_cfg.get("nperseg")
    nperseg = sig_len if cfg_nperseg is None else min(cfg_nperseg, sig_len)
    cfg_noverlap = psd_cfg.get("noverlap")
    if cfg_noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = min(cfg_noverlap, max(0, nperseg - 1))
    return nperseg, noverlap


# --- Target building --------------------------------------------------------
def load_fmri_fc_target(path: Path, tr_s: float, burn_s: float = 10.0):
    """Load fMRI timecourses and compute target FC matrix."""
    mat = sio.loadmat(path)
    tc = mat.get("tc")
    if tc is None:
        raise ValueError(f"'tc' not found in {path}")
    # tc shape: (n_roi, T)
    tc = np.asarray(tc, dtype=float)
    if tc.ndim != 2:
        raise ValueError(f"'tc' must be 2D (n_roi, T). Got shape {tc.shape}")
    
    # Remove burn-in period
    burn_samples = int(burn_s / tr_s)
    tc = tc[:, burn_samples:]
    
    # Z-score per ROI
    tc = tc - tc.mean(axis=1, keepdims=True)
    std = tc.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    tc = tc / std
    
    # Compute FC matrix
    fc = fc_matrix(tc.T)  # transpose to (T, N) for fc_matrix
    return fc, tc.shape[0]




def _zscore(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=float)
    sig = sig - sig.mean()
    std = sig.std()
    if std > 0:
        sig = sig / std
    return sig


def fc_matrix(ts: np.ndarray) -> np.ndarray:
    """Compute FC matrix from time series (T, N)."""
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 2:
        raise ValueError("time series must be 2D (T, N)")
    return np.corrcoef(ts, rowvar=False)


def upper_tri_flat(mat: np.ndarray) -> np.ndarray:
    """Extract upper triangular elements (excluding diagonal)."""
    n = mat.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    return mat[triu_idx]


# --- Main workflow ----------------------------------------------------------
def main():
    if not STRUCT_PATH.exists():
        raise FileNotFoundError(f"Structural file not found: {STRUCT_PATH}")
    if not FMRI_PATH.exists():
        raise FileNotFoundError(f"fMRI timecourse file not found: {FMRI_PATH}")

    register_custom_losses()

    # Connectivity from structural
    network_cfg = {"W_path": str(STRUCT_PATH), "normalize": "row_sum", "G_net": 1.5}
    conn = load_connectivity(network_cfg)
    network_cfg["n_nodes"] = conn.n_nodes
    print(f"[info] loaded structural connectivity {STRUCT_PATH} (n_nodes={conn.n_nodes})")

    # Target fMRI: compute FC matrix
    print(f"[info] loading fMRI and computing target FC...")
    target_fc, n_nodes = load_fmri_fc_target(FMRI_PATH, TR_S, BURN_S)
    target_fc_flat = upper_tri_flat(target_fc)
    print(f"[info] target FC shape: {target_fc.shape}, n_nodes: {n_nodes}")
    
    # Update network config with actual node count
    network_cfg["n_nodes"] = n_nodes

    # Simulation configs
    sim_cfg = dict(SIM_CFG)
    
    # Search space and optimizer
    search_space = SearchSpace(
        bounds={
            'A': (2.0, 8.0),        # 縮小範圍，避免過度興奮
            'B': (15.0, 30.0),      # 避免癲癇範圍 (<30)
            'G': (12.0, 25.0),      # 維持正常範圍 (>12)
            'a': (80.0, 150.0),     # 興奮性時間常數
            'b': (40.0, 80.0),      # 慢抑制時間常數  
            'g': (400.0, 800.0),    # 快抑制時間常數
            'G_net': (0.1, 3.0),    # 網絡耦合強度
        },
        log_scale={'a', 'b', 'g'},
        fixed={}
    )
    opt_cfg = {"budget": 200, "optimizer": "NGOpt", "num_workers": 1, "seed": 0}

    noise_cfg = {"seed": opt_cfg.get("seed", 0)}

    def objective_fn(params: dict):
        """Objective: maximize FC-FC correlation (minimize negative correlation)."""
        params_local = dict(params)
        candidate_G_net = params_local.pop("G_net", network_cfg.get("G_net", 0.0))
        net_cfg = dict(network_cfg)
        net_cfg["G_net"] = candidate_G_net

        sim_res = simulate(sim_cfg=sim_cfg, model_cfg=params_local, network_cfg=net_cfg, noise_cfg=noise_cfg)
        
        # Extract PSP signals directly (no BOLD conversion)
        lfp = sim_res.lfp  # shape: (T, N)
        
        # Remove burn-in
        burn_samples = int(BURN_S / (sim_cfg["dt_ms"] / 1000.0))
        if burn_samples < lfp.shape[0]:
            lfp_clean = lfp[burn_samples:]
        else:
            lfp_clean = lfp
        
        # Compute simulated FC
        sim_fc = fc_matrix(lfp_clean)
        sim_fc_flat = upper_tri_flat(sim_fc)
        
        # Compute FC-FC correlation
        if len(sim_fc_flat) != len(target_fc_flat):
            return float("inf")
        
        fc_corr = np.corrcoef(sim_fc_flat, target_fc_flat)[0, 1]
        
        # Return negative correlation for minimization
        return -fc_corr

    optimizer = NevergradOptimizer(
        search_space=search_space,
        objective_fn=objective_fn,
        budget=opt_cfg.get("budget", 15),
        optimizer_name=opt_cfg.get("optimizer", "NGOpt"),
        num_workers=opt_cfg.get("num_workers", 1),
        seed=opt_cfg.get("seed", None),
    )

    print("[info] starting FC optimization (maximizing FC-FC correlation)...")
    result = optimizer.run(verbose=True)
    print("\n[done] best_loss (negative corr):", result.best_loss)
    print("[done] best_params:", result.best_params)
    best_fc_corr = -result.best_loss
    print(f"[done] best FC-FC correlation: {best_fc_corr:.4f}")
    
    if best_fc_corr >= TARGET_FC_CORR:
        print(f"✅ TARGET ACHIEVED: FC correlation >= {TARGET_FC_CORR}")
    else:
        print(f"⚠️ Target not achieved. Best: {best_fc_corr:.4f} < {TARGET_FC_CORR}")

    # Re-evaluate best params for visualization
    print("[info] evaluating best params for visualization...")
    best_net_cfg = dict(network_cfg)
    best_net_cfg["G_net"] = result.best_params.get("G_net", network_cfg.get("G_net", 1.5))
    best_local_params = dict(result.best_params)
    best_local_params.pop("G_net", None)
    
    res_best = simulate(sim_cfg=sim_cfg, model_cfg=best_local_params, network_cfg=best_net_cfg, noise_cfg={"seed": 999})
    
    # Extract PSP signals
    lfp_best = res_best.lfp
    burn_samples = int(BURN_S / (sim_cfg["dt_ms"] / 1000.0))
    if burn_samples < lfp_best.shape[0]:
        lfp_clean = lfp_best[burn_samples:]
    else:
        lfp_clean = lfp_best
    
    # Compute final FC matrix
    sim_fc_best = fc_matrix(lfp_clean)
    sim_fc_flat_best = upper_tri_flat(sim_fc_best)
    
    # Final metrics
    final_fc_corr = np.corrcoef(sim_fc_flat_best, target_fc_flat)[0, 1]
    print(f"[info] final FC-FC correlation: {final_fc_corr:.4f}")

    if PLOT:
        OUTDIR.mkdir(parents=True, exist_ok=True)

        # 1) FC matrix comparison
        fig_fc, axes_fc = plt.subplots(1, 3, figsize=(15, 4))
        
        # Target FC
        im0 = axes_fc[0].imshow(target_fc, cmap="RdBu_r", vmin=-1, vmax=1)
        axes_fc[0].set_title("Target FC (fMRI)")
        axes_fc[0].set_xlabel("ROI")
        axes_fc[0].set_ylabel("ROI")
        plt.colorbar(im0, ax=axes_fc[0], fraction=0.046, pad=0.04)
        
        # Simulated FC
        im1 = axes_fc[1].imshow(sim_fc_best, cmap="RdBu_r", vmin=-1, vmax=1)
        axes_fc[1].set_title("Simulated FC (PSP)")
        axes_fc[1].set_xlabel("ROI")
        axes_fc[1].set_ylabel("ROI")
        plt.colorbar(im1, ax=axes_fc[1], fraction=0.046, pad=0.04)
        
        # FC-FC scatter plot
        sample_idx = np.random.choice(len(target_fc_flat), min(2000, len(target_fc_flat)), replace=False)
        axes_fc[2].scatter(target_fc_flat[sample_idx], sim_fc_flat_best[sample_idx], s=5, alpha=0.3)
        axes_fc[2].plot([-1, 1], [-1, 1], 'r--', lw=2)
        axes_fc[2].set_xlabel("Target FC")
        axes_fc[2].set_ylabel("Simulated FC")
        axes_fc[2].set_title(f"FC-FC Correlation: r = {final_fc_corr:.3f}")
        axes_fc[2].set_xlim(-1, 1)
        axes_fc[2].set_ylim(-1, 1)
        axes_fc[2].grid(alpha=0.3)
        
        fig_fc.tight_layout()
        fig_fc.savefig(OUTDIR / "fc_comparison.png", dpi=150)

        # 2) Structural connectivity matrix
        fig_conn, ax_conn = plt.subplots(1, 1, figsize=(6, 5))
        im = ax_conn.imshow(conn.W, cmap="viridis", origin="lower")
        ax_conn.set_title("Structural Connectivity (W)")
        ax_conn.set_xlabel("Source node")
        ax_conn.set_ylabel("Target node")
        fig_conn.colorbar(im, ax=ax_conn, fraction=0.046, pad=0.04, label="weight (normalized)")
        fig_conn.tight_layout()
        fig_conn.savefig(OUTDIR / "connectivity.png", dpi=150)

        print(f"[info] saved plots -> {OUTDIR}")


if __name__ == "__main__":
    main()
