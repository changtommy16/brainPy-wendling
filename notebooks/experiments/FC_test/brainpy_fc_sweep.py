#!/usr/bin/env python
"""
FC correlation sweeps for BrainPy Wendling network against HCP target FC.

This script tests how normalization, noise, heterogeneity, and PSP/BOLD choices
affect FC-FC correlation. Results are saved under FC_test/results/<run_id>/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import numpy as np
import scipy.io as sio

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wendling_sim import simulate  # noqa: E402
from wendling_sim.connectivity.io import load_connectivity  # noqa: E402
from wendling_sim.model.params import STANDARD_PARAMS  # noqa: E402

EPS = 1e-12

# Paths (using CONFIG)
DEFAULT_STRUCT_PATH = PROJECT_ROOT / "data" / "demo_HCP_data" / "102816" / "structural" / "DTI_CM.mat"
DEFAULT_FMRI_PATH = PROJECT_ROOT / "data" / "demo_HCP_data" / "102816" / "functional" / "TC_rsfMRI_REST1_LR.mat"
DEFAULT_NEUROLIB_DATASET = "hcp"

# Configuration settings (moved from CLI)
# 選項 1: 快速測試所有 case
CONFIG_ALL_CASES = {
    "case": ["baseline_psp", "neurolib_like_psp", "neurolib_like_bold"],  # Sweep all cases
    "struct_path": DEFAULT_STRUCT_PATH,
    "fmri_path": DEFAULT_FMRI_PATH,
    "tr_s": 0.72,
    "burn_s": 0.0,
    "seed": 0,
    "quick": True,  # Use smaller grid for faster testing
    "jit": True,
    "duration_ms": 20000.0,
    "dt_ms": 0.5,
    "outdir": Path(__file__).resolve().parent / "results",
    "use_neurolib_dataset": False,
    "neurolib_dataset": DEFAULT_NEUROLIB_DATASET,
}

# 選項 2: 只測試 PSP (比較快)
CONFIG_PSP_ONLY = {
    "case": ["baseline_psp", "neurolib_like_psp"],
    "struct_path": DEFAULT_STRUCT_PATH,
    "fmri_path": DEFAULT_FMRI_PATH,
    "tr_s": 0.72,
    "burn_s": 0.0,
    "seed": 0,
    "quick": True,
    "jit": True,
    "duration_ms": 2000.0,
    "dt_ms": 0.5,
    "outdir": Path(__file__).resolve().parent / "results",
    "use_neurolib_dataset": False,
    "neurolib_dataset": DEFAULT_NEUROLIB_DATASET,
}

# 選項 2b: PSP + no normalization (聚焦問題)
CONFIG_PSP_NO_NORM = {
    "case": ["neurolib_like_psp_no_norm"],
    "struct_path": DEFAULT_STRUCT_PATH,
    "fmri_path": DEFAULT_FMRI_PATH,
    "tr_s": 0.72,
    "burn_s": 0.0,
    "seed": 0,
    "quick": True,
    "jit": True,
    "duration_ms": 2000.0,
    "dt_ms": 0.5,
    "outdir": Path(__file__).resolve().parent / "results",
    "use_neurolib_dataset": False,
    "neurolib_dataset": DEFAULT_NEUROLIB_DATASET,
}

# 選項 3: 完整 sweep (較慢)
CONFIG_FULL = {
    "case": ["baseline_psp", "neurolib_like_psp", "neurolib_like_bold"],
    "struct_path": DEFAULT_STRUCT_PATH,
    "fmri_path": DEFAULT_FMRI_PATH,
    "tr_s": 0.72,
    "burn_s": 1.0,  # Full burn-in
    "seed": 0,
    "quick": False,  # Full grid
    "jit": True,
    "duration_ms": 10000.0,  # Full duration
    "dt_ms": 0.5,
    "outdir": Path(__file__).resolve().parent / "results",
    "use_neurolib_dataset": False,
    "neurolib_dataset": DEFAULT_NEUROLIB_DATASET,
}

# 選項 4: 使用 Neurolib dataset (80 nodes, mean FC)
CONFIG_NEUROLIB_FULL = {
    "case": ["neurolib_like_psp", "neurolib_like_bold"],
    "struct_path": None,
    "fmri_path": None,
    "tr_s": 0.72,
    "burn_s": 2.0,
    "seed": 42,
    "quick": False,
    "jit": True,
    "duration_ms": 10000.0,
    "dt_ms": 0.1,
    "outdir": Path(__file__).resolve().parent / "results",
    "use_neurolib_dataset": True,
    "neurolib_dataset": DEFAULT_NEUROLIB_DATASET,
}

# 選擇要使用的配置
CONFIG = CONFIG_NEUROLIB_FULL  # 修改這裡來選擇配置

# Base parameter profiles for heterogeneity sampling
BASE_PROFILES = {
    "brainpy": {
        "A": float(STANDARD_PARAMS["A"]),
        "B": float(STANDARD_PARAMS["B"]),
        "G": float(STANDARD_PARAMS["G"]),
        "p_mean": float(STANDARD_PARAMS["p_mean"]),
    },
    "neurolib": {
        "A": 5.0,
        "B": 22.0,
        "G": 18.0,
        "p_mean": 90.0,
    },
}

# Default cases
CASES = {
    "baseline_psp": {
        "use_bold": False,
        "normalize": "row_sum",
        "dt_ms": 1.0,
        "duration_ms": 120000.0,
        "integrator": "euler",
        "p_sigma": 2.0,
        "base_profile": "brainpy",
        "G_net_values": [1, 2, 3 ,4 ,6],
        "het_values": [0.0],
    },
    "neurolib_like_psp": {
        "use_bold": False,
        "normalize": "max",
        "dt_ms": 0.1,
        "duration_ms": 10000.0,
        "integrator": "euler",
        "p_sigma": 30.0,
        "base_profile": "neurolib",
        "random_init": True,
        "G_net_values": [1, 2, 3 ,4 ,6],
        "het_values": [0.0, 0.1, 0.2, 0.3],
    },
    "neurolib_like_psp_no_norm": {
        "use_bold": False,
        "normalize": "none",
        "dt_ms": 0.1,
        "duration_ms": 10000.0,
        "integrator": "euler",
        "p_sigma": 30.0,
        "base_profile": "neurolib",
        "random_init": True,
        "G_net_values": [1, 2, 3 ,4 , 6],
        "het_values": [0.0, 0.1, 0.2, 0.3],
    },
    "neurolib_like_bold": {
        "use_bold": True,
        "normalize": "max",
        "dt_ms": 0.1,
        "duration_ms": 10000.0,
        "integrator": "euler",
        "p_sigma": 30.0,
        "base_profile": "neurolib",
        "random_init": True,
        "G_net_values": [20, 30, 40, 50, 60],
        "het_values": [0.0, 0.1],
    },
}

# Quick mode parameters (extended G_net range)
QUICK_G_NET = [0.1, 0.3, 1.0, 3.0]  # Extended to 3.0
QUICK_HET = [0.0, 0.2]


def fc_matrix(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 2:
        raise ValueError("time series must be 2D (T, N)")
    return np.corrcoef(ts, rowvar=False)


def upper_tri_flat(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    idx = np.triu_indices(n, k=1)
    return mat[idx]


def corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 2:
        return float("nan")
    if np.std(a) < EPS or np.std(b) < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_fmri_fc_target(path: Path, tr_s: float, burn_s: float = 10.0):
    mat = sio.loadmat(path)
    tc = mat.get("tc")
    if tc is None:
        raise ValueError(f"'tc' not found in {path}")
    tc = np.asarray(tc, dtype=float)
    if tc.ndim != 2:
        raise ValueError(f"'tc' must be 2D (n_roi, T). Got shape {tc.shape}")

    burn_samples = int(burn_s / tr_s)
    tc = tc[:, burn_samples:]

    tc = tc - tc.mean(axis=1, keepdims=True)
    std = tc.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    tc = tc / std

    fc = fc_matrix(tc.T)
    return fc


def load_neurolib_target_fc(dataset_name: str):
    neurolib_root = PROJECT_ROOT / "Neurolib_package"
    sys.path.insert(0, str(neurolib_root))
    from neurolib.utils.loadData import Dataset

    ds = Dataset(dataset_name)
    target_fc = np.mean(ds.FCs, axis=0)
    return target_fc, ds.Cmat


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
    lfp = np.asarray(lfp, dtype=float)
    if lfp.ndim == 1:
        lfp = lfp[:, None]
    dt_s = dt_ms / 1000.0
    sub = max(1, int(round(hemo_dt / dt_s)))
    hemo_dt = sub * dt_s
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
        u_j = u[:, j]
        u_j = (u_j - u_j.mean()) / (u_j.std() + EPS)
        for t_idx in range(n_t):
            ui = u_j[t_idx]
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

    step = max(1, int(round(tr_s / hemo_dt)))
    bold = bold[::step]
    bold = bold - bold.mean(axis=0, keepdims=True)
    std = bold.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    bold = bold / std
    return bold


def build_heterogeneous_params(
    base_profile: str,
    n_nodes: int,
    heterogeneity: float,
    seed: int,
    clip: dict | None = None,
) -> dict:
    base = dict(BASE_PROFILES[base_profile])
    if heterogeneity <= 0:
        return base
    rng = np.random.default_rng(seed)
    out = {}
    for key in ("A", "B", "G", "p_mean"):
        base_val = float(base[key])
        jitter = rng.uniform(-heterogeneity, heterogeneity, size=n_nodes)
        vals = base_val * (1.0 + jitter)
        if clip and key in clip:
            lo, hi = clip[key]
            vals = np.clip(vals, lo, hi)
        out[key] = vals
    return out


def run_case(
    case_name: str,
    case_cfg: dict,
    struct_path: Path | None,
    struct_matrix: np.ndarray | None,
    target_fc_flat: np.ndarray,
    tr_s: float,
    burn_s: float,
    seed: int,
    quick: bool,
    outdir: Path,
    jit: bool,
    duration_ms_override: float | None,
    dt_ms_override: float | None,
) -> dict:
    results = []
    g_values = QUICK_G_NET if quick else case_cfg["G_net_values"]
    h_values = QUICK_HET if quick else case_cfg["het_values"]

    for het in h_values:
        for g_net in g_values:
            net_cfg = {
                "normalize": case_cfg["normalize"],
                "G_net": float(g_net),
            }
            if struct_matrix is not None:
                net_cfg["W"] = struct_matrix
            else:
                if struct_path is None:
                    raise ValueError("struct_path is required when not using neurolib dataset.")
                net_cfg["W_path"] = str(struct_path)
            conn = load_connectivity(net_cfg)
            n_nodes = conn.n_nodes

            model_cfg = build_heterogeneous_params(
                base_profile=case_cfg["base_profile"],
                n_nodes=n_nodes,
                heterogeneity=het,
                seed=seed,
                clip={"B": (10.0, 35.0), "G": (8.0, 30.0)},
            )
            if case_cfg.get("random_init", False):
                model_cfg["random_init"] = True
                model_cfg["init_seed"] = seed
            dt_ms = float(case_cfg["dt_ms"]) if dt_ms_override is None else float(dt_ms_override)
            duration_ms = float(case_cfg["duration_ms"]) if duration_ms_override is None else float(duration_ms_override)
            sim_cfg = {
                "dt_ms": dt_ms,
                "duration_ms": duration_ms,
                "jit": bool(jit),
                "integrator": case_cfg["integrator"],
            }
            noise_cfg = {"seed": seed, "p_sigma": float(case_cfg["p_sigma"])}

            print(
                f"[{case_name}] start G_net={g_net:.3f}, het={het:.2f}, "
                f"dt_ms={sim_cfg['dt_ms']}, duration_ms={sim_cfg['duration_ms']}",
                flush=True,
            )
            t_start = time.time()
            try:
                sim_res = simulate(sim_cfg=sim_cfg, model_cfg=model_cfg, network_cfg=net_cfg, noise_cfg=noise_cfg)
                lfp = sim_res.lfp
                burn_samples = int(burn_s / (sim_cfg["dt_ms"] / 1000.0))
                if burn_samples < lfp.shape[0]:
                    lfp = lfp[burn_samples:]
                if case_cfg["use_bold"]:
                    bold = balloon_bold_from_lfp(lfp, dt_ms=sim_cfg["dt_ms"], tr_s=tr_s)
                    bold_burn = int(burn_s / tr_s)
                    if bold_burn < bold.shape[0]:
                        bold = bold[bold_burn:]
                    sim_fc = fc_matrix(bold)
                else:
                    sim_fc = fc_matrix(lfp)

                sim_fc_flat = upper_tri_flat(sim_fc)
                fc_corr = corr_1d(sim_fc_flat, target_fc_flat)
                elapsed = time.time() - t_start
                results.append(
                    {
                        "case": case_name,
                        "G_net": float(g_net),
                        "heterogeneity": float(het),
                        "fc_corr": float(fc_corr) if np.isfinite(fc_corr) else None,
                        "elapsed_s": float(elapsed),
                    }
                )
                status = "***" if np.isfinite(fc_corr) and fc_corr >= 0.4 else ""
                print(
                    f"[{case_name}] G_net={g_net:.3f}, het={het:.2f}: fc_corr={fc_corr:.4f} ({elapsed:.1f}s) {status}"
                )
            except Exception as exc:
                elapsed = time.time() - t_start
                results.append(
                    {
                        "case": case_name,
                        "G_net": float(g_net),
                        "heterogeneity": float(het),
                        "fc_corr": None,
                        "elapsed_s": float(elapsed),
                        "error": str(exc),
                    }
                )
                print(f"[{case_name}] G_net={g_net:.3f}, het={het:.2f}: ERROR ({elapsed:.1f}s) {exc}")

    out_path = outdir / f"{case_name}_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    best = None
    for row in results:
        if row.get("fc_corr") is None:
            continue
        if best is None or row["fc_corr"] > best["fc_corr"]:
            best = row
    summary = {"case": case_name, "best": best, "count": len(results)}
    return summary


def run_sweeps() -> None:
    """Run FC correlation sweeps with internal configuration."""
    # Validate paths (unless using neurolib dataset)
    if not CONFIG.get("use_neurolib_dataset", False):
        if not CONFIG["struct_path"].exists():
            raise FileNotFoundError(f"Structural file not found: {CONFIG['struct_path']}")
        if not CONFIG["fmri_path"].exists():
            raise FileNotFoundError(f"fMRI timecourse file not found: {CONFIG['fmri_path']}")

    selected_cases = CONFIG["case"] if CONFIG["case"] else list(CASES.keys())
    for case in selected_cases:
        if case not in CASES:
            raise KeyError(f"Unknown case: {case}. Available: {list(CASES.keys())}")

    # 計算總組合數
    total_combinations = 0
    for case in selected_cases:
        case_cfg = CASES[case]
        g_net_values = QUICK_G_NET if CONFIG["quick"] else case_cfg["G_net_values"]
        het_values = QUICK_HET if CONFIG["quick"] else case_cfg["het_values"]
        combinations = len(g_net_values) * len(het_values)
        total_combinations += combinations
        print(f"[info] {case}: {len(g_net_values)} G_net × {len(het_values)} het = {combinations} combinations")
    
    print(f"[info] Total: {total_combinations} simulations across {len(selected_cases)} cases")
    print(f"[info] Estimated time: {total_combinations * 0.6:.1f}s (≈{total_combinations * 0.6 / 60:.1f} minutes)")

    if CONFIG.get("use_neurolib_dataset", False):
        target_fc, struct_matrix = load_neurolib_target_fc(CONFIG.get("neurolib_dataset", DEFAULT_NEUROLIB_DATASET))
    else:
        target_fc = load_fmri_fc_target(CONFIG["fmri_path"], CONFIG["tr_s"], burn_s=CONFIG["burn_s"])
        struct_matrix = None
    target_fc_flat = upper_tri_flat(target_fc)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    outdir = CONFIG["outdir"] / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    print(f"\n[info] running {len(selected_cases)} cases, output to {outdir}")

    for case in selected_cases:
        print(f"\n[info] running case: {case}")
        summary = run_case(
            case_name=case,
            case_cfg=CASES[case],
            struct_path=CONFIG["struct_path"],
            struct_matrix=struct_matrix,
            target_fc_flat=target_fc_flat,
            tr_s=CONFIG["tr_s"],
            burn_s=CONFIG["burn_s"],
            seed=CONFIG["seed"],
            quick=CONFIG["quick"],
            outdir=outdir,
            jit=CONFIG["jit"],
            duration_ms_override=CONFIG["duration_ms"],
            dt_ms_override=CONFIG["dt_ms"],
        )
        summaries.append(summary)

    # Save summary
    summary_path = outdir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n[done] results saved to {outdir}")
    print(f"[done] summary saved to {summary_path}")
    
    # 顯示最佳結果
    print(f"\n[summary] Best results:")
    for summary in summaries:
        best = summary.get("best")
        if best and best.get("fc_corr") is not None:
            print(f"  {summary['case']}: fc_corr={best['fc_corr']:.4f} (G_net={best['G_net']:.3f}, het={best['heterogeneity']:.2f})")


if __name__ == "__main__":
    run_sweeps()
