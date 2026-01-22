import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wendling_sim import simulate
from wendling_sim.features.psd import compute_psd
from wendling_sim.loss.registry import build_loss

def check_optimization_result():
    # 1. Load the results
    results_dir = PROJECT_ROOT / "results" / "opt_test_20260122_180645"
    summary_path = results_dir / "summary.json"
    target_psd_path = results_dir / "target_psd.npz"
    
    if not summary_path.exists():
        print(f"Error: {summary_path} not found.")
        return

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # 2. Extract info
    best_params = summary["best_params"]
    print(f"Best Params: {best_params}")
    
    # 3. Load Target PSD
    target_data = np.load(target_psd_path)
    target_psd = target_data["psd"]
    target_freqs = target_data["freqs"]
    print(f"Target PSD shape: {target_psd.shape}, Freqs shape: {target_freqs.shape}")

    # 4. Re-run simulation with best params
    print("\nRe-running simulation...")
    # We need to construct the full param dict. The optimization likely only optimized A, B, G.
    # We need the base defaults for others (like p_mean, p_sigma).
    # Based on summary.json "true_params", it seems this was a Type4 target.
    from wendling_sim.model.params import get_type_params
    
    # Start with default Type4 but override with optimized values
    model_cfg = get_type_params("Type4") 
    model_cfg.update(best_params) # Override A, B, G

    # Simulation config must match what was likely used (from summary.json: duration_ms=5000)
    sim_cfg = {
        "dt_ms": 0.1,  # Standard default
        "duration_ms": summary.get("duration_ms", 5000.0),
        "jit": True
    }
    
    # Run
    res = simulate(
        sim_cfg=sim_cfg,
        model_cfg=model_cfg,
        noise_cfg={"seed": 0} # Assuming seed 0 for deterministic check
    )
    
    # 5. Compute PSD of the new simulation
    # We need to know the PSD config used during optimization.
    # Usually run_optimize.py uses defaults: roi="mean", freq_range=(0.5, 30) or similar.
    # Let's try to match the target_freqs.
    
    fs = 1000.0 / sim_cfg["dt_ms"]
    
    # Note: We must ensure nperseg matches to get the same frequency bins.
    # If target_freqs has specific length, we can infer nperseg.
    # freqs = fs/nperseg * k. 
    # Delta freq = target_freqs[1] - target_freqs[0]
    df = target_freqs[1] - target_freqs[0]
    nperseg = int(fs / df)
    print(f"Inferred nperseg: {nperseg} (df={df:.4f} Hz, fs={fs} Hz)")

    sim_psd_res = compute_psd(
        res.lfp, 
        fs=fs, 
        nperseg=nperseg, 
        roi="mean",
        freq_range=(target_freqs[0], target_freqs[-1])
    )
    
    sim_psd = sim_psd_res.psd
    sim_freqs = sim_psd_res.freqs
    
    print(f"Sim PSD shape: {sim_psd.shape}, Freqs shape: {sim_freqs.shape}")
    
    # 6. Check Loss
    # Default loss is usually 'psd_mse'
    loss_fn = build_loss("psd_mse")
    
    # Note: psd_mse might require inputs to be same shape
    # Let's interpolate sim_psd to target_freqs just in case, though they should match if nperseg is right.
    if sim_psd.shape != target_psd.shape:
        print("Shapes mismatch, doing interp...")
        sim_psd = np.interp(target_freqs, sim_freqs, sim_psd.ravel())
        sim_freqs = target_freqs
        
    loss_val = loss_fn(sim_psd, target_psd, freqs=sim_freqs)
    print(f"\nRe-calculated Loss: {loss_val}")
    print(f"Loss from summary:  {summary['best_loss']}")
    
    # 7. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(target_freqs, target_psd, 'k-', lw=2, label="Target (Type4 True)")
    plt.plot(sim_freqs, sim_psd, 'r--', lw=2, label=f"Best Fit (A={best_params.get('A'):.2f}, B={best_params.get('B'):.2f}, G={best_params.get('G'):.2f})")
    plt.title(f"Optimization Check\nLoss: {loss_val:.2e}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_plot = PROJECT_ROOT / "debug_opt_plot.png"
    plt.savefig(out_plot)
    print(f"Plot saved to {out_plot}")

if __name__ == "__main__":
    check_optimization_result()
