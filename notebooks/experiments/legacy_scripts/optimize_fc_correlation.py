"""
Optimize Wendling Model Parameters for FC Correlation
======================================================
Goal: Achieve FC-FC correlation >= 0.4 with empirical HCP data

Key parameters to optimize:
- K_gl: Global coupling strength
- heterogeneity: Node parameter variation
- B_base, G_base: Base inhibitory parameters
"""

if __name__ == '__main__':
    import sys
    sys.path.insert(0, r'C:\Epilepsy_project\Neurolib_desktop\Neurolib_package')
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    from neurolib.models.wendling import WendlingModel
    from neurolib.utils.loadData import Dataset
    import time
    
    print("=" * 70)
    print("Wendling FC Optimization - Target: r >= 0.4")
    print("=" * 70)
    
    # Load HCP data
    print("\nLoading HCP dataset...")
    ds = Dataset("hcp")
    N = ds.Cmat.shape[0]
    empirical_fc = np.mean(ds.FCs, axis=0)
    
    print(f"  Regions: {N}")
    print(f"  Subjects: {len(ds.FCs)}")
    print(f"  Empirical FC mean: {np.mean(np.abs(empirical_fc[~np.eye(N, dtype=bool)])):.3f}")
    
    # Parameter grid
    K_gl_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    het_values = [0.0, 0.1, 0.2, 0.3]
    
    results = []
    best_corr = -1
    best_params = None
    
    print(f"\nParameter sweep: {len(K_gl_values)} x {len(het_values)} = {len(K_gl_values)*len(het_values)} combinations")
    print("-" * 70)
    
    for K_gl in K_gl_values:
        for het in het_values:
            start = time.time()
            
            # Create model
            model = WendlingModel(Cmat=ds.Cmat, Dmat=ds.Dmat, heterogeneity=het, seed=42)
            model.params['duration'] = 10000  # 10 seconds
            model.params['dt'] = 0.1
            model.params['K_gl'] = K_gl
            
            # Run
            try:
                model.run()
                
                # Extract signals (PSP)
                signals = np.zeros((N, len(model.t)))
                for i in range(N):
                    signals[i, :] = model.y1[i, :] - model.y2[i, :] - model.y3[i, :]
                
                # Discard transient (first 2 seconds)
                discard_idx = int(2000 / 0.1)
                signals_clean = signals[:, discard_idx:]
                
                # Compute simulated FC
                sim_fc = np.corrcoef(signals_clean)
                
                # FC-FC correlation
                sim_fc_flat = sim_fc[~np.eye(N, dtype=bool)]
                emp_fc_flat = empirical_fc[~np.eye(N, dtype=bool)]
                fc_corr, _ = pearsonr(sim_fc_flat, emp_fc_flat)
                
                elapsed = time.time() - start
                
                results.append({
                    'K_gl': K_gl, 'het': het, 'fc_corr': fc_corr,
                    'sim_fc_mean': np.mean(np.abs(sim_fc_flat))
                })
                
                status = "***" if fc_corr > 0.4 else ""
                print(f"  K_gl={K_gl:.2f}, het={het:.1f}: FC-corr={fc_corr:.3f}, "
                      f"sim_FC_mean={np.mean(np.abs(sim_fc_flat)):.3f} ({elapsed:.1f}s) {status}")
                
                if fc_corr > best_corr:
                    best_corr = fc_corr
                    best_params = {'K_gl': K_gl, 'het': het}
                    best_sim_fc = sim_fc.copy()
                
            except Exception as e:
                print(f"  K_gl={K_gl:.2f}, het={het:.1f}: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nBest FC-FC correlation: {best_corr:.3f}")
    print(f"Best parameters: K_gl={best_params['K_gl']}, heterogeneity={best_params['het']}")
    
    if best_corr >= 0.4:
        print("\n✅ TARGET ACHIEVED: FC correlation >= 0.4!")
    else:
        print(f"\n⚠️ Target not achieved. Best: {best_corr:.3f} < 0.4")
        print("   Try: longer simulation, different B/G parameters, or finer grid")
    
    # Plot best result
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    im0 = axes[0].imshow(empirical_fc, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title(f"Empirical FC")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    im1 = axes[1].imshow(best_sim_fc, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title(f"Simulated FC (K_gl={best_params['K_gl']}, het={best_params['het']})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # FC-FC scatter
    sample_idx = np.random.choice(len(emp_fc_flat), min(2000, len(emp_fc_flat)), replace=False)
    axes[2].scatter(emp_fc_flat[sample_idx], best_sim_fc[~np.eye(N, dtype=bool)][sample_idx], 
                   s=5, alpha=0.3)
    axes[2].plot([-1, 1], [-1, 1], 'r--', lw=2)
    axes[2].set_xlabel("Empirical FC")
    axes[2].set_ylabel("Simulated FC")
    axes[2].set_title(f"FC-FC Correlation: r = {best_corr:.3f}")
    axes[2].set_xlim(-1, 1)
    axes[2].set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig("./results/fc_optimization_result.png", dpi=150)
    plt.close()
    print(f"\nSaved: ./results/fc_optimization_result.png")
    
    # Heatmap of results
    import pandas as pd
    df = pd.DataFrame(results)
    pivot = df.pivot(index='het', columns='K_gl', values='fc_corr')
    
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=0.5, aspect='auto')
    plt.colorbar(label='FC Correlation')
    plt.xticks(range(len(K_gl_values)), [f'{k:.2f}' for k in K_gl_values])
    plt.yticks(range(len(het_values)), [f'{h:.1f}' for h in het_values])
    plt.xlabel('K_gl (Global Coupling)')
    plt.ylabel('Heterogeneity')
    plt.title('FC Correlation Parameter Sweep')
    
    # Annotate
    for i, h in enumerate(het_values):
        for j, k in enumerate(K_gl_values):
            val = pivot.loc[h, k]
            color = 'white' if val < 0.25 else 'black'
            plt.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig("./results/fc_optimization_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved: ./results/fc_optimization_heatmap.png")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
