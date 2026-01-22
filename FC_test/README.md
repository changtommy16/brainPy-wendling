# FC_test

Small experiment suite for BrainPy Wendling FC correlation vs HCP target FC.

Key ideas tested:
- Connectivity normalization (row_sum vs max)
- Noise level (p_sigma)
- Heterogeneity across nodes (A/B/G/p_mean)
- PSP-based FC vs BOLD-like FC

## Run

Default: run all cases (may be slow).

```bash
conda run -n brainpy_model python FC_test/brainpy_fc_sweep.py
```

Quick grid:

```bash
conda run -n brainpy_model python FC_test/brainpy_fc_sweep.py --quick
```

Shorter simulation (useful to confirm it is running):

```bash
conda run -n brainpy_model python FC_test/brainpy_fc_sweep.py --case neurolib_like_psp --quick --duration-ms 2000 --dt-ms 0.5
```

Single case:

```bash
conda run -n brainpy_model python FC_test/brainpy_fc_sweep.py --case neurolib_like_bold --quick
```

No normalization (closest to raw Cmat):

```bash
conda run -n brainpy_model python FC_test/brainpy_fc_sweep.py --case neurolib_like_psp_no_norm --quick
```

Notebook walkthrough:

- `FC_test/FC_debug_steps.ipynb`

Neurolib dataset note:
- `FC_test/brainpy_fc_sweep.py` can use the Neurolib HCP dataset (80 nodes, mean FC)
  via the `CONFIG_NEUROLIB_FULL` config.

Fine sweep (Neurolib target):

```bash
/home/changtommy16/miniconda3/envs/brainpy311/bin/python -u FC_test/brainpy_fc_neurolib_fine.py
```

Results are written under `FC_test/results/<run_id>/`.
Each case writes a JSON list plus a `summary.json` with the best FC correlation.

## Notes

- `neurolib_like_*` cases use max-normalized SC, higher noise, and smaller dt to
  better match the Neurolib Wendling setup.
- Heterogeneity is applied by sampling per-node A/B/G/p_mean.
- Use `--jit` if you want BrainPy to JIT compile.

If the data paths differ, pass `--struct-path` and `--fmri-path`.
