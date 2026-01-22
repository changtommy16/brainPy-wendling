# WENDLING Subnetworks & Whole‑Brain Modeling — SPEC & TODO (v0.3)

> **Status**: ready-to-implement next milestone  
> **Scope**: extend your BrainPy-native Wendling single-node simulator → **subnetworks** → **whole brain**  
> **Public API (kept minimal)**: `simulate()` and `optimize()`  
> **Fit target**: PSD features computed from **LFP proxy**

---

## 0. Index (導航)

- [1. Goals](#1-goals)
- [2. Locked Decisions](#2-locked-decisions)
- [3. System Definition](#3-system-definition)
- [4. Connectivity Interface](#4-connectivity-interface)
- [5. Delays Interface](#5-delays-interface)
- [6. Simulation Pipeline](#6-simulation-pipeline)
- [7. PSD Feature Pipeline](#7-psd-feature-pipeline)
- [8. Optimization Pipeline (nevergrad)](#8-optimization-pipeline-nevergrad)
- [9. Project Layout](#9-project-layout)
- [10. TODO Roadmap](#10-todo-roadmap)
- [11. Acceptance Tests](#11-acceptance-tests)
- [12. Non‑goals (for now)](#12-non-goals-for-now)

---

## 1. Goals

### 1.1 Near-term (v0.3 → v0.4)
- Support **N-node Wendling network** with:
  - shared local parameters across nodes
  - general connectivity input (file / generator / custom)
  - optional delays (on/off)
  - BrainPy-native multi-step execution (Runner + integrator)
- Compute **PSD features** from LFP and define an initial **PSD loss**.

### 1.2 Mid-term (v0.5+)
- Whole-brain connectomes (e.g., 68–400+ nodes)
- ROI/parcel aggregation
- Delay ON support validated on toy + real connectomes
- Performance profiling + JIT strategies

---

## 2. Locked Decisions

These are now **hard constraints** to prevent refactors later.

### 2.1 Array/time conventions
- **Time-major output**: `lfp[t, i]` (T × N)
- **Internal time unit**: seconds (`dt_s`, `tau_s`), even if user provides ms.

### 2.2 Connectivity semantics
- **W[i, j] means j → i** (target row i, source col j)
- Default: remove self-loops (`diag(W)=0`), unless explicitly kept.

### 2.3 Coupling variable (what travels along edges)
- Default: `out_j(t) = S(v_pyr_j(t))`
- `v_pyr = y1 - y2 - y3` (same ingredients as LFP proxy)

### 2.4 Fitting observable (what you fit to PSD)
- `LFP_i(t) = y1_i(t) - y2_i(t) - y3_i(t)`
- PSD features are computed from `LFP_i(t)` (or ROI-aggregated LFP).

### 2.5 Injection site
- Network input enters the **y6 equation** (pyramidal drive term), consistent with your diagram.

---

## 3. System Definition

### 3.1 Per-node local states
Vectorize each local state across nodes:

- `y0..y4`: shape (N,)
- `y5..y9`: shape (N,)

### 3.2 Sigmoid
`S(v) = 2e0 / (1 + exp(r*(v0 - v)))`

### 3.3 LFP proxy
`LFP_i(t) = y1_i - y2_i - y3_i`

### 3.4 Coupling term

Let:
- `W`: (N, N), with **W[i, j] = j → i**
- `G_net`: global coupling gain
- `out(t)`: coupling variable vector (N,)
- optional delays `tau_ij` (seconds)

**No delays (delay.enabled = false)**:
- `u_net(t) = G_net * (W @ out(t))`

**With delays (delay.enabled = true)**:
- `u_net_i(t) = G_net * Σ_j W[i, j] * out_j(t - tau_ij)`
- If using tract length L and velocity v: `tau_ij = L_ij / v`

### 3.5 Injection into y6
In your network update, the y6 derivative for each node i becomes:

- `dy1 = y6`
- `dy6 = A*a*( C2*S(C1*y0) + p(t) + u_net_i(t) + u_stim_i(t) ) - 2*a*y6 - a^2*y1`

Notes:
- `p(t)` can be per-node noise or shared noise, but must be controlled by config.
- `u_stim` is optional (can be `0` for PSD-only baseline).

---

## 4. Connectivity Interface

### 4.1 Internal representation (runtime standard)
All sources must be converted to:

- `W: float32[N, N]`
- optional `L: float32[N, N]` or `tau_s: float32[N, N]`
- `labels: list[str]` optional
- `meta: dict` (parcellation name, subject id, notes)

### 4.2 Supported sources

#### A) File
- W: `.npy` or `.csv`
- Optional:
  - L: `.npy` / `.csv`
  - labels: `.txt` / `.csv` / `.json`

#### B) Generator (controlled graphs)
- `erdos_renyi`
- `small_world`
- `ring_lattice`
- `stochastic_block_model` (modular subnetworks)

Generator options must include:
- seed
- sparsity (0–1)
- weight distribution (e.g., lognormal/normal/uniform)
- symmetry (optional)

#### C) Custom callable
User provides a Python callable:
- `fn(n_nodes: int, seed: int, **kwargs) -> (W, optional L/tau, labels, meta)`

### 4.3 Normalization (recommended defaults)
Provide `normalize` option:
- `none`
- `row_sum` (normalize incoming weights per target i)
- `max` (divide by global max)

Default: `row_sum` (helps avoid exploding inputs when W is dense).

---

## 5. Delays Interface

### 5.1 Config
- `delay.enabled: bool`
- `delay.units: "s" | "ms"` (converted to seconds internally)
- Provide either:
  - `tau_path` (direct delays), OR
  - `L_path` + `velocity`

### 5.2 Implementation policy
- Must be **BrainPy-native** (prefer BrainPy delay variables / DDE-capable integrators)
- Avoid hand-written ring buffers unless absolutely necessary.

---

## 6. Simulation Pipeline

### 6.1 simulate() (public API)
**Goal**: one entrypoint that works for single-node and N-node

Inputs (minimal):
- `params_local` (A,B,G,a,b,g,e0,v0,r,p_mean,p_sigma,...)
- `network_cfg` (n_nodes, W source, coupling config, delays config)
- `sim_cfg` (dt_s, duration_s, backend="DSRunner", monitors, storage)

Outputs:
- `SimResult`
  - `t_s: float32[T]`
  - `lfp: float32[T, N]`  ← time-major locked
  - optional `states` (off by default for wholebrain)
  - `meta` (seed, config hash, connectivity meta)

### 6.2 Runner strategy
- Phase 1: `DSRunner` monitors `lfp` (and optionally minimal states for debug)
- Keep a small debug flag to record more states on demand.

---

## 7. PSD Feature Pipeline

### 7.1 features_psd() (internal)
Inputs:
- `lfp[T, N]`
- sampling info `fs = 1/dt_s`
- frequency range `[fmin, fmax]`
- ROI selection (optional)

Outputs:
- `PsdResult`
  - `freqs[F]`
  - `psd[F, N]` (or `psd[F, n_roi]`)
  - summary features (optional): bandpower, peak freq, peak power

### 7.2 ROI aggregation options
- `none`: keep per-node PSD
- `mean`: average LFP across nodes then PSD
- `subset`: only compute PSD for selected nodes

---

## 8. Optimization Pipeline (nevergrad)

### 8.1 optimize() (public API)
Input:
- target PSD (or target features)
- search space for:
  - local params: Option-2 subset `(A,B,G,a,b,g)`
  - network param: `G_net` (at minimum)
  - optional noise params: `p_sigma` etc.
- budget, num_workers, optimizer name

Loop:
1. `candidate = opt.ask()`
2. simulate(candidate) → PSD
3. loss = loss_fn(PSD, target_PSD)
4. `opt.tell(candidate, loss)`
5. end: `opt.provide_recommendation()`

Outputs:
- `OptResult`
  - best params
  - history (loss curve)
  - configs + seeds

---

## 9. Project Layout

Keep things discoverable with an **index-first structure**:

```
core_wendling/
  README.md
  INDEX.md                  # "導航入口"：links to docs + modules
  docs/
    SPEC_SINGLE_NODE.md
    SPEC_NETWORK_WHOLEBRAIN.md    # <-- this file
    CONFIG_SCHEMA.md
  model/
    wendling_single.py
    wendling_network.py     # N nodes, vectorized
  connectivity/
    io.py                   # load/save W/L/tau/labels, normalization, semantics checks
    generators.py           # ER / small-world / ring / SBM
  sim/
    runner.py               # DSRunner wrapper, output schema
  features/
    psd.py                  # welch PSD + summaries
  loss/
    psd_loss.py             # scalar loss
  optimize/
    nevergrad_engine.py
    search_space.py
  utils/
    config.py               # YAML/JSON config load + validation
    logging.py
```

---

## 10. TODO Roadmap

### Phase 1 — Subnetwork (no delays), PSD-only
- [x] Implement `WendlingNetwork` with vectorized states (N,)
- [x] Implement coupling `u_net = G_net * (W @ out)`
- [x] Inject coupling into y6 equation
- [x] DSRunner simulate() returns `lfp[T,N]` time-major
- [x] PSD features (Welch) + simple PSD MSE loss
- [x] nevergrad optimize() for `(A,B,G,a,b,g,G_net)` with budget control

### Phase 2 — Wholebrain (file connectome), still no delays
- [ ] Connectivity IO robust checks:
  - enforce W semantics
  - diag handling
  - normalization
- [ ] Labels support + ROI selection
- [ ] Performance profiling (JIT/vmap strategies; memory footprint)

### Phase 3 — Delays ON (switchable)
- [ ] Support `tau_s[N,N]` (direct) or `L[N,N] + v` (derive)
- [ ] Implement delay-enabled coupling variable fetch (BrainPy delay variable approach)
- [ ] Validate with toy delayed ring and compare expected phase lag patterns

### Phase 4 — Usability
- [ ] Example scripts:
  - random graph subnetwork
  - AAL-like wholebrain W load
  - optimize PSD for one ROI
- [ ] Output storage policy:
  - default save PSD summaries
  - optional save raw LFP

---

## 11. Acceptance Tests

### 11.1 Direction convention test (W semantics)
Construct 2-node network:
- set W[1,0]=1, all else 0
- confirm node 1 changes when node 0 is perturbed, but node 0 does not receive from node 1.

### 11.2 Coupling off sanity test
- set `G_net=0`
- confirm network equals N independent single-nodes (within numerical tolerance).

### 11.3 PSD smoke test
- simulate 10s
- PSD computes without NaNs
- peak frequency / bandpower metrics are finite

### 11.4 Delay toggle test
- run same config with delay.enabled off/on (with tau=0)
- confirm outputs match (within tolerance)

---

## 12. Non‑goals (for now)

- Region-wise local parameters (heterogeneity)
- Model comparison and advanced multi-objective fitting
- Multi-modal observables (ERP, ITPC) — later milestone
- Distributed/HPC execution

---
