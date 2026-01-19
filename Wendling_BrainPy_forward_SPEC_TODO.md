# BrainPy Wendling Single Node (Forward / Evolution Only) — SPEC & TODO
> You will provide the **neurolib Wendling model code** (so ODEs/equations already exist).  
 - The neurolib_wendling codes are in `./neurolib_wendling/`. This includes the ODEs/equations and the parameters used in the model.
> This SPEC focuses on **how to port & run it in BrainPy** with a minimal, single-folder layout: `./core_wendling/`.

---

## 0) Goal
Implement a **single-node Wendling neural mass model** in **BrainPy** for **forward simulation only**:
- deterministic/reproducible runs (seeded),
- flexible external input (baseline + periodic/ASSR-like),
- LFP proxy + selected state monitors,
- clean interfaces that won’t block later multi-node / whole-brain extension.
- **Readability should be improved for human understanding. Certain level of linear thinking and detailed comments on functions, input/output parameters, and the logic of the code are required.**

Key BrainPy concepts to align with:
- A `DynamicalSystem` typically defines the **per-time-step evolving function**; multi-step running is handled by **DSRunner** or **IntegratorRunner**.  
  References: BrainPy `DynamicalSystem`, DSRunner, IntegratorRunner docs.

---

## 1) Folder Layout (single folder, per your request)
```
./core_wendling/
  wendling_bp.py         # Wendling model class + ODE wrapper + LFP proxy
  stim.py                # input generators (baseline / sine / window)
  run_forward.py         # run entry: config -> build -> run -> save
  plot_basic.py          # plot time series + PSD
  sanity_check.py        # minimal checks: no NaN/Inf, reproducibility
  config.yaml            # dt/duration/params/stim/monitors/seed
  README.md              # one-page usage
```

---

## 2) Model SPEC (port from neurolib)
### 2.1 Mandatory mapping block (top of `wendling_bp.py`)
Because neurolib already defines ODEs, your biggest risk is **naming + units + injection site**.
At the top of `wendling_bp.py`, include a mapping comment like:

- **State vector**
  - `x1 = ...`  (neurolib: `<name>`)
  - `x2 = ...`
  - ...
- **Parameter dict**
  - `A, a, B, b, G, g, ...`
  - sigmoid params `e0, v0, r`
- **Time units**
  - confirm whether ODEs assume **seconds** or **milliseconds**
  - decide `dt_ms` in config and convert inside code consistently
- **Input injection site**
  - where `u(t)` is added (which population / which equation term)
- **Output (LFP proxy)**
  - `lfp(t) = ...` explicitly define the scalar observable (the choice of LFP proxy should be documenteds)

### 2.2 Two choices you must “lock”
1) **Time base**: ms vs s, and the conversion rules (dt, time constants, stimulus frequency)
2) **LFP proxy**: exact formula and sign convention

---

## 3) BrainPy Implementation Requirements (forward only)
### 3.1 Model class
In `wendling_bp.py`, implement:

**`class WendlingSingleNode(bp.DynamicalSystem)`**
- holds state variables as BrainPy/JAX arrays
- holds parameters (from config)
- defines the per-step evolution
- provides:
  - `reset(seed: int | None)`
  - `lfp()` → scalar observable at current step
  - `get_state_dict()` (optional but helpful)

> Principle: model defines step-wise evolution; the runner handles multi-step execution.

### 3.2 Runner choice (pick ONE and stick with it)
You only need one of these (either is fine for forward simulation):
- **IntegratorRunner** (recommended if you define ODE derivatives and integrate)
- **DSRunner** (good when you implement explicit per-step update and want flexible inputs/monitors)

### 3.3 Monitors (must-have)
Store outputs in `runner.mon[...]`:
- `lfp`
- 2–4 key internal states (you decide which are most informative)
- optional debug switch to record all states

---

## 4) Input / Stimulus API (minimal but future-proof)
Implement in `stim.py`:

### 4.1 baseline drive
- constant baseline input with optional on/off window

### 4.2 periodic drive (ASSR minimal)
- `u(t) = amp * sin(2π f t + phase)` where `f ∈ {20,30,40}` Hz
- support onset/offset in ms

### 4.3 Injection rule (must be documented)
Document in code and README:
- which equation receives `u(t)` and with what operation (e.g., `+ u(t)`)

---

## 5) `config.yaml` minimal schema (fixed keys)
```yaml
seed: 0
dt_ms: 0.1
duration_ms: 20000

params:
  # keep names consistent with neurolib to reduce errors
  A: 3.25
  a: 100.0
  B: 22.0
  b: 50.0
  G: 10.0
  g: 350.0
  e0: 2.5
  v0: 6.0
  r: 0.56
  # add the rest as in neurolib

stim:
  kind: "sine"        # "baseline" | "sine"
  f_hz: 40
  amp: 1.0
  phase: 0.0
  onset_ms: 500
  offset_ms: 19500

monitors:
  - lfp
  - x_pyr
  - x_inh_fast
```

---

## 6) Definition of Done (acceptance criteria)
1) **Runs**
- `python core_wendling/run_forward.py` completes with no error
- `runner.mon['lfp']` length ≈ `duration_ms / dt_ms`

2) **Reproducible**
- same config + seed → identical (or numerically near-identical) `lfp`

3) **Stimulus sanity**
- increasing `amp` yields a visible increase in response (time-domain amplitude and/or a spectral bump near `f_hz`)

4) **Minimal plotting**
- `plot_basic.py` produces:
  - `lfp(t)`
  - PSD (simple FFT is OK)

---

## 7) TODO Checklist (for the AI implementer)
### A) Porting & alignment (highest priority)
- [ ] Paste neurolib ODEs into `wendling_bp.py`
- [ ] Write the mapping block (states/params/units/injection/LFP proxy)
- [ ] Verify **ms vs s** conversions (dt, a/b/g time constants, stimulus freq)
- [ ] Define `lfp()` exactly and test sign convention

### B) Forward runner
- [ ] Choose Runner: IntegratorRunner **or** DSRunner
- [ ] Wire monitors (include callable for `lfp` if needed)
- [ ] Implement `run_forward.py`:
  - load YAML
  - build model
  - build stimulus `u(t)`
  - run
  - save outputs (`npz` recommended)

### C) Inputs
- [ ] Implement baseline + sine
- [ ] Implement onset/offset windowing
- [ ] Confirm `f_hz` conversion is correct under your time base

### D) Sanity checks
- [ ] 5–10s run: assert no NaN/Inf in `lfp`
- [ ] rerun with same seed: outputs match
- [ ] monitor keys exist and shapes correct

### E) Plot
- [ ] Plot time series
- [ ] Plot PSD (FFT)
- [ ] Optional: compare 20/30/40 Hz in one script (not required)

---

## 8) What YOU should read in BrainPy docs (forward-only, minimal set)
You do **not** need to read all tutorials. Only these pages are needed to understand the runner/inputs/monitors pattern:

- **Quickstart: Simulation** (runner usage, inputs, `runner.mon`)  
  https://brainpy.readthedocs.io/quickstart/simulation.html

- **DynamicalSystem** (what the model class conceptually provides)  
  https://brainpy.readthedocs.io/apis/generated/brainpy.DynamicalSystem.html

- **IntegratorRunner** (if you choose integrator-driven forward simulation)  
  https://brainpy.readthedocs.io/apis/generated/brainpy.IntegratorRunner.html

- **DSRunner** (if you choose DSRunner; especially inputs/monitors patterns)  
  https://brainpy.readthedocs.io/apis/generated/brainpy.DSRunner.html

---

## 9) Notes for future extension (do NOT implement now)
When you later move to multi-node / whole-brain, avoid refactoring by keeping:
- `set_external_drive(u_t)` as the single input entry point
- `get_output()` / `lfp()` as the single observable output
- params organized as a dict that can be vectorized across nodes
