"""
DSRunner wrapper for Wendling simulation.

Produces standardized SimResult from BrainPy DSRunner execution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import brainpy as bp
import brainpy.math as bm

from wendling_sim.model.wendling_single import WendlingSingleNode
from wendling_sim.model.wendling_network import WendlingNetwork
from wendling_sim.model.params import merge_params, STANDARD_PARAMS
from wendling_sim.connectivity.io import load_connectivity, Connectivity


@dataclass
class SimResult:
    """
    Standardized simulation result.
    
    Attributes:
        t_s: Time array (seconds)
        lfp: LFP proxy array (time-major, shape (T, N))
        states: Optional dict of monitored state variables
        meta: Simulation metadata
    """
    t_s: np.ndarray
    lfp: np.ndarray
    states: Optional[Dict[str, np.ndarray]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def t(self) -> np.ndarray:
        """Alias for time array (seconds)."""
        return self.t_s
    
    def save(self, path: str):
        """Save to npz file."""
        data = {'t_s': self.t_s, 'lfp': self.lfp, 'meta': np.array(self.meta, dtype=object)}
        if self.states is not None:
            data['states'] = np.array(self.states, dtype=object)
        np.savez(path, **data)
    
    @classmethod
    def load(cls, path: str) -> 'SimResult':
        """Load from npz file."""
        data = np.load(path, allow_pickle=True)
        t_s = data['t_s'] if 't_s' in data.files else data['t']
        states = data['states'].item() if 'states' in data.files else None
        meta = data['meta'].item() if 'meta' in data.files else {}
        return cls(t_s=t_s, lfp=data['lfp'], states=states, meta=meta)


def _parse_time_cfg(sim_cfg: Dict[str, Any]) -> Tuple[float, float]:
    """Extract dt and duration in seconds."""
    if 'dt_s' in sim_cfg:
        dt_s = sim_cfg['dt_s']
    else:
        dt_ms = sim_cfg.get('dt_ms', 0.1)
        dt_s = dt_ms / 1000.0
    if 'duration_s' in sim_cfg:
        duration_s = sim_cfg['duration_s']
    else:
        duration_ms = sim_cfg.get('duration_ms', 5000.0)
        duration_s = duration_ms / 1000.0
    return float(dt_s), float(duration_s)


def _prepare_params(params: Optional[Dict[str, Any]], noise_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge base params with overrides and noise settings."""
    merged = merge_params(STANDARD_PARAMS, params or {})
    if 'p_mean' in noise_cfg:
        merged['p_mean'] = noise_cfg['p_mean']
    if 'p_sigma' in noise_cfg:
        merged['p_sigma'] = noise_cfg['p_sigma']
    return merged


def _resolve_integrator(sim_cfg: Dict[str, Any]):
    """Resolve integrator choice for DSRunner."""
    name = sim_cfg.get('integrator', 'rk4')
    try:
        from brainpy.integrators import ode

        if name == 'rk4':
            return ode.RK4()
        if name in ('euler',):
            return ode.Euler()
        if name in ('exp_euler', 'expeuler', 'exponential_euler'):
            return ode.ExponentialEuler()
    except Exception:
        return None
    return None


def run_simulation(
    sim_cfg: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    network_cfg: Optional[Dict[str, Any]] = None,
    monitors: Optional[List[str]] = None,
    noise_cfg: Optional[Dict[str, Any]] = None,
) -> SimResult:
    """
    Run forward simulation using BrainPy DSRunner.
    
    Args:
        sim_cfg: Simulation config (dt_s/duration_s or dt_ms/duration_ms, jit, seed)
        params: Local Wendling parameters (merged with defaults)
        network_cfg: Connectivity + coupling configuration (optional)
        monitors: State variables to monitor (e.g., ['y0', 'y1'])
        noise_cfg: Noise configuration (p_mean, p_sigma, seed, mode/shared)
    
    Returns:
        SimResult with time-major LFP and metadata.
    """
    sim_cfg = sim_cfg or {}
    noise_cfg = noise_cfg or {}
    network_cfg = network_cfg or {}

    # Time and execution settings
    dt_s, duration_s = _parse_time_cfg(sim_cfg)
    jit = sim_cfg.get('jit', True)
    progress_bar = sim_cfg.get('progress_bar', False)
    debug = sim_cfg.get('debug', False)
    seed = noise_cfg.get('seed', sim_cfg.get('seed', None))
    integrator = _resolve_integrator(sim_cfg)

    # Set seed for reproducibility
    if seed is not None:
        bm.random.seed(seed)
        np.random.seed(seed)

    # Connectivity
    connectivity: Connectivity = load_connectivity(network_cfg)
    is_network = connectivity.n_nodes > 1 or network_cfg.get('enabled', False)

    # Merge parameters (local)
    full_params = _prepare_params(params, noise_cfg)

    # Build model
    if is_network:
        G_net = network_cfg.get('G_net', 0.0)
        model = WendlingNetwork(
            n_nodes=connectivity.n_nodes,
            params=full_params,
            W=connectivity.W,
            G_net=G_net,
            noise_cfg=noise_cfg,
        )
    else:
        model = WendlingSingleNode(params=full_params)

    # Set up monitors
    monitor_vars = ['lfp']
    if monitors:
        monitor_vars.extend(monitors)
    elif debug:
        monitor_vars.extend(['y0', 'y1', 'y2', 'y3', 'y4'])

    # Create DSRunner
    dsrunner_kwargs = dict(
        monitors=monitor_vars,
        dt=dt_s,
        jit=jit,
        progress_bar=progress_bar,
    )
    if integrator is not None:
        dsrunner_kwargs["integrator"] = integrator

    try:
        runner = bp.DSRunner(model, **dsrunner_kwargs)
    except TypeError:
        # Fallback for BrainPy versions that don't support integrator kwarg
        dsrunner_kwargs.pop("integrator", None)
        runner = bp.DSRunner(model, **dsrunner_kwargs)

    # Run simulation
    runner.run(duration_s)

    # Extract results
    t_s = np.asarray(runner.mon.ts)
    lfp = np.asarray(runner.mon['lfp'])
    if lfp.ndim == 1:
        lfp = lfp[:, None]

    # Extract monitored states
    states = None
    if monitors or debug:
        states = {}
        for var in monitor_vars:
            if var == 'lfp':
                continue
            if var in runner.mon:
                arr = np.asarray(runner.mon[var])
                if arr.ndim == 1:
                    arr = arr[:, None]
                states[var] = arr

    # Build metadata
    meta = {
        'dt_s': dt_s,
        'duration_s': duration_s,
        'dt_ms': dt_s * 1000.0,
        'duration_ms': duration_s * 1000.0,
        'seed': seed,
        'noise_mode': noise_cfg.get('mode', 'gaussian'),
        'params': full_params,
        'backend': 'DSRunner',
        'jit': jit,
        'n_nodes': connectivity.n_nodes,
        'G_net': network_cfg.get('G_net', 0.0),
        'connectivity_meta': connectivity.meta,
    }

    return SimResult(t_s=t_s, lfp=lfp, states=states, meta=meta)
