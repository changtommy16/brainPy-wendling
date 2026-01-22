"""
Public API for Wendling simulation and optimization.

Only two functions exposed:
    simulate() - Forward simulation
    optimize() - Parameter optimization
"""

from typing import Dict, Any, Optional
import numpy as np

from wendling_sim.sim.runner import run_simulation, SimResult
from wendling_sim.features.psd import compute_psd
from wendling_sim.loss.registry import build_loss
from wendling_sim.optimize.search_space import SearchSpace
from wendling_sim.optimize.nevergrad_engine import NevergradOptimizer, OptResult


def simulate(
    sim_cfg: Dict[str, Any] = None,
    model_cfg: Dict[str, Any] = None,
    network_cfg: Dict[str, Any] = None,
    stim_cfg: Dict[str, Any] = None,
    noise_cfg: Dict[str, Any] = None,
    monitor_cfg: Dict[str, Any] = None,
) -> SimResult:
    """
    Run forward simulation (single-node or network).
    
    Args:
        sim_cfg: Simulation config (dt_s/duration_s or dt_ms/duration_ms, jit, seed)
        model_cfg: Local Wendling parameters (A, B, G, a, b, g, etc.)
        network_cfg: Connectivity + coupling configuration (n_nodes, W, generator, G_net)
        stim_cfg: Reserved for future external stimulus
        noise_cfg: Noise config (mode, seed, p_mean, p_sigma)
        monitor_cfg: Monitor config (variables to record)
    
    Returns:
        SimResult with time-major LFP, states, and meta
    """
    sim_cfg = sim_cfg or {}
    model_cfg = model_cfg or {}
    network_cfg = network_cfg or {}
    noise_cfg = noise_cfg or {}
    monitor_cfg = monitor_cfg or {}

    monitors = monitor_cfg.get('variables') or monitor_cfg.get('monitors')

    return run_simulation(
        sim_cfg=sim_cfg,
        params=model_cfg,
        network_cfg=network_cfg,
        monitors=monitors,
        noise_cfg=noise_cfg,
    )


def optimize(
    opt_cfg: Dict[str, Any],
    objective_cfg: Dict[str, Any],
    target_psd: Optional[np.ndarray] = None,
    target_freqs: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> OptResult:
    """
    Run nevergrad optimization to find parameters minimizing PSD loss.
    
    Args:
        opt_cfg: Optimization config
            - budget: Number of evaluations
            - optimizer: Nevergrad optimizer name
            - num_workers: Parallel workers
            - seed: Random seed
            - search_space: Bounds dict or SearchSpace
        objective_cfg: Objective config
            - sim_cfg: Simulation config for each evaluation
            - psd_cfg: PSD computation config (fs, nperseg, roi, etc.)
            - loss_name: Loss function name
            - loss_cfg: Loss function config
            - network_cfg: Connectivity for evaluation
            - noise_cfg: Noise settings for evaluation
        target_psd: Target PSD array to fit
        target_freqs: Target frequency array (optional)
        verbose: Print progress
    
    Returns:
        OptResult with best_params, best_loss, history
    """
    opt_cfg = opt_cfg or {}
    objective_cfg = objective_cfg or {}

    # Extract configs
    budget = opt_cfg.get('budget', 100)
    optimizer_name = opt_cfg.get('optimizer', 'NGOpt')
    num_workers = opt_cfg.get('num_workers', 1)
    seed = opt_cfg.get('seed', None)

    # Build search space
    search_space_cfg = opt_cfg.get('search_space', None)
    if isinstance(search_space_cfg, SearchSpace):
        search_space = search_space_cfg
    elif isinstance(search_space_cfg, dict):
        search_space = SearchSpace(bounds=search_space_cfg)
    else:
        search_space = SearchSpace.default()

    # Extract simulation config
    sim_cfg = objective_cfg.get('sim_cfg', {})
    psd_cfg = objective_cfg.get('psd_cfg', {})
    network_cfg = objective_cfg.get('network_cfg', opt_cfg.get('network_cfg', {}))
    noise_cfg = objective_cfg.get('noise_cfg', {})
    loss_name = objective_cfg.get('loss_name', 'psd_mse')
    loss_cfg = objective_cfg.get('loss_cfg', {})

    if target_psd is None:
        raise ValueError("target_psd is required for optimization.")

    # Build loss function
    if 'losses' in objective_cfg and objective_cfg['losses'] is not None:
        loss_components = []
        for item in objective_cfg['losses']:
            if isinstance(item, str):
                loss_components.append({'name': item, 'weight': 1.0, 'cfg': {}})
            else:
                loss_components.append(
                    {
                        'name': item['name'],
                        'weight': float(item.get('weight', 1.0)),
                        'cfg': item.get('cfg', {}),
                    }
                )

        component_fns = [
            (c['name'], c['weight'], build_loss(c['name'], c['cfg']))
            for c in loss_components
        ]

        def loss_fn(psd, target, freqs):
            total = 0.0
            details = {}
            for name, weight, fn in component_fns:
                val = float(fn(psd, target, freqs))
                details[name] = val
                total += weight * val
            details['_total'] = total
            return total, details
    else:
        loss_fn = build_loss(loss_name, loss_cfg)

    # Use fixed seed for deterministic evaluation
    eval_seed = seed or 42
    noise_cfg_eval = dict(noise_cfg)
    noise_cfg_eval.setdefault('seed', eval_seed)

    # Sampling frequency for PSD
    if 'fs' in psd_cfg and psd_cfg['fs'] is not None:
        fs = psd_cfg['fs']
    else:
        dt_s = sim_cfg.get('dt_s')
        if dt_s is None:
            dt_ms = sim_cfg.get('dt_ms', 0.1)
            dt_s = dt_ms / 1000.0
        fs = 1.0 / dt_s

    roi = psd_cfg.get('roi', 'none')
    roi_nodes = psd_cfg.get('roi_nodes', None)

    def objective_fn(params: Dict[str, float]) -> float:
        """Objective: simulate -> PSD -> loss."""
        # Separate network gain from local params if present
        params_local = dict(params)
        candidate_G_net = params_local.pop('G_net', network_cfg.get('G_net', 0.0))

        net_cfg = dict(network_cfg or {})
        net_cfg['G_net'] = candidate_G_net

        # Run simulation with candidate params (fixed seed for reproducibility)
        result = run_simulation(
            sim_cfg=sim_cfg,
            params=params_local,
            network_cfg=net_cfg,
            noise_cfg=noise_cfg_eval,
            monitors=None,
        )

        # Compute PSD
        psd_result = compute_psd(
            result.lfp,
            fs=fs,
            nperseg=psd_cfg.get('nperseg', None),
            freq_range=psd_cfg.get('freq_range', (1.0, 50.0)),
            roi=roi,
            roi_nodes=roi_nodes,
        )

        # Compute loss
        psd_array = psd_result.psd
        target_array = target_psd
        if getattr(psd_array, "ndim", 1) == 2 and psd_array.shape[1] == 1:
            psd_array = psd_array[:, 0]
        if getattr(target_array, "ndim", 1) == 2 and target_array.shape[1] == 1:
            target_array = target_array[:, 0]
        res = loss_fn(psd_array, target_array, psd_result.freqs)
        return res

    # Create and run optimizer
    optimizer = NevergradOptimizer(
        search_space=search_space,
        objective_fn=objective_fn,
        budget=budget,
        optimizer_name=optimizer_name,
        num_workers=num_workers,
        seed=seed,
    )

    return optimizer.run(verbose=verbose)
