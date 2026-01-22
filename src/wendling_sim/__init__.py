"""
Wendling BrainPy Simulator + Nevergrad Optimization

Public API:
    simulate(sim_cfg, model_cfg, stim_cfg, noise_cfg, monitor_cfg) -> SimResult
    optimize(opt_cfg, objective_cfg) -> OptResult
"""

from wendling_sim.api import simulate, optimize

__all__ = ['simulate', 'optimize']
__version__ = '0.3.0'
