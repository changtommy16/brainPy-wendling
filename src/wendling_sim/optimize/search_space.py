"""
Search space definition for nevergrad optimization.

Default search space (Option 2 + G_net): {A, B, G, a, b, g, G_net}
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

try:
    import nevergrad as ng
except ImportError:
    ng = None


@dataclass
class SearchSpace:
    """
    Parameter search space for optimization.
    
    Attributes:
        bounds: Dict mapping param name to (low, high) bounds
        log_scale: Set of param names to use log-scale
        fixed: Dict of fixed parameter values (not optimized)
    """
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    log_scale: set = field(default_factory=set)
    fixed: Dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def default(cls) -> 'SearchSpace':
        """Create default search space for A, B, G, a, b, g, and G_net."""
        return cls(
            bounds={
                'A': (1.0, 10.0),
                'B': (5.0, 40.0),
                'G': (5.0, 40.0),
                'a': (50.0, 200.0),
                'b': (20.0, 120.0),
                'g': (200.0, 1200.0),
                'G_net': (0.0, 5.0),
            },
            log_scale={'a', 'b', 'g'},  # Time constants in log scale
            fixed={},
        )
    
    def to_nevergrad(self) -> 'ng.p.Dict':
        """Convert to nevergrad parametrization."""
        if ng is None:
            raise ImportError("nevergrad not installed. Run: pip install nevergrad")
        
        params = {}
        for name, (low, high) in self.bounds.items():
            if name in self.log_scale:
                # Log-scale parameter
                params[name] = ng.p.Log(lower=low, upper=high)
            else:
                # Linear-scale parameter
                params[name] = ng.p.Scalar(lower=low, upper=high)
        
        return ng.p.Dict(**params)
    
    def clip(self, params: Dict[str, float]) -> Dict[str, float]:
        """Clip parameters to bounds."""
        clipped = {}
        for name, value in params.items():
            if name in self.bounds:
                low, high = self.bounds[name]
                clipped[name] = float(np.clip(value, low, high))
            else:
                clipped[name] = value
        return clipped
    
    def sample_random(self, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """Sample random parameters within bounds."""
        if rng is None:
            rng = np.random.default_rng()
        
        params = {}
        for name, (low, high) in self.bounds.items():
            if name in self.log_scale:
                # Log-uniform sampling
                params[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            else:
                # Uniform sampling
                params[name] = float(rng.uniform(low, high))
        
        # Add fixed params
        params.update(self.fixed)
        return params
