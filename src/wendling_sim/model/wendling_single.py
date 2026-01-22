"""
WendlingSingleNode: BrainPy DynamicalSystem for single-node Wendling model.

The model implements 10 ODEs representing a cortical column with:
- Pyramidal cells (main population)
- Excitatory interneurons
- Slow inhibitory interneurons (GABA_A,slow)
- Fast inhibitory interneurons (GABA_A,fast)

Reference: Wendling et al. (2002) - Epileptic fast activity can be explained 
by a model of impaired GABAergic dendritic inhibition.
"""

from typing import Dict, Optional
import brainpy as bp
import brainpy.math as bm
import numpy as np

from wendling_sim.model.params import STANDARD_PARAMS, merge_params


class WendlingSingleNode(bp.DynamicalSystem):
    """
    Single-node Wendling-Chauvel neural mass model.
    
    State variables (y0-y9):
        y0: PSP from pyramidal -> excitatory
        y1: PSP from excitatory -> pyramidal  
        y2: PSP from slow inhibitory -> pyramidal
        y3: PSP from fast inhibitory -> pyramidal
        y4: PSP from pyramidal -> slow inhibitory
        y5-y9: Derivatives of y0-y4
    
    LFP proxy: y1 - y2 - y3 (sum of PSPs at pyramidal population)
    """
    
    def __init__(self, params: Optional[Dict] = None, name: Optional[str] = None):
        super().__init__(name=name)
        
        # Merge with defaults
        p = merge_params(STANDARD_PARAMS, params or {})
        
        # Synaptic gains
        self.A = p['A']
        self.B = p['B']
        self.G = p['G']
        
        # Time constants
        self.a = p['a']
        self.b = p['b']
        self.g = p['g']
        
        # Sigmoid parameters
        self.e0 = p['e0']
        self.v0 = p['v0']
        self.r = p['r']
        
        # Connectivity (scaled by C)
        C = p['C']
        self.C1 = p['C1'] * C
        self.C2 = p['C2'] * C
        self.C3 = p['C3'] * C
        self.C4 = p['C4'] * C
        self.C5 = p['C5'] * C
        self.C6 = p['C6'] * C
        self.C7 = p['C7'] * C
        
        # Input parameters
        self.p_mean = p['p_mean']
        self.p_sigma = p['p_sigma']
        
        # State variables (10 ODEs)
        self.y0 = bm.Variable(bm.zeros(1))
        self.y1 = bm.Variable(bm.zeros(1))
        self.y2 = bm.Variable(bm.zeros(1))
        self.y3 = bm.Variable(bm.zeros(1))
        self.y4 = bm.Variable(bm.zeros(1))
        self.y5 = bm.Variable(bm.zeros(1))
        self.y6 = bm.Variable(bm.zeros(1))
        self.y7 = bm.Variable(bm.zeros(1))
        self.y8 = bm.Variable(bm.zeros(1))
        self.y9 = bm.Variable(bm.zeros(1))
        
        # LFP output
        self.lfp = bm.Variable(bm.zeros(1))
        
        # Store full params for reference
        self._params = p
    
    def sigmoid(self, v):
        """Sigmoid activation function (firing rate)."""
        return (2.0 * self.e0) / (1.0 + bm.exp(self.r * (self.v0 - v)))
    
    def update(self, inp=None):
        """
        Single-step update (called by DSRunner).
        
        Args:
            inp: External input p(t). If None, uses p_mean + noise.
        """
        dt = bm.dt  # dt is provided in seconds by the runner
        
        # External input
        if inp is None:
            p_t = self.p_mean + self.p_sigma * bm.random.randn(1)
        else:
            p_t = inp
        
        # Current state
        y0, y1, y2, y3, y4 = self.y0.value, self.y1.value, self.y2.value, self.y3.value, self.y4.value
        y5, y6, y7, y8, y9 = self.y5.value, self.y6.value, self.y7.value, self.y8.value, self.y9.value
        
        # Compute derivatives (10 ODEs)
        # PSP from pyramidal -> excitatory interneurons
        dy0 = y5
        dy5 = self.A * self.a * self.sigmoid(y1 - y2 - y3) - 2.0 * self.a * y5 - self.a**2 * y0
        
        # PSP from excitatory -> pyramidal (with external input)
        dy1 = y6
        dy6 = self.A * self.a * (p_t + self.C2 * self.sigmoid(self.C1 * y0)) - 2.0 * self.a * y6 - self.a**2 * y1
        
        # PSP from slow inhibitory -> pyramidal
        dy2 = y7
        dy7 = self.B * self.b * self.C4 * self.sigmoid(self.C3 * y0) - 2.0 * self.b * y7 - self.b**2 * y2
        
        # PSP from fast inhibitory -> pyramidal
        dy3 = y8
        dy8 = self.G * self.g * self.C7 * self.sigmoid(self.C5 * y0 - self.C6 * y4) - 2.0 * self.g * y8 - self.g**2 * y3
        
        # PSP from pyramidal -> slow inhibitory
        dy4 = y9
        dy9 = self.B * self.b * self.sigmoid(self.C3 * y0) - 2.0 * self.b * y9 - self.b**2 * y4
        
        # Euler integration
        self.y0.value = y0 + dt * dy0
        self.y1.value = y1 + dt * dy1
        self.y2.value = y2 + dt * dy2
        self.y3.value = y3 + dt * dy3
        self.y4.value = y4 + dt * dy4
        self.y5.value = y5 + dt * dy5
        self.y6.value = y6 + dt * dy6
        self.y7.value = y7 + dt * dy7
        self.y8.value = y8 + dt * dy8
        self.y9.value = y9 + dt * dy9
        
        # LFP proxy: sum of PSPs at pyramidal population
        self.lfp.value = self.y1.value - self.y2.value - self.y3.value
    
    def reset_state(self, batch_size=None):
        """Reset all state variables to zero."""
        self.y0.value = bm.zeros(1)
        self.y1.value = bm.zeros(1)
        self.y2.value = bm.zeros(1)
        self.y3.value = bm.zeros(1)
        self.y4.value = bm.zeros(1)
        self.y5.value = bm.zeros(1)
        self.y6.value = bm.zeros(1)
        self.y7.value = bm.zeros(1)
        self.y8.value = bm.zeros(1)
        self.y9.value = bm.zeros(1)
        self.lfp.value = bm.zeros(1)
    
    @property
    def params(self) -> Dict:
        """Return current parameter dict."""
        return dict(self._params)
