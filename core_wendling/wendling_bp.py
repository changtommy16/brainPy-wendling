"""
BrainPy Wendling Single-Node Neural Mass Model

Ported from neurolib Wendling implementation. This module defines:
- WendlingSingleNode: BrainPy DynamicalSystem for the 10-ODE Wendling-Chauvel model
- ODE derivatives and sigmoid function
- LFP proxy computation

============================================================================
MAPPING BLOCK (neurolib -> BrainPy)
============================================================================

STATE VECTOR (10 ODEs):
    y0: Pyramidal → Excitatory interneurons (membrane potential, mV)
    y1: Excitatory interneurons → Pyramidal (EPSP, mV)
    y2: Slow inhibitory → Pyramidal (slow IPSP, mV)
    y3: Fast inhibitory → Pyramidal (fast IPSP, mV)
    y4: Pyramidal → Fast inhibitory (membrane potential, mV)
    y5: dy0/dt (derivative of y0)
    y6: dy1/dt (derivative of y1)
    y7: dy2/dt (derivative of y2)
    y8: dy3/dt (derivative of y3)
    y9: dy4/dt (derivative of y4)

PARAMETER DICT:
    A: Excitatory gain (mV), default=5.0
    a: Excitatory time constant (1/s), default=100.0
    B: Slow inhibitory gain (mV), default=22.0
    b: Slow inhibitory time constant (1/s), default=50.0
    G: Fast inhibitory gain (mV), default=18.0
    g: Fast inhibitory time constant (1/s), default=500.0
    C: Base connectivity constant, default=135.0
    C1-C7: Derived connectivity constants
    e0: Half max firing rate (Hz), default=2.5
    v0: Firing threshold (mV), default=6.0
    r: Sigmoid slope (1/mV), default=0.56

TIME UNITS:
    - Config uses milliseconds (dt_ms, duration_ms)
    - ODEs internally use SECONDS (a, b, g are in 1/s)
    - Conversion: dt_s = dt_ms / 1000.0

INPUT INJECTION SITE:
    - External input u(t) is added to dy6 equation (pyramidal cells)
    - Equation: dy6 = A*a*(C2*S(C1*y0) + p_mean + u(t)) - 2*a*y6 - a²*y1

LFP PROXY (Output Observable):
    - lfp(t) = y1 - y2 - y3
    - This is the pyramidal membrane potential (EPSP - slow_IPSP - fast_IPSP)
    - Sign convention: positive = net excitation

============================================================================
References:
    - Wendling et al. (2002) European Journal of Neuroscience 15(9):1499-1508
    - Köksal Ersöz et al. (2020) PLoS Computational Biology 16(11):e1008430
============================================================================
"""

import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp
from typing import Dict, Optional, Callable


# =============================================================================
# Sigmoid Function
# =============================================================================

def sigmoid(v: bm.Array, e0: float = 2.5, v0: float = 6.0, r: float = 0.56) -> bm.Array:
    """
    Wendling sigmoid function: converts membrane potential to firing rate.
    
    S(v) = 2 * e0 / (1 + exp(r * (v0 - v)))
    
    Args:
        v: Membrane potential (mV)
        e0: Half of maximum firing rate (Hz), default=2.5
        v0: Firing threshold (mV), default=6.0
        r: Sigmoid slope (1/mV), default=0.56
    
    Returns:
        Firing rate (Hz)
    """
    return 2.0 * e0 / (1.0 + bm.exp(r * (v0 - v)))


# =============================================================================
# Wendling Single Node Model (BrainPy DynamicalSystem)
# =============================================================================

class WendlingSingleNode(bp.DynamicalSystem):
    """
    Single-node Wendling-Chauvel neural mass model implemented as BrainPy DynamicalSystem.
    
    This model represents a cortical column with:
    - Pyramidal cells
    - Excitatory interneurons  
    - Slow inhibitory interneurons (GABA_A, somatic)
    - Fast inhibitory interneurons (GABA_A, dendritic)
    
    The model uses 10 ODEs (5 membrane potentials + 5 derivatives).
    
    Usage:
        model = WendlingSingleNode(params)
        runner = bp.DSRunner(model, monitors=['lfp', 'y0', 'y1'], inputs=['u'])
        runner.run(duration_ms)
    """
    
    def __init__(
        self,
        params: Dict,
        name: Optional[str] = None
    ):
        """
        Initialize Wendling single-node model.
        
        Args:
            params: Dictionary containing model parameters:
                - A, a, B, b, G, g: Gain and time constants
                - C, C1-C7: Connectivity constants
                - e0, v0, r: Sigmoid parameters
                - p_mean, p_sigma: Input mean and noise std
                - dt_ms: Time step in milliseconds
            name: Optional name for the model
        """
        super().__init__(name=name)
        
        # ---------------------------------------------------------------------
        # Store parameters
        # ---------------------------------------------------------------------
        # Gains (mV)
        self.A = params.get('A', 5.0)
        self.B = params.get('B', 22.0)
        self.G = params.get('G', 18.0)
        
        # Time constants (convert to 1/s if given in 1/ms)
        # neurolib stores a/b/g as 1/ms values that get converted
        a_raw = params.get('a', 100.0)
        b_raw = params.get('b', 50.0)
        g_raw = params.get('g', 500.0)
        
        # If values > 1, assume they are in 1/s already (Wendling paper convention)
        # If values < 1, they were stored as 1/ms and need conversion
        self.a = a_raw if a_raw > 1.0 else a_raw * 1000.0  # 1/s
        self.b = b_raw if b_raw > 1.0 else b_raw * 1000.0  # 1/s
        self.g = g_raw if g_raw > 1.0 else g_raw * 1000.0  # 1/s
        
        # Connectivity constants
        C = params.get('C', 135.0)
        self.C1 = params.get('C1', 1.0 * C)
        self.C2 = params.get('C2', 0.8 * C)
        self.C3 = params.get('C3', 0.25 * C)
        self.C4 = params.get('C4', 0.25 * C)
        self.C5 = params.get('C5', 0.3 * C)
        self.C6 = params.get('C6', 0.1 * C)
        self.C7 = params.get('C7', 0.8 * C)
        
        # Sigmoid parameters
        self.e0 = params.get('e0', 2.5)
        self.v0 = params.get('v0', 6.0)
        self.r = params.get('r', 0.56)
        
        # Input parameters
        self.p_mean = params.get('p_mean', 90.0)
        self.p_sigma = params.get('p_sigma', 2.0)
        
        # Time step (ms -> s for internal computation)
        self.dt_ms = params.get('dt_ms', 0.1)
        self.dt_s = self.dt_ms / 1000.0
        
        # Random seed
        self.seed = params.get('seed', 0)
        
        # ---------------------------------------------------------------------
        # State variables (BrainPy Variables)
        # ---------------------------------------------------------------------
        # Membrane potentials (mV)
        self.y0 = bm.Variable(bm.zeros(1))  # Pyr -> Exc
        self.y1 = bm.Variable(bm.zeros(1))  # Exc -> Pyr (EPSP)
        self.y2 = bm.Variable(bm.zeros(1))  # SlowInh -> Pyr (slow IPSP)
        self.y3 = bm.Variable(bm.zeros(1))  # FastInh -> Pyr (fast IPSP)
        self.y4 = bm.Variable(bm.zeros(1))  # Pyr -> FastInh
        
        # Derivatives
        self.y5 = bm.Variable(bm.zeros(1))  # dy0/dt
        self.y6 = bm.Variable(bm.zeros(1))  # dy1/dt
        self.y7 = bm.Variable(bm.zeros(1))  # dy2/dt
        self.y8 = bm.Variable(bm.zeros(1))  # dy3/dt
        self.y9 = bm.Variable(bm.zeros(1))  # dy4/dt
        
        # External input (set by runner or stim module)
        self.u = bm.Variable(bm.zeros(1))
        
        # LFP output (computed each step)
        self._lfp = bm.Variable(bm.zeros(1))
        
        # Random key for noise
        self.rng = bm.random.RandomState(self.seed)
    
    def reset_state(self, batch_size: Optional[int] = None):
        """
        Reset all state variables to zero.
        
        Args:
            batch_size: Not used for single-node model
        """
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
        self.u.value = bm.zeros(1)
        self._lfp.value = bm.zeros(1)
        
        # Reset RNG with stored seed
        self.rng = bm.random.RandomState(self.seed)
    
    def lfp(self) -> bm.Array:
        """
        Compute LFP proxy: y1 - y2 - y3
        
        This represents the net membrane potential at pyramidal cells:
        EPSP - slow_IPSP - fast_IPSP
        
        Returns:
            LFP value (mV)
        """
        return self.y1.value - self.y2.value - self.y3.value
    
    def get_state_dict(self) -> Dict[str, bm.Array]:
        """
        Get current state as dictionary.
        
        Returns:
            Dictionary of state variable names to values
        """
        return {
            'y0': self.y0.value,
            'y1': self.y1.value,
            'y2': self.y2.value,
            'y3': self.y3.value,
            'y4': self.y4.value,
            'y5': self.y5.value,
            'y6': self.y6.value,
            'y7': self.y7.value,
            'y8': self.y8.value,
            'y9': self.y9.value,
            'lfp': self.lfp(),
        }
    
    def update(self, x=None):
        """
        Single time step update using Euler integration.
        
        This method is called by DSRunner at each time step.
        Implements the 10-ODE Wendling system with Euler-Maruyama integration.
        
        Args:
            x: Optional external input (overrides self.u if provided)
        """
        # Get external input
        u_ext = x if x is not None else self.u.value
        
        # Generate noise
        xi = self.rng.normal(0.0, 1.0, (1,))
        p_t = self.p_mean + self.p_sigma * xi
        
        # Current state values
        y0_val = self.y0.value
        y1_val = self.y1.value
        y2_val = self.y2.value
        y3_val = self.y3.value
        y4_val = self.y4.value
        y5_val = self.y5.value
        y6_val = self.y6.value
        y7_val = self.y7.value
        y8_val = self.y8.value
        y9_val = self.y9.value
        
        # Sigmoid function shorthand
        S = lambda v: sigmoid(v, self.e0, self.v0, self.r)
        
        # Pyramidal membrane potential (for output)
        v_pyr = y1_val - y2_val - y3_val
        
        # ---------------------------------------------------------------------
        # Compute derivatives (10-ODE Wendling system)
        # ---------------------------------------------------------------------
        # Equation 1-2: Pyramidal -> Excitatory
        dy0 = y5_val
        dy5 = self.A * self.a * S(v_pyr) - 2.0 * self.a * y5_val - self.a**2 * y0_val
        
        # Equation 3-4: Excitatory -> Pyramidal (with external input)
        dy1 = y6_val
        dy6 = self.A * self.a * (self.C2 * S(self.C1 * y0_val) + p_t + u_ext) \
              - 2.0 * self.a * y6_val - self.a**2 * y1_val
        
        # Equation 5-6: Slow inhibitory -> Pyramidal
        dy2 = y7_val
        dy7 = self.B * self.b * (self.C4 * S(self.C3 * y0_val)) \
              - 2.0 * self.b * y7_val - self.b**2 * y2_val
        
        # Equation 7-8: Fast inhibitory -> Pyramidal
        dy3 = y8_val
        dy8 = self.G * self.g * (self.C7 * S(self.C5 * y0_val - self.C6 * y4_val)) \
              - 2.0 * self.g * y8_val - self.g**2 * y3_val
        
        # Equation 9-10: Pyramidal -> Fast inhibitory
        dy4 = y9_val
        dy9 = self.B * self.b * S(self.C3 * y0_val) \
              - 2.0 * self.b * y9_val - self.b**2 * y4_val
        
        # ---------------------------------------------------------------------
        # Euler update (dt in seconds)
        # ---------------------------------------------------------------------
        dt = self.dt_s
        
        self.y0.value = y0_val + dt * dy0
        self.y1.value = y1_val + dt * dy1
        self.y2.value = y2_val + dt * dy2
        self.y3.value = y3_val + dt * dy3
        self.y4.value = y4_val + dt * dy4
        self.y5.value = y5_val + dt * dy5
        self.y6.value = y6_val + dt * dy6
        self.y7.value = y7_val + dt * dy7
        self.y8.value = y8_val + dt * dy8
        self.y9.value = y9_val + dt * dy9
        
        # Update LFP
        self._lfp.value = self.lfp()
        
        return self._lfp.value


# =============================================================================
# Default Parameters
# =============================================================================

def get_default_params() -> Dict:
    """
    Get default Wendling model parameters.
    
    Returns:
        Dictionary of default parameters
    """
    return {
        # Gains (mV)
        'A': 5.0,
        'B': 22.0,
        'G': 18.0,
        
        # Time constants (1/s)
        'a': 100.0,
        'b': 50.0,
        'g': 500.0,
        
        # Connectivity
        'C': 135.0,
        'C1': 135.0,
        'C2': 108.0,
        'C3': 33.75,
        'C4': 33.75,
        'C5': 40.5,
        'C6': 13.5,
        'C7': 108.0,
        
        # Sigmoid
        'e0': 2.5,
        'v0': 6.0,
        'r': 0.56,
        
        # Input
        'p_mean': 90.0,
        'p_sigma': 2.0,
        
        # Time
        'dt_ms': 0.1,
        'seed': 0,
    }
