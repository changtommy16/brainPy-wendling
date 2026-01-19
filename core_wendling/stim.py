"""
Stimulus Generators for Wendling Model

This module provides input signal generators for the Wendling neural mass model:
- BaselineStimulus: Constant drive with optional onset/offset window
- SineStimulus: Periodic sinusoidal drive (for ASSR experiments)
- create_stimulus: Factory function to create stimulus from config

============================================================================
INJECTION RULE:
    External input u(t) is added to the pyramidal cell equation (dy6):
    dy6 = A*a*(C2*S(C1*y0) + p_mean + u(t)) - 2*a*y6 - a²*y1
    
    The stimulus is additive and in units of Hz (firing rate).
============================================================================
"""

import numpy as np
import brainpy.math as bm
from typing import Dict, Optional, Union, Callable


# =============================================================================
# Baseline Stimulus
# =============================================================================

class BaselineStimulus:
    """
    Constant baseline input with optional onset/offset window.
    
    Args:
        amp: Stimulus amplitude (Hz)
        onset_ms: Stimulus onset time (ms)
        offset_ms: Stimulus offset time (ms)
        dt_ms: Time step (ms)
    
    Usage:
        stim = BaselineStimulus(amp=10.0, onset_ms=500, offset_ms=5000)
        u_t = stim(t_ms)  # Get stimulus value at time t_ms
    """
    
    def __init__(
        self,
        amp: float = 0.0,
        onset_ms: float = 0.0,
        offset_ms: float = float('inf'),
        dt_ms: float = 0.1
    ):
        self.amp = amp
        self.onset_ms = onset_ms
        self.offset_ms = offset_ms
        self.dt_ms = dt_ms
    
    def __call__(self, t_ms: float) -> float:
        """
        Get stimulus value at time t.
        
        Args:
            t_ms: Current time in milliseconds
        
        Returns:
            Stimulus value (Hz)
        """
        if self.onset_ms <= t_ms < self.offset_ms:
            return self.amp
        return 0.0
    
    def get_array(self, duration_ms: float) -> np.ndarray:
        """
        Generate full stimulus array for given duration.
        
        Args:
            duration_ms: Total duration in milliseconds
        
        Returns:
            Array of stimulus values
        """
        n_steps = int(duration_ms / self.dt_ms)
        t = np.arange(n_steps) * self.dt_ms
        stim = np.zeros(n_steps)
        
        mask = (t >= self.onset_ms) & (t < self.offset_ms)
        stim[mask] = self.amp
        
        return stim


# =============================================================================
# Sinusoidal Stimulus (for ASSR)
# =============================================================================

class SineStimulus:
    """
    Periodic sinusoidal stimulus for ASSR experiments.
    
    u(t) = amp * sin(2π * f_hz * t + phase)
    
    Supports common ASSR frequencies: 20, 30, 40 Hz.
    
    Args:
        f_hz: Stimulus frequency (Hz), e.g., 40 for 40 Hz ASSR
        amp: Stimulus amplitude (Hz)
        phase: Phase offset (radians), default=0
        onset_ms: Stimulus onset time (ms)
        offset_ms: Stimulus offset time (ms)
        dt_ms: Time step (ms)
    
    Usage:
        stim = SineStimulus(f_hz=40, amp=1.0, onset_ms=500, offset_ms=19500)
        u_t = stim(t_ms)
    """
    
    def __init__(
        self,
        f_hz: float = 40.0,
        amp: float = 1.0,
        phase: float = 0.0,
        onset_ms: float = 0.0,
        offset_ms: float = float('inf'),
        dt_ms: float = 0.1
    ):
        self.f_hz = f_hz
        self.amp = amp
        self.phase = phase
        self.onset_ms = onset_ms
        self.offset_ms = offset_ms
        self.dt_ms = dt_ms
        
        # Precompute angular frequency (rad/ms)
        # f_hz is in cycles/second, we need rad/ms
        # ω = 2π * f_hz / 1000 (converting Hz to rad/ms)
        self.omega_per_ms = 2.0 * np.pi * f_hz / 1000.0
    
    def __call__(self, t_ms: float) -> float:
        """
        Get stimulus value at time t.
        
        Args:
            t_ms: Current time in milliseconds
        
        Returns:
            Stimulus value (Hz)
        """
        if self.onset_ms <= t_ms < self.offset_ms:
            return self.amp * np.sin(self.omega_per_ms * t_ms + self.phase)
        return 0.0
    
    def get_array(self, duration_ms: float) -> np.ndarray:
        """
        Generate full stimulus array for given duration.
        
        Args:
            duration_ms: Total duration in milliseconds
        
        Returns:
            Array of stimulus values
        """
        n_steps = int(duration_ms / self.dt_ms)
        t = np.arange(n_steps) * self.dt_ms
        stim = np.zeros(n_steps)
        
        mask = (t >= self.onset_ms) & (t < self.offset_ms)
        stim[mask] = self.amp * np.sin(self.omega_per_ms * t[mask] + self.phase)
        
        return stim


# =============================================================================
# Factory Function
# =============================================================================

def create_stimulus(stim_config: Dict, dt_ms: float = 0.1) -> Union[BaselineStimulus, SineStimulus]:
    """
    Create stimulus object from configuration dictionary.
    
    Args:
        stim_config: Stimulus configuration with keys:
            - kind: "baseline" or "sine"
            - amp: Amplitude (Hz)
            - onset_ms: Onset time (ms)
            - offset_ms: Offset time (ms)
            - f_hz: Frequency (Hz), only for "sine"
            - phase: Phase (radians), only for "sine"
        dt_ms: Time step (ms)
    
    Returns:
        Stimulus object (BaselineStimulus or SineStimulus)
    
    Example:
        config = {'kind': 'sine', 'f_hz': 40, 'amp': 1.0, 'onset_ms': 500, 'offset_ms': 19500}
        stim = create_stimulus(config, dt_ms=0.1)
    """
    kind = stim_config.get('kind', 'baseline')
    amp = stim_config.get('amp', 0.0)
    onset_ms = stim_config.get('onset_ms', 0.0)
    offset_ms = stim_config.get('offset_ms', float('inf'))
    
    if kind == 'baseline':
        return BaselineStimulus(
            amp=amp,
            onset_ms=onset_ms,
            offset_ms=offset_ms,
            dt_ms=dt_ms
        )
    
    elif kind == 'sine':
        f_hz = stim_config.get('f_hz', 40.0)
        phase = stim_config.get('phase', 0.0)
        return SineStimulus(
            f_hz=f_hz,
            amp=amp,
            phase=phase,
            onset_ms=onset_ms,
            offset_ms=offset_ms,
            dt_ms=dt_ms
        )
    
    else:
        raise ValueError(f"Unknown stimulus kind: {kind}. Use 'baseline' or 'sine'.")


# =============================================================================
# Input Function for DSRunner
# =============================================================================

def make_input_func(
    stim: Union[BaselineStimulus, SineStimulus],
    dt_ms: float = 0.1
) -> Callable:
    """
    Create an input function compatible with BrainPy DSRunner.
    
    Args:
        stim: Stimulus object
        dt_ms: Time step (ms)
    
    Returns:
        Function that takes time index and returns stimulus value
    
    Usage:
        stim = SineStimulus(f_hz=40, amp=1.0)
        input_func = make_input_func(stim, dt_ms=0.1)
        runner = bp.DSRunner(model, fun_inputs=input_func)
    """
    def input_func(tdi):
        # tdi is TimeDerivativeIndex containing current time info
        # tdi.t is current time in the runner's time unit
        t_ms = tdi.t  # Assuming runner uses ms
        return stim(t_ms)
    
    return input_func


# =============================================================================
# Utility: Generate Stimulus Array for Pre-computed Input
# =============================================================================

def generate_stimulus_array(
    stim_config: Dict,
    duration_ms: float,
    dt_ms: float = 0.1
) -> np.ndarray:
    """
    Generate full stimulus array from config.
    
    This is useful for pre-computing the entire stimulus before running
    the simulation, which can be more efficient.
    
    Args:
        stim_config: Stimulus configuration dictionary
        duration_ms: Total simulation duration (ms)
        dt_ms: Time step (ms)
    
    Returns:
        Stimulus array of shape (n_steps,)
    """
    stim = create_stimulus(stim_config, dt_ms=dt_ms)
    return stim.get_array(duration_ms)
