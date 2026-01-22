"""
Standard parameters and type-specific parameters for Wendling model.

Note on p_sigma (input noise standard deviation):
    Both p_sigma=2.0 and p_sigma=30.0 can produce all 6 activity types.
    - p_sigma=2.0:  Lower noise floor, cleaner spectral peaks
    - p_sigma=30.0: Higher noise floor, broader spectral features
"""

from typing import Dict, Any

# Standard physiological parameters
STANDARD_PARAMS: Dict[str, float] = {
    # Synaptic gains
    'A': 3.25,      # Excitatory gain (mV)
    'B': 22.0,      # Slow inhibitory gain (mV)
    'G': 10.0,      # Fast inhibitory gain (mV)
    
    # Time constants (1/s)
    'a': 100.0,     # Excitatory time constant
    'b': 50.0,      # Slow inhibitory time constant
    'g': 500.0,     # Fast inhibitory time constant
    
    # Sigmoid parameters
    'e0': 2.5,      # Half of max firing rate (1/s)
    'v0': 6.0,      # Potential at half max (mV)
    'r': 0.56,      # Sigmoid slope (1/mV)
    
    # Connectivity (dimensionless)
    'C': 135.0,
    'C1': 1.0,      # C1 = C
    'C2': 0.8,      # C2 = 0.8*C
    'C3': 0.25,     # C3 = 0.25*C
    'C4': 0.25,     # C4 = 0.25*C
    'C5': 0.3,      # C5 = 0.3*C
    'C6': 0.1,      # C6 = 0.1*C
    'C7': 0.8,      # C7 = 0.8*C
    
    # Input parameters
    'p_mean': 90.0,
    'p_sigma': 2.0,
}

# Six activity types from Wendling et al.
TYPE_PARAMS: Dict[str, Dict[str, Any]] = {
    "Type1": {
        "name": "Type 1: Background activity",
        "description": "Normal background activity, slow waves (1-7 Hz)",
        "params": {"A": 5.0, "B": 50.0, "G": 15.0, "p_mean": 90.0, "p_sigma": 2.0},
        "expected": {"freq_range": (1, 7), "pattern": "background"},
    },
    "Type2": {
        "name": "Type 2: Sporadic spikes",
        "description": "Sporadic spikes (1-5 Hz)",
        "params": {"A": 5.0, "B": 40.0, "G": 15.0, "p_mean": 90.0, "p_sigma": 2.0},
        "expected": {"freq_range": (1, 5), "pattern": "sporadic_spikes"},
    },
    "Type3": {
        "name": "Type 3: Sustained SWD",
        "description": "Sustained spike-wave discharge (3-6 Hz)",
        "params": {"A": 5.0, "B": 25.0, "G": 15.0, "p_mean": 90.0, "p_sigma": 2.0},
        "expected": {"freq_range": (3, 6), "pattern": "sustained_SWD"},
    },
    "Type4": {
        "name": "Type 4: Alpha-like rhythm",
        "description": "Alpha-like rhythm (8-13 Hz)",
        "params": {"A": 5.0, "B": 10.0, "G": 15.0, "p_mean": 90.0, "p_sigma": 2.0},
        "expected": {"freq_range": (8, 13), "pattern": "alpha_like"},
    },
    "Type5": {
        "name": "Type 5: LVFA",
        "description": "Low-voltage fast activity (10-20 Hz)",
        "params": {"A": 5.0, "B": 5.0, "G": 25.0, "p_mean": 90.0, "p_sigma": 2.0},
        "expected": {"freq_range": (10, 20), "pattern": "LVFA"},
    },
    "Type6": {
        "name": "Type 6: Quasi-sinusoidal",
        "description": "Quasi-sinusoidal activity (9-13 Hz)",
        "params": {"A": 5.0, "B": 15.0, "G": 0.0, "p_mean": 90.0, "p_sigma": 2.0},
        "expected": {"freq_range": (9, 13), "pattern": "quasi_sinusoidal"},
    },
}


def get_default_params() -> Dict[str, float]:
    """Return a copy of standard parameters."""
    return dict(STANDARD_PARAMS)


def get_type_params(type_name: str) -> Dict[str, float]:
    """Return parameter overrides for a specific activity type."""
    if type_name not in TYPE_PARAMS:
        raise KeyError(f"Unknown type: {type_name}. Available: {list(TYPE_PARAMS.keys())}")
    return dict(TYPE_PARAMS[type_name]["params"])


def merge_params(base: Dict, overrides: Dict) -> Dict:
    """Merge base parameters with overrides."""
    result = dict(base)
    result.update(overrides)
    return result
