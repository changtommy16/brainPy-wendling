"""
Wendling parameter sets (six activity types) for single-node validation.

Source: neurolib_wendling/STANDARD_PARAMETERS.py
These sets are intended for comparing model output against the paper.
"""

from typing import Dict, List


WENDLING_TYPE_PARAMS: Dict[str, Dict] = {
    "Type1": {
        "name": "Type 1: Background activity",
        "description": "Normal background activity, slow waves (1-7 Hz)",
        "params": {
            "A": 5.0,
            "B": 50.0,
            "G": 15.0,
            "p_mean": 90.0,
            "p_sigma": 30.0,
        },
        "expected": {"freq_range": (1, 7), "pattern": "background"},
    },
    "Type2": {
        "name": "Type 2: Sporadic spikes",
        "description": "Sporadic spikes (1-5 Hz)",
        "params": {
            "A": 5.0,
            "B": 40.0,
            "G": 15.0,
            "p_mean": 90.0,
            "p_sigma": 30.0,
        },
        "expected": {"freq_range": (1, 5), "pattern": "sporadic_spikes"},
    },
    "Type3": {
        "name": "Type 3: Sustained SWD",
        "description": "Sustained spike-wave discharge (3-6 Hz)",
        "params": {
            "A": 5.0,
            "B": 25.0,
            "G": 15.0,
            "p_mean": 90.0,
            "p_sigma": 30.0,
        },
        "expected": {"freq_range": (3, 6), "pattern": "sustained_SWD"},
    },
    "Type4": {
        "name": "Type 4: Alpha-like rhythm",
        "description": "Alpha-like rhythm (8-13 Hz)",
        "params": {
            "A": 5.0,
            "B": 10.0,
            "G": 15.0,
            "p_mean": 90.0,
            "p_sigma": 30.0,
        },
        "expected": {"freq_range": (8, 13), "pattern": "alpha_like"},
    },
    "Type5": {
        "name": "Type 5: LVFA",
        "description": "Low-voltage fast activity (10-20 Hz)",
        "params": {
            "A": 5.0,
            "B": 5.0,
            "G": 25.0,
            "p_mean": 90.0,
            "p_sigma": 30.0,
        },
        "expected": {"freq_range": (10, 20), "pattern": "LVFA"},
    },
    "Type6": {
        "name": "Type 6: Quasi-sinusoidal",
        "description": "Quasi-sinusoidal activity (9-13 Hz)",
        "params": {
            "A": 5.0,
            "B": 15.0,
            "G": 0.0,
            "p_mean": 90.0,
            "p_sigma": 30.0,
        },
        "expected": {"freq_range": (9, 13), "pattern": "quasi_sinusoidal"},
    },
}


def list_type_params() -> List[str]:
    """Return available type names in a stable order."""
    return list(WENDLING_TYPE_PARAMS.keys())


def get_type_params(type_name: str) -> Dict:
    """Return a shallow copy of the parameter dict for a given type."""
    if type_name not in WENDLING_TYPE_PARAMS:
        raise KeyError(f"Unknown type: {type_name}")
    return dict(WENDLING_TYPE_PARAMS[type_name]["params"])
