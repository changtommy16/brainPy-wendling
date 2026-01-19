"""
Sanity Checks for Wendling Model

Provides validation functions to ensure model correctness:
- check_no_nan_inf: Verify outputs contain no NaN or Inf values
- check_reproducibility: Verify same seed produces identical outputs
- check_monitor_shapes: Verify monitor arrays have correct shapes
- run_all_checks: Run all sanity checks

Usage:
    python core_wendling/sanity_check.py
    
    # Or programmatically:
    from core_wendling.sanity_check import run_all_checks
    run_all_checks(config)
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_wendling.run_forward import load_config, run_forward


# =============================================================================
# Check Functions
# =============================================================================

def check_no_nan_inf(results: dict, verbose: bool = True) -> bool:
    """
    Check that no output arrays contain NaN or Inf values.
    
    Args:
        results: Dictionary of result arrays
        verbose: Whether to print status
    
    Returns:
        True if all values are finite, False otherwise
    """
    all_finite = True
    
    for key, arr in results.items():
        if isinstance(arr, np.ndarray):
            has_nan = np.any(np.isnan(arr))
            has_inf = np.any(np.isinf(arr))
            
            if has_nan or has_inf:
                all_finite = False
                if verbose:
                    print(f"  [FAIL] {key}: NaN={has_nan}, Inf={has_inf}")
            else:
                if verbose:
                    print(f"  [PASS] {key}: No NaN/Inf")
    
    return all_finite


def check_reproducibility(config: dict, verbose: bool = True) -> bool:
    """
    Check that same seed produces identical outputs.
    
    Args:
        config: Configuration dictionary
        verbose: Whether to print status
    
    Returns:
        True if outputs are reproducible, False otherwise
    """
    # Run twice with same config
    results1 = run_forward(config, verbose=False)
    results2 = run_forward(config, verbose=False)
    
    # Compare LFP
    lfp1 = results1['lfp']
    lfp2 = results2['lfp']
    
    max_diff = np.max(np.abs(lfp1 - lfp2))
    is_reproducible = max_diff < 1e-10
    
    if verbose:
        if is_reproducible:
            print(f"  [PASS] Reproducibility: max_diff = {max_diff:.2e}")
        else:
            print(f"  [FAIL] Reproducibility: max_diff = {max_diff:.2e}")
    
    return is_reproducible


def check_monitor_shapes(results: dict, config: dict, verbose: bool = True) -> bool:
    """
    Check that monitor arrays have correct shapes.
    
    Args:
        results: Dictionary of result arrays
        config: Configuration dictionary
        verbose: Whether to print status
    
    Returns:
        True if all shapes are correct, False otherwise
    """
    dt_ms = config.get('dt_ms', 0.1)
    duration_ms = config.get('duration_ms', 20000)
    expected_length = int(duration_ms / dt_ms)
    
    all_correct = True
    
    for key, arr in results.items():
        if isinstance(arr, np.ndarray) and arr.ndim == 1:
            actual_length = len(arr)
            is_correct = actual_length == expected_length
            
            if not is_correct:
                all_correct = False
            
            if verbose:
                status = "[PASS]" if is_correct else "[FAIL]"
                print(f"  {status} {key}: shape={arr.shape}, expected={expected_length}")
    
    return all_correct


def check_stimulus_response(results: dict, config: dict, verbose: bool = True) -> bool:
    """
    Check that stimulus produces visible response.
    
    Compares LFP amplitude before and during stimulus.
    
    Args:
        results: Dictionary of result arrays
        config: Configuration dictionary
        verbose: Whether to print status
    
    Returns:
        True if stimulus produces response, False otherwise
    """
    stim_config = config.get('stim', {})
    onset_ms = stim_config.get('onset_ms', 0)
    offset_ms = stim_config.get('offset_ms', float('inf'))
    dt_ms = config.get('dt_ms', 0.1)
    
    t = results['t']
    lfp = results['lfp']
    
    # Get baseline period (before stimulus)
    baseline_mask = t < onset_ms
    if np.sum(baseline_mask) < 100:
        baseline_mask = t < 500  # Use first 500ms if onset is early
    
    # Get stimulus period
    stim_mask = (t >= onset_ms) & (t < offset_ms)
    
    if np.sum(baseline_mask) < 10 or np.sum(stim_mask) < 10:
        if verbose:
            print("  [SKIP] Stimulus response: insufficient data")
        return True
    
    baseline_std = np.std(lfp[baseline_mask])
    stim_std = np.std(lfp[stim_mask])
    
    # Check if stimulus period has different variance
    ratio = stim_std / (baseline_std + 1e-10)
    has_response = ratio > 0.1  # Allow wide range
    
    if verbose:
        status = "[PASS]" if has_response else "[WARN]"
        print(f"  {status} Stimulus response: baseline_std={baseline_std:.4f}, stim_std={stim_std:.4f}, ratio={ratio:.2f}")
    
    return has_response


# =============================================================================
# Run All Checks
# =============================================================================

def run_all_checks(config: dict = None, verbose: bool = True) -> bool:
    """
    Run all sanity checks.
    
    Args:
        config: Configuration dictionary (loads default if None)
        verbose: Whether to print status
    
    Returns:
        True if all checks pass, False otherwise
    """
    # Load default config if not provided
    if config is None:
        config_path = Path(__file__).parent / 'config.yaml'
        config = load_config(str(config_path))
    
    # Use shorter duration for quick checks
    check_config = config.copy()
    check_config['duration_ms'] = min(config.get('duration_ms', 20000), 5000)  # Max 5s for checks
    
    if verbose:
        print("=" * 60)
        print("WENDLING MODEL SANITY CHECKS")
        print("=" * 60)
        print(f"\nRunning with duration={check_config['duration_ms']}ms, dt={check_config['dt_ms']}ms\n")
    
    all_pass = True
    
    # Run simulation
    if verbose:
        print("[1] Running simulation...")
    results = run_forward(check_config, verbose=False)
    
    # Check 1: No NaN/Inf
    if verbose:
        print("\n[2] Checking for NaN/Inf values...")
    if not check_no_nan_inf(results, verbose):
        all_pass = False
    
    # Check 2: Reproducibility
    if verbose:
        print("\n[3] Checking reproducibility...")
    if not check_reproducibility(check_config, verbose):
        all_pass = False
    
    # Check 3: Monitor shapes
    if verbose:
        print("\n[4] Checking monitor shapes...")
    if not check_monitor_shapes(results, check_config, verbose):
        all_pass = False
    
    # Check 4: Stimulus response
    if verbose:
        print("\n[5] Checking stimulus response...")
    check_stimulus_response(results, check_config, verbose)  # Warning only, don't fail
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        if all_pass:
            print("ALL CHECKS PASSED ✓")
        else:
            print("SOME CHECKS FAILED ✗")
        print("=" * 60)
    
    return all_pass


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Wendling model sanity checks')
    parser.add_argument('--config', '-c', type=str, default=None, help='Path to config file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    config = None
    if args.config:
        config = load_config(args.config)
    
    success = run_all_checks(config, verbose=not args.quiet)
    sys.exit(0 if success else 1)
