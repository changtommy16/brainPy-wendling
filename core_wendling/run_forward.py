"""
Wendling Forward Simulation Runner

Entry point for running forward simulation of the Wendling neural mass model.

Usage:
    python core_wendling/run_forward.py
    python core_wendling/run_forward.py --config custom_config.yaml
    
This script:
1. Loads configuration from YAML file
2. Builds the Wendling model
3. Generates stimulus signal
4. Runs forward simulation with DSRunner
5. Saves outputs to file
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import yaml
import brainpy as bp
import brainpy.math as bm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_wendling.wendling_bp import WendlingSingleNode, get_default_params
from core_wendling.stim import create_stimulus, generate_stimulus_array


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_params(config: Dict) -> Dict:
    """
    Merge config params with defaults to create complete parameter dict.
    
    Args:
        config: Configuration dictionary from YAML
    
    Returns:
        Complete parameter dictionary
    """
    params = get_default_params()
    
    # Override with config values
    if 'params' in config:
        params.update(config['params'])
    
    # Add runtime params
    params['dt_ms'] = config.get('dt_ms', 0.1)
    params['seed'] = config.get('seed', 0)
    
    # Compute derived connectivity constants if C is provided
    if 'C' in params and 'C1' not in config.get('params', {}):
        C = params['C']
        params['C1'] = 1.0 * C
        params['C2'] = 0.8 * C
        params['C3'] = 0.25 * C
        params['C4'] = 0.25 * C
        params['C5'] = 0.3 * C
        params['C6'] = 0.1 * C
        params['C7'] = 0.8 * C
    
    return params


# =============================================================================
# Forward Simulation
# =============================================================================

def run_forward(config: Dict, verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Run forward simulation based on configuration.
    
    Args:
        config: Configuration dictionary
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing:
            - 't': Time array (ms)
            - 'lfp': LFP signal
            - 'stim': Stimulus signal
            - Other monitored variables
    """
    # -------------------------------------------------------------------------
    # Extract config
    # -------------------------------------------------------------------------
    seed = config.get('seed', 0)
    dt_ms = config.get('dt_ms', 0.1)
    duration_ms = config.get('duration_ms', 20000)
    stim_config = config.get('stim', {'kind': 'baseline', 'amp': 0.0})
    monitors_list = config.get('monitors', ['lfp'])
    
    # Set random seed for reproducibility
    bm.random.seed(seed)
    np.random.seed(seed)
    
    if verbose:
        print(f"[run_forward] Configuration:")
        print(f"  - seed: {seed}")
        print(f"  - dt_ms: {dt_ms}")
        print(f"  - duration_ms: {duration_ms}")
        print(f"  - stim: {stim_config['kind']} @ {stim_config.get('f_hz', 'N/A')} Hz")
    
    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    params = merge_params(config)
    model = WendlingSingleNode(params)
    model.reset_state()
    
    if verbose:
        print(f"[run_forward] Model built with params:")
        print(f"  - A={params['A']}, B={params['B']}, G={params['G']}")
        print(f"  - a={params['a']}, b={params['b']}, g={params['g']}")
    
    # -------------------------------------------------------------------------
    # Generate stimulus
    # -------------------------------------------------------------------------
    stim_array = generate_stimulus_array(stim_config, duration_ms, dt_ms)
    n_steps = len(stim_array)
    
    if verbose:
        print(f"[run_forward] Stimulus generated: {n_steps} steps")
    
    # -------------------------------------------------------------------------
    # Run simulation manually (step-by-step for input injection)
    # -------------------------------------------------------------------------
    # Pre-allocate output arrays
    lfp_out = np.zeros(n_steps)
    y0_out = np.zeros(n_steps)
    y1_out = np.zeros(n_steps)
    y2_out = np.zeros(n_steps)
    y3_out = np.zeros(n_steps)
    y4_out = np.zeros(n_steps)
    
    if verbose:
        print(f"[run_forward] Running simulation...")
    
    # Step-by-step integration
    for i in range(n_steps):
        # Set external input
        model.u.value = bm.asarray([stim_array[i]])
        
        # Update model
        model.update()
        
        # Record outputs
        lfp_out[i] = float(model._lfp.value[0])
        y0_out[i] = float(model.y0.value[0])
        y1_out[i] = float(model.y1.value[0])
        y2_out[i] = float(model.y2.value[0])
        y3_out[i] = float(model.y3.value[0])
        y4_out[i] = float(model.y4.value[0])
    
    # -------------------------------------------------------------------------
    # Prepare output
    # -------------------------------------------------------------------------
    t = np.arange(n_steps) * dt_ms
    
    results = {
        't': t,
        'lfp': lfp_out,
        'stim': stim_array,
        'y0': y0_out,
        'y1': y1_out,
        'y2': y2_out,
        'y3': y3_out,
        'y4': y4_out,
    }
    
    if verbose:
        print(f"[run_forward] Simulation complete!")
        print(f"  - LFP range: [{lfp_out.min():.2f}, {lfp_out.max():.2f}]")
        print(f"  - LFP length: {len(lfp_out)} samples")
    
    return results


# =============================================================================
# Save Results
# =============================================================================

def save_results(results: Dict[str, np.ndarray], config: Dict, verbose: bool = True):
    """
    Save simulation results to file.
    
    Args:
        results: Dictionary of result arrays
        config: Configuration dictionary
        verbose: Whether to print progress
    """
    output_config = config.get('output', {})
    save_dir = output_config.get('save_dir', './results')
    filename = output_config.get('filename', 'wendling_forward')
    fmt = output_config.get('format', 'npz')
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    if fmt == 'npz':
        filepath = os.path.join(save_dir, f"{filename}.npz")
        np.savez(filepath, **results)
    elif fmt == 'csv':
        import pandas as pd
        filepath = os.path.join(save_dir, f"{filename}.csv")
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unknown output format: {fmt}")
    
    if verbose:
        print(f"[run_forward] Results saved to: {filepath}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for forward simulation."""
    parser = argparse.ArgumentParser(description='Run Wendling forward simulation')
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output messages'
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config is None:
        # Use default config in same directory
        config_path = Path(__file__).parent / 'config.yaml'
    else:
        config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    verbose = not args.quiet
    
    # Run simulation
    results = run_forward(config, verbose=verbose)
    
    # Save results
    if not args.no_save:
        save_results(results, config, verbose=verbose)
    
    return results


if __name__ == '__main__':
    main()
