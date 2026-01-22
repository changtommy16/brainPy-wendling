"""
Logging utilities for optimization runs.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union


class EvalLogger:
    """
    Logger for optimization evaluations.
    
    Writes CSV rows with:
        eval_id, timestamp, loss, A, B, G, a, b, g, sim_time_s, seed, etc.
    """
    
    def __init__(self, log_path: Union[str, Path], param_names: list = None):
        """
        Initialize logger.
        
        Args:
            log_path: Path to CSV log file
            param_names: Parameter names to log
        """
        self.log_path = Path(log_path)
        self.param_names = param_names or ['A', 'B', 'G', 'a', 'b', 'g']
        self.eval_id = 0
        
        # Create header
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_header()
    
    def _write_header(self):
        """Write CSV header."""
        header = ['eval_id', 'timestamp', 'loss'] + self.param_names + ['sim_time_s', 'seed']
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    def log(
        self,
        loss: float,
        params: Dict[str, float],
        sim_time_s: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """
        Log one evaluation.
        
        Args:
            loss: Loss value
            params: Parameter dict
            sim_time_s: Simulation time in seconds
            seed: Random seed used
        """
        self.eval_id += 1
        timestamp = datetime.now().isoformat()
        
        row = [self.eval_id, timestamp, loss]
        row.extend(params.get(name, '') for name in self.param_names)
        row.extend([sim_time_s or '', seed or ''])
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def save_summary(self, summary: Dict[str, Any], path: Union[str, Path] = None):
        """Save optimization summary to JSON."""
        path = path or self.log_path.with_suffix('.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
