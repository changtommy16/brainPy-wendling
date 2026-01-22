"""
Nevergrad optimization engine with ask/tell interface.
"""

from typing import Dict, Callable, Optional, List, Any
from dataclasses import dataclass, field
import time
import numpy as np

try:
    import nevergrad as ng
except ImportError:
    ng = None

from wendling_sim.optimize.search_space import SearchSpace


@dataclass
class OptResult:
    """
    Optimization result.
    
    Attributes:
        best_params: Optimal parameter dict
        best_loss: Final loss value
        history: List of (params, loss) tuples
        meta: Optimization metadata
    """
    best_params: Dict[str, float]
    best_loss: float
    history: List[tuple] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class NevergradOptimizer:
    """
    Nevergrad-based optimizer using ask/tell interface.
    
    Example:
        optimizer = NevergradOptimizer(search_space, objective_fn, budget=100)
        result = optimizer.run()
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        objective_fn: Callable[[Dict[str, float]], float],
        budget: int = 100,
        optimizer_name: str = 'NGOpt',
        num_workers: int = 1,
        seed: Optional[int] = None,
    ):
        """
        Initialize optimizer.
        
        Args:
            search_space: Parameter search space
            objective_fn: Function mapping params -> loss
            budget: Number of evaluations
            optimizer_name: Nevergrad optimizer name
            num_workers: Parallel workers (1 = sequential)
            seed: Random seed
        """
        if ng is None:
            raise ImportError("nevergrad not installed. Run: pip install nevergrad")
        
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.budget = budget
        self.optimizer_name = optimizer_name
        self.num_workers = num_workers
        self.seed = seed
        
        # Build nevergrad parametrization
        self.parametrization = search_space.to_nevergrad()
        
        # Create optimizer
        self.optimizer = ng.optimizers.registry[optimizer_name](
            parametrization=self.parametrization,
            budget=budget,
            num_workers=num_workers,
        )
        
        if seed is not None:
            self.optimizer.parametrization.random_state = np.random.RandomState(seed)
        
        # History
        self.history: List[tuple] = []
        self._details_history: List[Optional[Dict[str, Any]]] = []
    
    def run(self, verbose: bool = True) -> OptResult:
        """
        Run optimization loop.
        
        Args:
            verbose: Print progress
        
        Returns:
            OptResult with best params and history
        """
        start_time = time.time()
        
        for i in range(self.budget):
            # Ask for candidate
            candidate = self.optimizer.ask()
            params = dict(candidate.value)
            
            # Evaluate objective
            try:
                result = self.objective_fn(params)
                details: Optional[Dict[str, Any]] = None
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                    loss = float(result[0])
                    details = result[1]
                else:
                    loss = float(result)
            except Exception as e:
                if verbose:
                    print(f"[{i+1}/{self.budget}] Error: {e}")
                loss = float('inf')
                details = {"error": str(e)}
            
            # Tell optimizer
            self.optimizer.tell(candidate, loss)
            
            # Record history
            self.history.append((params.copy(), loss))
            self._details_history.append(details)
            
            if verbose and (i + 1) % max(1, self.budget // 10) == 0:
                best_so_far = min(h[1] for h in self.history)
                print(f"[{i+1}/{self.budget}] loss={loss:.6f}, best={best_so_far:.6f}")
        
        # Get recommendation
        recommendation = self.optimizer.provide_recommendation()
        best_params = dict(recommendation.value)
        best_loss = min(h[1] for h in self.history)
        
        elapsed = time.time() - start_time
        
        meta = {
            'optimizer': self.optimizer_name,
            'budget': self.budget,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'elapsed_s': elapsed,
            'loss_details': self._details_history,
        }
        
        return OptResult(
            best_params=best_params,
            best_loss=best_loss,
            history=self.history,
            meta=meta,
        )
    
    def step(self) -> tuple:
        """
        Single ask/tell step (for custom loops).
        
        Returns:
            (params, candidate) tuple
        """
        candidate = self.optimizer.ask()
        params = dict(candidate.value)
        return params, candidate
    
    def tell(self, candidate, loss: float):
        """Tell optimizer about evaluation result."""
        self.optimizer.tell(candidate, loss)
        self.history.append((dict(candidate.value), loss))
        self._details_history.append(None)
