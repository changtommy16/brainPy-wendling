"""
Connectivity generators for Wendling networks.

Supported generators:
    - erdos_renyi
    - small_world (Watts-Strogatz style)
    - ring_lattice
    - stochastic_block_model
"""

from typing import Optional, Sequence
import numpy as np


def _sample_weights(dist: str, shape, rng: np.random.Generator, scale: float = 1.0):
    """Sample edge weights based on distribution name."""
    dist = dist.lower()
    if dist == 'lognormal':
        return rng.lognormal(mean=0.0, sigma=1.0, size=shape) * scale
    if dist == 'normal':
        # Ensure positivity for coupling strengths
        return np.abs(rng.normal(loc=scale, scale=scale * 0.5, size=shape))
    if dist == 'uniform':
        return rng.uniform(low=0.0, high=scale, size=shape)
    raise ValueError(f"Unknown weight distribution: {dist}")


def erdos_renyi(
    n_nodes: int,
    p: float = 0.1,
    weight_dist: str = 'lognormal',
    weight_scale: float = 1.0,
    seed: Optional[int] = None,
    symmetric: bool = False,
) -> np.ndarray:
    """Erdos-Renyi random graph."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n_nodes, n_nodes)) < p
    np.fill_diagonal(mask, False)
    weights = _sample_weights(weight_dist, (n_nodes, n_nodes), rng, scale=weight_scale)
    W = mask.astype(np.float32) * weights.astype(np.float32)
    if symmetric:
        upper = np.triu(W, k=1)
        W = upper + upper.T
    return W.astype(np.float32)


def ring_lattice(
    n_nodes: int,
    k: int = 2,
    weight: float = 1.0,
    weight_dist: str = 'uniform',
    seed: Optional[int] = None,
) -> np.ndarray:
    """Ring lattice with k nearest neighbors (undirected)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    rng = np.random.default_rng(seed)
    W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    half_k = int(max(1, k // 2))
    base_weights = _sample_weights(weight_dist, (n_nodes, half_k), rng, scale=weight)
    for i in range(n_nodes):
        for idx, offset in enumerate(range(1, half_k + 1)):
            j = (i + offset) % n_nodes
            w_val = base_weights[i, idx] if base_weights.ndim == 2 else weight
            W[i, j] = w_val
            W[j, i] = w_val
    np.fill_diagonal(W, 0.0)
    return W


def small_world(
    n_nodes: int,
    k: int = 4,
    beta: float = 0.1,
    weight_dist: str = 'lognormal',
    weight_scale: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Watts-Strogatz style small-world graph (undirected)."""
    rng = np.random.default_rng(seed)
    if k >= n_nodes:
        raise ValueError("k must be < n_nodes for small_world generator.")

    # Start from ring lattice
    W = ring_lattice(n_nodes, k=k, weight=weight_scale, weight_dist=weight_dist, seed=seed)
    adjacency = W > 0

    # Rewire edges
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if not adjacency[i, j]:
                continue
            if rng.random() < beta:
                # Remove existing edge
                adjacency[i, j] = False
                adjacency[j, i] = False
                # Choose new target
                new_targets = [t for t in range(n_nodes) if t != i and not adjacency[i, t]]
                if not new_targets:
                    continue
                new_j = rng.choice(new_targets)
                adjacency[i, new_j] = True
                adjacency[new_j, i] = True

    weights = _sample_weights(weight_dist, adjacency.shape, rng, scale=weight_scale)
    W = adjacency.astype(np.float32) * weights.astype(np.float32)
    np.fill_diagonal(W, 0.0)
    return W


def stochastic_block_model(
    n_nodes: int,
    n_blocks: int = 4,
    p_in: float = 0.3,
    p_out: float = 0.05,
    weight_dist: str = 'lognormal',
    weight_scale: float = 1.0,
    seed: Optional[int] = None,
    block_sizes: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Simple stochastic block model with equal-sized blocks by default."""
    rng = np.random.default_rng(seed)
    if block_sizes is None:
        base = n_nodes // n_blocks
        block_sizes = [base] * n_blocks
        for idx in range(n_nodes - base * n_blocks):
            block_sizes[idx] += 1
    labels = []
    for b, size in enumerate(block_sizes):
        labels.extend([b] * size)
    labels = np.asarray(labels)

    W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    weights = _sample_weights(weight_dist, (n_nodes, n_nodes), rng, scale=weight_scale)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            prob = p_in if labels[i] == labels[j] else p_out
            if rng.random() < prob:
                W[i, j] = weights[i, j]
    return W.astype(np.float32)
