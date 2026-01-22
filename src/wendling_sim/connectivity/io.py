"""
Connectivity I/O and normalization utilities.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import numpy as np

from wendling_sim.connectivity import generators


@dataclass
class Connectivity:
    """Standard connectivity representation."""
    W: np.ndarray
    labels: Optional[list] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.W.shape[0])


def _load_matrix_from_path(path: Path) -> np.ndarray:
    """Load connectivity matrix from .npy, .csv, or .mat."""
    if not path.exists():
        raise FileNotFoundError(f"Connectivity file not found: {path}")
    if path.suffix.lower() == '.npy':
        return np.load(path)
    if path.suffix.lower() == '.csv':
        return np.loadtxt(path, delimiter=',')
    if path.suffix.lower() == '.mat':
        try:
            import scipy.io as sio
        except ImportError as exc:
            raise ImportError("scipy is required to load .mat connectivity files") from exc
        mat = sio.loadmat(path)
        # Heuristics: pick common keys
        for key in ['W', 'sc', 'len', 'data']:
            if key in mat and isinstance(mat[key], np.ndarray):
                arr = mat[key]
                if arr.ndim >= 2:
                    return np.asarray(arr)
        # Fallback: first ndarray with ndim>=2
        for v in mat.values():
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                return np.asarray(v)
        raise ValueError(f"No matrix found in .mat file: {path}")
    raise ValueError(f"Unsupported connectivity format: {path.suffix}")


def _load_labels(path: Path) -> list:
    """Load labels from txt/csv/json."""
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    if path.suffix.lower() in {'.txt', '.csv'}:
        text = path.read_text().strip()
        return [line.strip() for line in text.splitlines() if line.strip()]
    if path.suffix.lower() == '.json':
        import json
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported labels format: {path.suffix}")


def _normalize(W: np.ndarray, mode: str = 'row_sum') -> np.ndarray:
    """Normalize connectivity weights."""
    mode = (mode or 'row_sum').lower()
    if mode == 'none':
        return W
    if mode == 'row_sum':
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return W / row_sums
    if mode == 'max':
        max_val = np.max(W)
        return W / max_val if max_val > 0 else W
    raise ValueError(f"Unknown normalization mode: {mode}")


def _build_from_generator(cfg: Dict[str, Any], n_nodes: int) -> np.ndarray:
    """Create connectivity matrix from generator config."""
    name = cfg.get('name')
    if not name:
        raise ValueError("generator config requires a 'name'.")
    options = cfg.get('options', {})
    seed = options.get('seed', cfg.get('seed', None))
    if not hasattr(generators, name):
        raise ValueError(f"Unknown generator: {name}")
    gen_fn = getattr(generators, name)
    return gen_fn(n_nodes=n_nodes, seed=seed, **{k: v for k, v in options.items() if k != 'seed'})


def load_connectivity(network_cfg: Optional[Dict[str, Any]] = None) -> Connectivity:
    """
    Load or generate connectivity according to config.

    network_cfg keys (optional):
        - n_nodes: number of nodes (required for generators)
        - W: direct matrix or list
        - W_path: path to matrix (.npy or .csv)
        - generator: {name, options}
        - builder: callable returning (W, labels?, meta?)
        - normalize: none | row_sum | max (default row_sum)
        - remove_self_loops: bool (default True)
        - labels_path: optional labels file
    """
    cfg = network_cfg or {}
    n_nodes = cfg.get('n_nodes', None)
    labels = cfg.get('labels', None)
    meta = dict(cfg.get('meta', {}))

    # Resolve connectivity source
    if 'W' in cfg and cfg['W'] is not None:
        W = np.asarray(cfg['W'], dtype=np.float32)
    elif 'W_path' in cfg and cfg['W_path'] is not None:
        W = _load_matrix_from_path(Path(cfg['W_path']))
    elif 'generator' in cfg and cfg['generator'] is not None:
        if n_nodes is None:
            raise ValueError("n_nodes is required when using a generator.")
        W = _build_from_generator(cfg['generator'], n_nodes)
    elif 'builder' in cfg and cfg['builder'] is not None:
        builder: Callable = cfg['builder']
        if n_nodes is None:
            raise ValueError("n_nodes is required when using a builder.")
        built = builder(n_nodes=n_nodes, seed=cfg.get('seed', None), **cfg.get('builder_kwargs', {}))
        if isinstance(built, tuple):
            W = built[0]
            if len(built) > 1 and labels is None:
                labels = built[1]
            if len(built) > 2 and not meta:
                meta = built[2]
        else:
            W = built
    else:
        # Default: no coupling
        if n_nodes is None:
            n_nodes = 1
        W = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    if W.shape[0] != W.shape[1]:
        raise ValueError(f"Connectivity matrix must be square. Got shape {W.shape}.")
    if n_nodes is None:
        n_nodes = W.shape[0]
    if W.shape[0] != n_nodes:
        raise ValueError(f"W shape {W.shape} does not match n_nodes={n_nodes}.")

    # Remove self-loops unless requested
    if cfg.get('remove_self_loops', True):
        np.fill_diagonal(W, 0.0)

    # Normalize weights
    W = _normalize(W.astype(np.float32), cfg.get('normalize', 'row_sum'))

    # Load labels from path if provided
    labels_path = cfg.get('labels_path')
    if labels_path and labels is None:
        labels = _load_labels(Path(labels_path))

    meta.setdefault('note', 'W[i,j] = j->i (target row, source column)')
    return Connectivity(W=W.astype(np.float32), labels=labels, meta=meta)
