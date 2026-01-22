"""
WendlingNetwork: Vectorized Wendling-Chauvel model for N nodes.

Extends the single-node model with network coupling:
    - Shared local parameters across all nodes
    - Coupling term u_net injected into y5 derivative
    - Outputs time-major LFP with shape (T, N)
"""

from typing import Dict, Optional, Any
import numpy as np
import brainpy as bp
import brainpy.math as bm

from wendling_sim.model.params import STANDARD_PARAMS, merge_params


class WendlingNetwork(bp.DynamicalSystem):
    """
    N-node Wendling-Chauvel neural mass model with coupling.

    State variables (per node):
        y0: PSP from pyramidal -> excitatory
        y1: PSP from excitatory -> pyramidal
        y2: PSP from slow inhibitory -> pyramidal
        y3: PSP from fast inhibitory -> pyramidal
        y4: PSP from pyramidal -> slow inhibitory
        y5-y9: Derivatives of y0-y4

    Coupling:
        out_j(t) = S(v_pyr_j(t)) where v_pyr = y1 - y2 - y3
        u_net_i(t) = G_net * sum_j W[i, j] * out_j(t)
        u_net injected into y5 derivative.
    """

    def __init__(
        self,
        n_nodes: int,
        params: Optional[Dict[str, float]] = None,
        W=None,
        G_net: float = 1.0,
        noise_cfg: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if W is None:
            raise ValueError("Connectivity matrix W is required for WendlingNetwork.")
        if W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be square. Got {W.shape}.")
        if W.shape[0] != n_nodes:
            raise ValueError(f"W shape {W.shape} does not match n_nodes={n_nodes}.")

        # Merge parameters with defaults
        p = merge_params(STANDARD_PARAMS, params or {})

        # Network parameters
        self.n_nodes = n_nodes
        self.G_net = float(G_net)
        self.W = bm.asarray(W, dtype=bm.float32)

        # Synaptic gains
        self.A = p['A']
        self.B = p['B']
        self.G = p['G']

        # Time constants
        self.a = p['a']
        self.b = p['b']
        self.g = p['g']

        # Sigmoid parameters
        self.e0 = p['e0']
        self.v0 = p['v0']
        self.r = p['r']

        # Connectivity scaling (dimensionless)
        C = p['C']
        self.C1 = p['C1'] * C
        self.C2 = p['C2'] * C
        self.C3 = p['C3'] * C
        self.C4 = p['C4'] * C
        self.C5 = p['C5'] * C
        self.C6 = p['C6'] * C
        self.C7 = p['C7'] * C

        # Input parameters
        self.p_mean = self._to_vector(p['p_mean'])
        self.p_sigma = self._to_vector(p['p_sigma'])

        # Noise policy
        noise_cfg = noise_cfg or {}
        self.noise_mode = noise_cfg.get('mode', 'gaussian')
        self.noise_shared = noise_cfg.get('shared', False)

        # State variables (vectorized across nodes)
        zeros = bm.zeros(n_nodes, dtype=bm.float32)
        self.y0 = bm.Variable(zeros.copy())
        self.y1 = bm.Variable(zeros.copy())
        self.y2 = bm.Variable(zeros.copy())
        self.y3 = bm.Variable(zeros.copy())
        self.y4 = bm.Variable(zeros.copy())
        self.y5 = bm.Variable(zeros.copy())
        self.y6 = bm.Variable(zeros.copy())
        self.y7 = bm.Variable(zeros.copy())
        self.y8 = bm.Variable(zeros.copy())
        self.y9 = bm.Variable(zeros.copy())

        # LFP output (time-major monitoring will give (T, N))
        self.lfp = bm.Variable(zeros.copy())

        # Store params for metadata
        self._params = p

        # Optional random or provided initial conditions
        self._init_state(p, noise_cfg)

    def _init_state(self, p: Dict[str, Any], noise_cfg: Dict[str, Any]):
        """Initialize state variables from params or random seed."""
        def _as_vector(value, name):
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 0:
                arr = np.full(self.n_nodes, float(arr), dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            if arr.shape[0] != self.n_nodes:
                raise ValueError(f"{name} must have length {self.n_nodes}, got {arr.shape}")
            return arr

        keys = ["y0_init", "y1_init", "y2_init", "y3_init", "y4_init",
                "y5_init", "y6_init", "y7_init", "y8_init", "y9_init"]
        has_init = any(k in p for k in keys)
        if has_init:
            y0 = _as_vector(p.get("y0_init", 0.0), "y0_init")
            y1 = _as_vector(p.get("y1_init", 0.0), "y1_init")
            y2 = _as_vector(p.get("y2_init", 0.0), "y2_init")
            y3 = _as_vector(p.get("y3_init", 0.0), "y3_init")
            y4 = _as_vector(p.get("y4_init", 0.0), "y4_init")
            y5 = _as_vector(p.get("y5_init", 0.0), "y5_init")
            y6 = _as_vector(p.get("y6_init", 0.0), "y6_init")
            y7 = _as_vector(p.get("y7_init", 0.0), "y7_init")
            y8 = _as_vector(p.get("y8_init", 0.0), "y8_init")
            y9 = _as_vector(p.get("y9_init", 0.0), "y9_init")
        elif p.get("random_init", False) and self.n_nodes > 1:
            seed = p.get("init_seed", noise_cfg.get("seed", None))
            rng = np.random.default_rng(seed)
            y0 = rng.uniform(-0.5, 0.5, self.n_nodes).astype(np.float32)
            y1 = rng.uniform(-0.5, 0.5, self.n_nodes).astype(np.float32)
            y2 = rng.uniform(-0.5, 0.5, self.n_nodes).astype(np.float32)
            y3 = rng.uniform(-0.5, 0.5, self.n_nodes).astype(np.float32)
            y4 = rng.uniform(-0.5, 0.5, self.n_nodes).astype(np.float32)
            y5 = rng.uniform(-0.1, 0.1, self.n_nodes).astype(np.float32)
            y6 = rng.uniform(-0.1, 0.1, self.n_nodes).astype(np.float32)
            y7 = rng.uniform(-0.1, 0.1, self.n_nodes).astype(np.float32)
            y8 = rng.uniform(-0.1, 0.1, self.n_nodes).astype(np.float32)
            y9 = rng.uniform(-0.1, 0.1, self.n_nodes).astype(np.float32)
        else:
            return

        self.y0.value = bm.asarray(y0)
        self.y1.value = bm.asarray(y1)
        self.y2.value = bm.asarray(y2)
        self.y3.value = bm.asarray(y3)
        self.y4.value = bm.asarray(y4)
        self.y5.value = bm.asarray(y5)
        self.y6.value = bm.asarray(y6)
        self.y7.value = bm.asarray(y7)
        self.y8.value = bm.asarray(y8)
        self.y9.value = bm.asarray(y9)
        self.lfp.value = self.y1.value - self.y2.value - self.y3.value

    def _to_vector(self, value):
        """Ensure inputs are vectors of length n_nodes."""
        arr = bm.asarray(value, dtype=bm.float32)
        if arr.ndim == 0:
            arr = bm.ones(self.n_nodes, dtype=bm.float32) * float(arr)
        return arr

    def _draw_noise(self):
        """Generate noise term with optional sharing across nodes."""
        if self.noise_shared:
            if self.noise_mode == 'uniform':
                sample = bm.random.uniform(-1.0, 1.0)
            else:
                sample = bm.random.randn()
            return bm.ones(self.n_nodes, dtype=bm.float32) * sample

        if self.noise_mode == 'uniform':
            return bm.random.uniform(-1.0, 1.0, self.n_nodes)
        return bm.random.randn(self.n_nodes)

    def sigmoid(self, v):
        """Sigmoid activation function (firing rate)."""
        return (2.0 * self.e0) / (1.0 + bm.exp(self.r * (self.v0 - v)))

    def update(self, inp=None):
        """
        Single-step update for all nodes.

        Args:
            inp: Optional external input. If dict, supports:
                - 'p_ext': additive input to p(t) per node
                - 'u_stim': additive drive injected into y6 derivative
        """
        dt = bm.dt  # seconds

        # External input p(t)
        noise = self._draw_noise() * self.p_sigma
        p_t = self.p_mean + noise
        u_stim = 0.0
        if inp is not None:
            if isinstance(inp, dict):
                if 'p_ext' in inp:
                    p_t = p_t + bm.asarray(inp['p_ext'])
                if 'u_stim' in inp:
                    u_stim = inp['u_stim']
            else:
                p_t = p_t + bm.asarray(inp)

        # Current state
        y0, y1, y2, y3, y4 = self.y0.value, self.y1.value, self.y2.value, self.y3.value, self.y4.value
        y5, y6, y7, y8, y9 = self.y5.value, self.y6.value, self.y7.value, self.y8.value, self.y9.value

        # Coupling variable and network input
        v_pyr = y1 - y2 - y3
        out = self.sigmoid(v_pyr)
        u_net = self.G_net * bm.matmul(self.W, out)

        # Derivatives (vectorized)
        dy0 = y5
        dy5 = self.A * self.a * (self.sigmoid(y1 - y2 - y3) ) - 2.0 * self.a * y5 - self.a**2 * y0

        dy1 = y6
        dy6 = self.A * self.a * (self.C2 * self.sigmoid(self.C1 * y0) + p_t + u_stim + u_net) - 2.0 * self.a * y6 - self.a**2 * y1

        dy2 = y7
        dy7 = self.B * self.b * self.C4 * self.sigmoid(self.C3 * y0) - 2.0 * self.b * y7 - self.b**2 * y2

        dy3 = y8
        dy8 = self.G * self.g * self.C7 * self.sigmoid(self.C5 * y0 - self.C6 * y4) - 2.0 * self.g * y8 - self.g**2 * y3

        dy4 = y9
        dy9 = self.B * self.b * self.sigmoid(self.C3 * y0) - 2.0 * self.b * y9 - self.b**2 * y4

        # Euler integration
        self.y0.value = y0 + dt * dy0
        self.y1.value = y1 + dt * dy1
        self.y2.value = y2 + dt * dy2
        self.y3.value = y3 + dt * dy3
        self.y4.value = y4 + dt * dy4
        self.y5.value = y5 + dt * dy5
        self.y6.value = y6 + dt * dy6
        self.y7.value = y7 + dt * dy7
        self.y8.value = y8 + dt * dy8
        self.y9.value = y9 + dt * dy9

        # LFP proxy per node
        self.lfp.value = self.y1.value - self.y2.value - self.y3.value

    def reset_state(self, batch_size=None):
        """Reset all state variables to zero."""
        zeros = bm.zeros(self.n_nodes, dtype=bm.float32)
        self.y0.value = zeros.copy()
        self.y1.value = zeros.copy()
        self.y2.value = zeros.copy()
        self.y3.value = zeros.copy()
        self.y4.value = zeros.copy()
        self.y5.value = zeros.copy()
        self.y6.value = zeros.copy()
        self.y7.value = zeros.copy()
        self.y8.value = zeros.copy()
        self.y9.value = zeros.copy()
        self.lfp.value = zeros.copy()

    @property
    def params(self) -> Dict:
        """Return current parameter dict."""
        return dict(self._params)
