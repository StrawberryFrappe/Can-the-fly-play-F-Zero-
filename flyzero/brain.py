"""Leaky integrate-and-fire simulation of a whole connectome.

Same model and parameters as Shiu et al. 2024 (Nature 634, 210-219), which showed
that this very simple model, with weights taken straight from FlyWire synapse
counts, predicts sensorimotor circuits in the real fly:

    dv/dt = (g - (v - v_rest)) / tau_m        (held at v_reset while refractory)
    dg/dt = -g / tau_syn
    presynaptic spike -> g_post += w_syn * signed_synapse_count  (after a delay)

Sensory neurons are driven with Poisson spike trains whose events add a large
kick (``f_poisson * w_syn``) directly to the membrane potential.

Spikes are sparse, so each step only touches the rows of the weight matrix
belonging to neurons that fired. With numba installed the whole time loop runs
as one compiled kernel (several times faster); otherwise plain NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

try:
    import numba
except ImportError:  # pragma: no cover
    numba = None


def _kernel(steps, t, n_delay, v, g, ref, pending, counts, indptr, indices, data,
            input_idx, p_in, kick, decay_syn, decay_m, v_rest, v_th, v_reset, n_ref, seed):
    np.random.seed(seed)
    n = v.shape[0]
    for _ in range(steps):
        slot = t % n_delay
        buf = pending[slot]
        for i in range(n):
            g[i] = g[i] * decay_syn + buf[i]
            buf[i] = 0.0
            if ref[i] > 0:
                ref[i] -= 1
            else:
                v[i] += (g[i] - (v[i] - v_rest)) * decay_m
        for j in range(input_idx.shape[0]):
            if np.random.random() < p_in[j]:
                v[input_idx[j]] += kick
        for i in range(n):
            if v[i] >= v_th and ref[i] <= 0:
                v[i] = v_reset
                ref[i] = n_ref
                counts[i] += 1
                # lands n_delay steps from now, i.e. in this same ring slot
                for k in range(indptr[i], indptr[i + 1]):
                    buf[indices[k]] += data[k]
        t += 1
    return t


if numba is not None:
    _kernel = numba.njit(cache=True, fastmath=True)(_kernel)


@dataclass
class LIFParams:
    v_rest: float = -52.0   # mV
    v_reset: float = -52.0  # mV
    v_th: float = -45.0     # mV
    tau_m: float = 20.0     # ms, membrane time constant
    tau_syn: float = 5.0    # ms, synaptic time constant
    t_ref: float = 2.2      # ms, refractory period
    delay: float = 1.8      # ms, synaptic delay
    w_syn: float = 0.275    # mV per synapse
    f_poisson: float = 250.0  # Poisson input weight = f_poisson * w_syn
    dt: float = 0.1         # ms


class Brain:
    def __init__(self, weights: sp.csr_matrix, params: LIFParams | None = None, seed: int = 0,
                 backend: str = "auto"):
        self.p = p = params or LIFParams()
        self.backend = ("numba" if numba is not None else "numpy") if backend == "auto" else backend
        w = sp.csr_matrix(weights, dtype=np.float32)
        w.sort_indices()
        self.n = w.shape[0]
        self.indptr = w.indptr.astype(np.int64)
        self.indices = w.indices.astype(np.int32)
        self.data = (w.data * p.w_syn).astype(np.float32)
        self.rng = np.random.default_rng(seed)

        self.n_delay = max(1, int(round(p.delay / p.dt)))
        self.n_ref = int(round(p.t_ref / p.dt))
        self.decay_m = np.float32(p.dt / p.tau_m)
        self.decay_syn = np.float32(np.exp(-p.dt / p.tau_syn))
        self.kick = np.float32(p.f_poisson * p.w_syn)
        self.reset()

        # external drive: neuron index -> Poisson rate (Hz)
        self.input_idx = np.zeros(0, np.int64)
        self.input_rate = np.zeros(0, np.float32)

    def reset(self):
        self.v = np.full(self.n, self.p.v_rest, np.float32)
        self.g = np.zeros(self.n, np.float32)
        self.ref = np.zeros(self.n, np.int16)
        self.pending = np.zeros((self.n_delay, self.n), np.float32)
        self.t = 0
        self.spike_count = np.zeros(self.n, np.int32)

    def set_input(self, idx: np.ndarray, rate_hz: np.ndarray):
        """Poisson drive (Hz) for the given neurons, replacing previous input."""
        self.input_idx = np.asarray(idx, np.int64)
        self.input_rate = np.broadcast_to(np.asarray(rate_hz, np.float32), self.input_idx.shape).copy()

    def _propagate(self, spiking: np.ndarray, out: np.ndarray):
        starts = self.indptr[spiking]
        lens = self.indptr[spiking + 1] - starts
        total = int(lens.sum())
        if total == 0:
            return
        # flat positions of all outgoing synapses of the spiking neurons
        offs = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(total)
        out += np.bincount(self.indices[offs], weights=self.data[offs], minlength=self.n).astype(np.float32)

    def run(self, duration_ms: float) -> np.ndarray:
        """Advance the simulation; returns per-neuron spike counts for this window."""
        steps = max(1, int(round(duration_ms / self.p.dt)))
        p = self.p
        counts = np.zeros(self.n, np.int32)
        p_in = self.input_rate * (p.dt / 1000.0)
        if self.backend == "numba":
            self.t = _kernel(steps, self.t, self.n_delay, self.v, self.g, self.ref, self.pending,
                             counts, self.indptr, self.indices, self.data, self.input_idx,
                             p_in.astype(np.float64), self.kick, self.decay_syn, self.decay_m,
                             np.float32(p.v_rest), np.float32(p.v_th), np.float32(p.v_reset),
                             np.int16(self.n_ref), int(self.rng.integers(2**31)))
            self.spike_count += counts
            return counts
        for _ in range(steps):
            slot = self.t % self.n_delay
            # synaptic input arriving now
            self.g *= self.decay_syn
            self.g += self.pending[slot]
            self.pending[slot] = 0.0

            active = self.ref <= 0
            self.v += np.where(active, (self.g - (self.v - p.v_rest)) * self.decay_m, 0.0).astype(np.float32)
            self.ref[~active] -= 1

            if len(self.input_idx):
                hit = self.rng.random(len(self.input_idx)) < p_in
                self.v[self.input_idx[hit]] += self.kick

            spiking = np.flatnonzero((self.v >= p.v_th) & active)
            if len(spiking):
                self.v[spiking] = p.v_reset
                self.ref[spiking] = self.n_ref
                counts[spiking] += 1
                # lands n_delay steps from now, i.e. in this same ring slot
                self._propagate(spiking, self.pending[slot])
            self.t += 1
        self.spike_count += counts
        return counts
