"""Leaky integrate-and-fire simulation of a whole connectome.

Same model and parameters as Shiu et al. 2024 (Nature 634, 210-219), which showed
that this very simple model, with weights taken straight from FlyWire synapse
counts, predicts sensorimotor circuits in the real fly:

    dv/dt = (g - (v - v_rest)) / tau_m        (held at v_reset while refractory)
    dg/dt = -g / tau_syn
    presynaptic spike -> g_post += w_syn * signed_synapse_count  (after a delay)

Sensory neurons are driven with Poisson spike trains whose events add a large
kick (``f_poisson * w_syn``) directly to the membrane potential.

Everything is plain NumPy/SciPy. Spikes are sparse, so each step only touches
the rows of the weight matrix belonging to neurons that fired.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


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
    def __init__(self, weights: sp.csr_matrix, params: LIFParams | None = None, seed: int = 0):
        self.p = p = params or LIFParams()
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
