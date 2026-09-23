"""The same leaky integrate-and-fire brain as ``brain.py``, for many flies at once on a GPU (CuPy).

B independent flies share the FlyWire weights; each has its own membrane state, input and
copy of the *plastic* synapses (the few thousand that learning may change). Per time step:

* ``_update``: one thread per (fly, neuron): synaptic decay + arriving input, membrane update,
  Poisson input (counter-based hash RNG), spike test. Spiking neurons go on a list.
* ``_propagate``: one warp per spike: scatters the neuron's CSR row (fixed weights, plus that
  fly's plastic weights) into the delay ring buffer with atomic adds.

Equations, parameters and the order of operations match ``brain._kernel``, so without Poisson
input both give the same spikes. Two differences, both far below anything that matters: float
summation order in the atomic adds, and conductances below 1e-7 mV (or voltages within 1e-5 mV of
rest, with no input) are snapped to exactly 0 / rest, so quiet neurons stop costing memory writes.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import cupy as cp

from .brain import LIFParams

_SRC = r"""
#define CHUNK 256
__device__ __forceinline__ unsigned int mix(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}

extern "C" __global__ void update(
    const int B, const int n, const int slot, const unsigned int step, const unsigned int seed,
    float* v, float* g, signed char* ref, float* pending, int* counts,
    const int* ext_idx, const int n_ext, const float* p_ext, const float* bias_ext,
    const float kick, const float decay_syn, const float decay_m,
    const float v_rest, const float v_th, const float v_reset, const int n_ref,
    const long long* indptr, const long long* pl_indptr, int* spikes, int* n_spikes)
{
    // memory-bound: per-fly arrays are read once and written only when they change; the
    // input/bias map (ext_idx) is shared by all flies and stays in L2
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)B * n) return;
    int b = (int)(tid / n);
    int i = (int)(tid - (long long)b * n);
    long long pi = (long long)slot * B * n + tid;       // pending is (n_delay, B, n)
    float buf = pending[pi];
    if (buf != 0.0f) pending[pi] = 0.0f;
    float g0 = g[tid];
    float gi = g0 * decay_syn + buf;
    if (fabsf(gi) < 1e-7f) gi = 0.0f;
    int r0 = ref[tid], r = r0;
    float v0 = v[tid], vi = v0;
    int j = ext_idx[i];
    long long e = (long long)b * n_ext + j;
    float bias = j >= 0 ? bias_ext[e] : 0.0f;
    if (r > 0) { r -= 1; }
    else if (gi != 0.0f || bias != 0.0f || vi != v_rest) {
        vi += (gi + bias - (vi - v_rest)) * decay_m;
        if (gi == 0.0f && bias == 0.0f && fabsf(vi - v_rest) < 1e-5f) vi = v_rest;
    }
    if (j >= 0) {
        float p = p_ext[e];
        if (p > 0.0f) {
            unsigned int h = mix((unsigned int)tid ^ mix(step * 0x9e3779b9U ^ mix(seed + (unsigned int)b)));
            if ((h >> 8) * (1.0f / 16777216.0f) < p) vi += kick;
        }
    }
    if (vi >= v_th && r <= 0) {
        vi = v_reset; r = n_ref; counts[tid] += 1;
        // work items of <= CHUNK synapses each, so one busy neuron can't stall a whole step
        long long deg = indptr[i + 1] - indptr[i];
        int nch = (int)((deg + CHUNK - 1) / CHUNK);
        if (nch == 0 && pl_indptr[i + 1] > pl_indptr[i]) nch = 1;
        if (nch > 0) {
            int k = atomicAdd(n_spikes, nch);
            for (int c = 0; c < nch; c++) spikes[k + c] = (int)(tid << 6) | c;
        }
    }
    if (gi != g0) g[tid] = gi;
    if (vi != v0) v[tid] = vi;
    if (r != r0) ref[tid] = (signed char)r;
}

extern "C" __global__ void propagate(
    const int B, const int n, const int slot, const int* spikes, int* n_spikes, const int next,
    const long long* indptr, const int* indices, const float* data,
    const long long* pl_indptr, const int* pl_post, const float* pl_w, const int n_pl,
    float* pending)
{
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int n_warps = (gridDim.x * blockDim.x) >> 5;
    const int total = n_spikes[next ^ 1];               // this step's list
    float* buf = pending + (long long)slot * B * n;
    for (int s = warp; s < total; s += n_warps) {
        int item = spikes[s];
        int tid = item >> 6, c = item & 63;
        int b = tid / n, i = tid - b * n;
        float* row = buf + (long long)b * n;
        long long end = indptr[i + 1], k0 = indptr[i] + (long long)c * CHUNK;
        long long k1 = k0 + CHUNK < end ? k0 + CHUNK : end;
        for (long long k = k0 + lane; k < k1; k += 32)
            atomicAdd(row + indices[k], data[k]);
        if (c == 0)
            for (long long k = pl_indptr[i] + lane; k < pl_indptr[i + 1]; k += 32)
                atomicAdd(row + pl_post[k], pl_w[(long long)b * n_pl + k]);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) n_spikes[next] = 0;   // the list the next step fills
}
"""

_mod = None


def _kernels():
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC, options=("--use_fast_math",))
    return _mod.get_function("update"), _mod.get_function("propagate")


class BatchBrain:
    """``batch`` flies with shared FlyWire weights and private plastic synapses.

    ``plastic_pos``: indices into the CSR ``data`` (as in ``Brain.data``) of synapses that may
    differ between flies. ``self.pw`` (batch, n_plastic) holds their current weights in mV,
    already multiplied by ``w_syn`` like ``Brain.data``.
    """

    def __init__(self, weights: sp.csr_matrix, params: LIFParams | None = None, batch: int = 1,
                 seed: int = 0, plastic_pos: np.ndarray | None = None):
        self.p = p = params or LIFParams()
        self.B = batch
        w = sp.csr_matrix(weights, dtype=np.float32)
        w.sort_indices()
        self.n = n = w.shape[0]
        # host copies with the same layout as brain.Brain (plasticity code indexes into them)
        self.indptr = w.indptr.astype(np.int64)
        self.indices = w.indices.astype(np.int32)
        self.data = (w.data * p.w_syn).astype(np.float32)
        pos = np.zeros(0, np.int64) if plastic_pos is None else np.sort(np.asarray(plastic_pos, np.int64))
        self.plastic_pos = pos
        rows = np.repeat(np.arange(n), np.diff(self.indptr))
        self.pl_pre = rows[pos]
        self.pl_post = self.indices[pos]
        fixed = self.data.copy()
        fixed[pos] = 0.0
        self.d_indptr = cp.asarray(self.indptr)
        self.d_indices = cp.asarray(self.indices)
        self.d_data = cp.asarray(fixed)
        self.d_pl_indptr = cp.asarray(np.searchsorted(self.pl_pre, np.arange(n + 1)).astype(np.int64))
        self.d_pl_post = cp.asarray(self.pl_post.astype(np.int32))
        self.w0 = self.data[pos].copy()
        self.pw = cp.asarray(np.tile(self.w0, (batch, 1)))
        self.n_delay = max(1, int(round(p.delay / p.dt)))
        self.n_ref = int(round(p.t_ref / p.dt))
        assert self.n_ref < 127, "refractory counter is int8 on the GPU"
        # spike work items are (fly * n + neuron) << 6 | chunk
        assert np.diff(self.indptr).max() <= 64 * 256 and batch * n < 2 ** 25
        self.decay_m = np.float32(p.dt / p.tau_m)
        self.decay_syn = np.float32(np.exp(-p.dt / p.tau_syn))
        self.kick = np.float32(p.f_poisson * p.w_syn)
        self.seed = np.uint32(seed)
        self.step_no = 0
        # neurons that take outside input (Poisson rates and/or a bias current): "ext" slots
        self.ext = np.zeros(0, np.int64)
        self._slot = np.full(n, -1, np.int32)
        self.d_ext_idx = cp.asarray(self._slot)
        self.p_ext = cp.zeros((batch, 0), cp.float32)      # Poisson input, Hz
        self.bias_ext = cp.zeros((batch, 0), cp.float32)   # mV
        # work-item list (see CHUNK): enough even if every neuron fired at once
        self.spikes = cp.zeros(batch * (n + w.nnz // 256 + 1), cp.int32)
        self.n_spikes = cp.zeros(2, cp.int32)
        self._upd, self._prop = _kernels()
        self.reset()

    def reset(self, flies=None):
        """Back to rest (all flies, or the given ones). Plastic weights are kept."""
        if flies is None:
            self.v = cp.full((self.B, self.n), self.p.v_rest, cp.float32)
            self.g = cp.zeros((self.B, self.n), cp.float32)
            self.ref = cp.zeros((self.B, self.n), cp.int8)
            self.pending = cp.zeros((self.n_delay, self.B, self.n), cp.float32)
            self.t = 0
            return
        f = cp.asarray(np.atleast_1d(flies))
        self.v[f] = self.p.v_rest
        self.g[f] = 0.0
        self.ref[f] = 0
        self.pending[:, f] = 0.0

    def slots(self, idx) -> cp.ndarray:
        """Ext slots of neurons ``idx`` (adding any that don't have one yet)."""
        idx = np.asarray(idx, np.int64).ravel()
        new = np.unique(idx[self._slot[idx] < 0])
        if len(new):
            k = len(self.ext)
            self.ext = np.r_[self.ext, new]
            self._slot[new] = np.arange(k, k + len(new), dtype=np.int32)
            self.d_ext_idx = cp.asarray(self._slot)
            pad = cp.zeros((self.B, len(new)), cp.float32)
            self.p_ext = cp.concatenate([self.p_ext, pad], 1)
            self.bias_ext = cp.concatenate([self.bias_ext, pad], 1)
        return cp.asarray(self._slot[idx])

    def set_bias(self, idx, mv, flies=None):
        """Steady depolarisation (mV) for neurons ``idx``, as ``Brain.set_bias`` (replaces it)."""
        j = self.slots(idx)
        sel = slice(None) if flies is None else cp.asarray(np.atleast_1d(flies))
        self.bias_ext[sel] = 0.0
        self.bias_ext[sel, j] = cp.asarray(mv, cp.float32)

    def set_input(self, idx, rate_hz):
        """Poisson drive (Hz): ``rate_hz`` (batch, len(idx)) or broadcastable, replacing earlier input."""
        j = self.slots(idx)
        self.p_ext.fill(0)
        self.p_ext[:, j] = cp.asarray(rate_hz, cp.float32) * np.float32(self.p.dt / 1000.0)

    def run(self, duration_ms: float) -> cp.ndarray:
        """Advance every fly; returns spike counts (batch, n) on the GPU for this window."""
        steps = max(1, int(round(duration_ms / self.p.dt)))
        p = self.p
        counts = cp.zeros((self.B, self.n), cp.int32)
        blocks = (self.B * self.n + 255) // 256
        n_pl = np.int32(self.pw.shape[1])
        pl_w = self.pw if self.pw.size else cp.zeros(1, cp.float32)
        n_ext = np.int32(len(self.ext))
        p_ext = self.p_ext if n_ext else cp.zeros(1, cp.float32)
        bias_ext = self.bias_ext if n_ext else cp.zeros(1, cp.float32)
        for _ in range(steps):
            slot = np.int32(self.t % self.n_delay)
            nxt = np.int32((self.step_no + 1) & 1)
            self._upd((blocks,), (256,), (
                np.int32(self.B), np.int32(self.n), slot, np.uint32(self.step_no & 0xFFFFFFFF), self.seed,
                self.v, self.g, self.ref, self.pending, counts, self.d_ext_idx, n_ext, p_ext, bias_ext,
                self.kick, self.decay_syn, self.decay_m, np.float32(p.v_rest), np.float32(p.v_th),
                np.float32(p.v_reset), np.int32(self.n_ref), self.d_indptr, self.d_pl_indptr,
                self.spikes, self.n_spikes[1 - nxt:]))
            self._prop((240,), (256,), (
                np.int32(self.B), np.int32(self.n), slot, self.spikes, self.n_spikes, nxt,
                self.d_indptr, self.d_indices, self.d_data, self.d_pl_indptr, self.d_pl_post, pl_w, n_pl,
                self.pending))
            self.t += 1
            self.step_no += 1
        return counts
