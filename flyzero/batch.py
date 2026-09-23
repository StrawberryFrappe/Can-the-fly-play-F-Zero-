"""Motor readout and learning rules for a ``gpu.BatchBrain`` (many flies at once).

Same rules as ``motor.MotorReadout``, ``instruct.InstructedPlasticity`` and
``learning.RewardPlasticity``; the only structural addition is ``fly_of_slot``: several slots
(simulated copies running different races) can belong to one fly. Their weight changes are
averaged and applied to every copy, i.e. one fly learning from several races at once, which is
the sequential rule with a larger batch.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import cupy as cp

from .connectome import Connectome
from .instruct import GROUPS as _GROUPS
from .instruct import InstructParams

# the batched rules also instruct the giant fiber (boost), so it stays quiet unless boosting
GROUPS = {**_GROUPS, "gf": ("DNp01", None)}
PLASTIC_TYPES = ("DNa02", "DNa01", "DNg02*", "DNp09", "MDN", "DNp01")
from .learning import LearnParams
from .motor import BUTTONS, MotorParams

_DEEP_SRC = r"""
extern "C" __global__ void deep_update(
    const int n_syn, const int B, const int* sel, const int* pre_k, const int* post_l1,
    const float* trace, const int n_tr, const float* delta, const int n_l1,
    float* pw, const int n_pl, const float* sign, const float* wmax,
    const float* coef, const int* fos, const int n_flies)
{
    // one thread per deep synapse: average the change over each fly's slots, apply to all of them
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_syn) return;
    int j = sel[s], pk = pre_k[s], pl = post_l1[s];
    float acc[16];
    for (int f = 0; f < n_flies; f++) acc[f] = 0.0f;
    for (int b = 0; b < B; b++)
        if (coef[b] != 0.0f)
            acc[fos[b]] += coef[b] * trace[(long long)b * n_tr + pk] * delta[(long long)b * n_l1 + pl];
    float sg = sign[j], mx = wmax[j];
    for (int b = 0; b < B; b++) {
        float d = acc[fos[b]];
        if (d == 0.0f) continue;
        long long q = (long long)b * n_pl + j;
        float w = sg * (pw[q] + d);
        pw[q] = sg * fminf(fmaxf(w, 0.0f), mx);
    }
}
"""

READOUT = ["steer_left", "steer_right", "lean_left", "lean_right", "wing_left", "wing_right",
           "accelerate", "brake", "boost"]


def plastic_positions(conn: Connectome, types=PLASTIC_TYPES) -> np.ndarray:
    """CSR positions (``Brain.data`` layout) of every FlyWire synapse onto the given cell types."""
    targets = np.unique(np.concatenate([conn.find(t) for t in types]))
    is_t = np.zeros(conn.n, bool)
    is_t[targets] = True
    w = sp.csr_matrix(conn.weights, dtype=np.float32)
    w.sort_indices()
    return np.flatnonzero(is_t[w.indices])


def targets(x: np.ndarray, p: InstructParams) -> list:
    """Target rates in ``GROUPS`` order for an intent vector (steer, lean, gas, brake, boost)."""
    from .instruct import targets_from_intent

    t = targets_from_intent(x, p)
    t["gf"] = p.low_hz + (p.high_hz - p.low_hz) * float(x[4])
    return [t[g] for g in GROUPS]


class BatchMotor:
    """``MotorReadout`` for every slot: smoothed DN rates -> button dicts."""

    def __init__(self, conn: Connectome, batch: int, params: MotorParams | None = None):
        self.p = p = params or MotorParams()
        self.B = batch
        idx, grp = [], []
        for k, name in enumerate(READOUT):
            t, side = p.groups[name]
            ii = conn.find(t, side)
            idx.append(ii)
            grp.append(np.full(len(ii), k))
        self.idx = cp.asarray(np.concatenate(idx))
        g = np.concatenate(grp)
        self.size = np.bincount(g, minlength=len(READOUT)).astype(np.float32)
        # (neurons, groups) averaging matrix
        m = np.zeros((len(g), len(READOUT)), np.float32)
        m[np.arange(len(g)), g] = 1.0 / np.maximum(self.size[g], 1)
        self.avg = cp.asarray(m)
        self.rates = np.zeros((batch, len(READOUT)), np.float32)

    def reset(self, slots=None):
        if slots is None:
            self.rates[:] = 0
        else:
            self.rates[np.atleast_1d(slots)] = 0

    def update(self, counts: cp.ndarray, window_ms: float) -> list[dict]:
        a = 1.0 - np.exp(-window_ms / self.p.tau_ms)
        inst = cp.asnumpy(counts[:, self.idx].astype(cp.float32) @ self.avg) * (1000.0 / window_ms)
        self.rates += a * (inst - self.rates)
        return [self.buttons(r) for r in self.rates]

    def buttons(self, r) -> dict:
        p = self.p
        steer = (r[0] - r[1]) + (r[5] - r[4]) + p.steer_bias
        lean = r[2] - r[3]
        b = {k: False for k in BUTTONS}
        b["LEFT"] = bool(steer > p.steer_threshold)
        b["RIGHT"] = bool(steer < -p.steer_threshold)
        b["L"] = bool(lean > p.lean_threshold)
        b["R"] = bool(lean < -p.lean_threshold)
        b["B"] = bool(r[6] > p.accel_threshold)
        b["Y"] = bool(r[7] > p.brake_threshold)
        b["A"] = bool(r[8] > p.boost_threshold)
        return b


class _Plastic:
    """Shared bookkeeping: the brain's plastic synapses, their caps and slot -> fly grouping."""

    def __init__(self, brain, cap: float, fly_of_slot=None):
        self.brain = brain
        self.B = brain.B
        self.pre = cp.asarray(brain.pl_pre)
        self.post = cp.asarray(brain.pl_post)
        self.w0 = cp.asarray(brain.w0)
        self.sign = cp.sign(self.w0)
        self.wmax = cap * cp.maximum(cp.abs(self.w0), brain.p.w_syn)
        fos = np.arange(self.B) if fly_of_slot is None else np.asarray(fly_of_slot)
        self.fly_of_slot = fos
        self.n_flies = int(fos.max()) + 1
        # (flies, slots) averaging matrix: mean change over the slots that are learning
        self._fos = cp.asarray(fos)

    def apply(self, dw: cp.ndarray, active: np.ndarray, sel=None):
        """Average ``dw`` (slots, n_pl) over each fly's active slots; update every slot of the fly.
        With ``sel`` (plastic indices), ``dw`` is (slots, len(sel)) and only those change."""
        act = cp.asarray(active.astype(np.float32))
        onehot = cp.zeros((self.n_flies, self.B), cp.float32)
        onehot[self._fos, cp.arange(self.B)] = act
        n = onehot.sum(1, keepdims=True)
        per_fly = (onehot / cp.maximum(n, 1.0)) @ dw
        if sel is None:
            w = self.brain.pw + per_fly[self._fos]
            self.brain.pw[...] = self.sign * cp.clip(self.sign * w, 0.0, self.wmax)
        else:
            sign = self.sign[sel]
            w = self.brain.pw[:, sel] + per_fly[self._fos]
            self.brain.pw[:, sel] = sign * cp.clip(sign * w, 0.0, self.wmax[sel])

    def state(self, fly: int) -> dict:
        slot = int(np.flatnonzero(self.fly_of_slot == fly)[0])
        st = {"pos": self.brain.plastic_pos.astype(np.int32), "data": cp.asnumpy(self.brain.pw[slot]),
              "w0": cp.asnumpy(self.w0)}
        if getattr(self, "ib", None) is not None:
            st["bias_idx"], st["bias"] = self.dn, cp.asnumpy(self.ib[slot])
        return st

    def load(self, st: dict, flies=None):
        """Load weights saved by this class or by ``InstructedPlasticity``/``RewardPlasticity``
        (entries matched by CSR position; positions not in the file keep FlyWire values)."""
        pos = self.brain.plastic_pos
        vals = cp.asnumpy(self.w0).copy()
        where = np.searchsorted(pos, st["pos"])
        ok = (where < len(pos)) & (pos[np.minimum(where, len(pos) - 1)] == st["pos"])
        vals[where[ok]] = st["data"][ok]
        slots = np.arange(self.B) if flies is None else np.flatnonzero(np.isin(self.fly_of_slot, flies))
        self.brain.pw[cp.asarray(slots)] = cp.asarray(vals)
        if "bias" in st:   # intrinsic excitability learned alongside (BatchInstruct eta_bias)
            j = self.brain.slots(st["bias_idx"])
            base = self.brain.bias_ext[:, j]
            if getattr(self, "ib", None) is not None:
                base = base - self.ib[:, self._dn_pos(st["bias_idx"])]
                self.ib[cp.asarray(slots)[:, None], cp.asarray(self._dn_pos(st["bias_idx"]))[None]] = cp.asarray(st["bias"], cp.float32)
            self.brain.bias_ext[cp.asarray(slots)[:, None], j[None]] = base[cp.asarray(slots)] + cp.asarray(st["bias"], cp.float32)

    def drift(self) -> np.ndarray:
        """Mean relative change from FlyWire, per slot."""
        return cp.asnumpy(cp.mean(cp.abs(self.brain.pw - self.w0) / self.wmax, 1))


class BatchInstruct(_Plastic):
    """Delta rule towards a teacher's target rates (``instruct.InstructedPlasticity``)."""

    def __init__(self, brain, conn: Connectome, params: InstructParams | None = None, fly_of_slot=None,
                 eta_bias: float = 0.0, bias_limit: float = 8.0):
        """``eta_bias``: intrinsic plasticity. Each instructed DN's own excitability (its steady
        bias current, mV) also follows the error: ``bias_j += eta_bias * (target_j - rate_j)``,
        within +-``bias_limit`` mV of where it started. Removes resting offsets the synapses
        struggle with (e.g. the connectome's left/right DNa02 asymmetry)."""
        self.p = p = params or InstructParams()
        super().__init__(brain, p.cap, fly_of_slot)
        self.eta_bias, self.bias_limit = eta_bias, bias_limit
        self.ib = None
        self.names = list(GROUPS)
        group_of = np.full(conn.n, -1, np.int32)
        for k, g in enumerate(self.names):
            t, s = GROUPS[g]
            group_of[conn.find(t, s)] = k
        pg = group_of[brain.pl_post]
        self.mask = cp.asarray(pg >= 0)                 # only synapses onto instructed groups learn
        self.post_group = cp.asarray(np.maximum(pg, 0))
        self.upre, pre_inv = np.unique(brain.pl_pre, return_inverse=True)
        self.pre_k = cp.asarray(pre_inv)
        self.upost, post_inv = np.unique(brain.pl_post, return_inverse=True)
        self.post_k = cp.asarray(post_inv)
        self.d_upre, self.d_upost = cp.asarray(self.upre), cp.asarray(self.upost)
        self.trace = cp.zeros((self.B, len(self.upre)), cp.float32)
        self.rate = cp.zeros((self.B, len(self.upost)), cp.float32)
        # per group: which post neurons (for the error report)
        self.group_members = [np.flatnonzero(group_of[self.upost] == k) for k in range(len(self.names))]
        # the instructed DNs themselves (for intrinsic plasticity)
        self.dn = np.flatnonzero(group_of >= 0)
        self.dn_group = cp.asarray(group_of[self.dn])
        assert np.isin(self.dn, self.upost).all(), "every instructed DN needs plastic input synapses"
        self.dn_rate_k = cp.asarray(np.searchsorted(self.upost, self.dn))
        self.dn_slots = brain.slots(self.dn)
        if eta_bias > 0:
            self.ib = cp.zeros((self.B, len(self.dn)), cp.float32)

    def _dn_pos(self, idx):
        assert np.isin(idx, self.dn).all(), "saved intrinsic biases are for other neurons (different GROUPS?)"
        return np.searchsorted(self.dn, idx)

    def reset(self, slots):
        s = cp.asarray(np.atleast_1d(slots))
        self.trace[s] = 0
        self.rate[s] = 0

    def step(self, counts: cp.ndarray, targets: np.ndarray | None, learn: np.ndarray, window_ms: float,
             eta=None):
        """``targets``: (slots, len(GROUPS)) target rates in ``GROUPS`` order; ``learn``: bool per slot;
        ``eta``: learning rate, scalar or per slot ((slots, 1) on the GPU); default ``p.eta``.
        Returns the mean |target - rate| (Hz) per slot (0 where not learning)."""
        p = self.p
        a_pre = 1 - np.exp(-window_ms / p.tau_pre_ms)
        a_rate = 1 - np.exp(-window_ms / p.tau_rate_ms)
        hz = 1000.0 / window_ms
        c_pre = counts[:, self.d_upre].astype(cp.float32)
        self.trace += a_pre * (c_pre * hz * (p.tau_pre_ms / 1000.0) - self.trace)
        c_post = counts[:, self.d_upost].astype(cp.float32)
        self.rate += a_rate * (c_post * hz - self.rate)
        if targets is None or not learn.any():
            return np.zeros(self.B)
        tgt = cp.asarray(targets, cp.float32)
        self._learn(tgt, learn, p.eta if eta is None else eta)
        if self.ib is not None:
            err = tgt[:, self.dn_group] - self.rate[:, self.dn_rate_k]
            act = cp.asarray(learn.astype(np.float32))
            onehot = cp.zeros((self.n_flies, self.B), cp.float32)
            onehot[self._fos, cp.arange(self.B)] = act
            per_fly = (onehot / cp.maximum(onehot.sum(1, keepdims=True), 1.0)) @ err
            step = self.eta_bias * per_fly[self._fos] * (cp.asarray(eta) / p.eta if eta is not None else 1.0)
            new = cp.clip(self.ib + step, -self.bias_limit, self.bias_limit)
            self.brain.bias_ext[:, self.dn_slots] += new - self.ib
            self.ib = new
        rate = cp.asnumpy(self.rate)
        out = np.zeros(self.B)
        for k, mem in enumerate(self.group_members):
            if len(mem):
                out += np.abs(targets[:, k] - rate[:, mem].mean(1))
        return np.where(learn, out / len(self.names), 0.0)


    def _learn(self, tgt, learn, eta):
        err = tgt[:, self.post_group] - self.rate[:, self.post_k]
        self.apply(eta * self.trace[:, self.pre_k] * err * self.mask, learn)


def deep_positions(conn: Connectome, dn_pos: np.ndarray) -> np.ndarray:
    """CSR positions of every synapse onto the motor DNs' presynaptic partners ("L1")."""
    w = sp.csr_matrix(conn.weights, dtype=np.float32)
    w.sort_indices()
    rows = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
    is_l1 = np.zeros(conn.n, bool)
    is_l1[np.unique(rows[dn_pos])] = True
    return np.flatnonzero(is_l1[w.indices])


class BatchInstructDeep(BatchInstruct):
    """Instructed learning one layer deeper: synapses onto the DNs' presynaptic partners (L1).

    The DN synapses learn exactly as in ``BatchInstruct``. Each L1 neuron i also gets an error
    signal from the motor DNs it synapses onto: ``delta_i = sum_j n_ij * err_j``, where n_ij is
    its current signed synapse count onto DN j (weights in units of one synapse) and err_j that
    DN's target - rate (Hz). A retrograde "your targets should fire more / less" signal, i.e.
    the delta rule passed back through the fly's own synapses (one step of backpropagation).
    Synapses onto L1 then learn ``dw_ki = eta_deep * trace_k * delta_i`` (Dale's law and caps as
    everywhere). Needs a brain whose plastic positions include ``deep_positions``.
    """

    def __init__(self, brain, conn: Connectome, params: InstructParams | None = None, fly_of_slot=None,
                 eta_deep: float = 1e-6, deep_scale=None, eta_bias: float = 0.0):
        """``deep_scale``: optional per-slot factor on ``eta_deep`` (a learning-rate sweep)."""
        super().__init__(brain, conn, params, fly_of_slot, eta_bias=eta_bias)
        self.eta_deep = eta_deep
        self.deep_scale = np.ones(self.B) if deep_scale is None else np.asarray(deep_scale, float)
        dn_syn = cp.asnumpy(self.mask)
        post = brain.pl_post
        is_dn = np.zeros(conn.n, bool)
        for g in self.names:
            is_dn[conn.find(*GROUPS[g])] = True
        self.l1 = np.unique(brain.pl_pre[dn_syn])
        l1_index = np.full(conn.n, -1, np.int64)
        l1_index[self.l1] = np.arange(len(self.l1))
        self.dn_sel = cp.asarray(np.flatnonzero(dn_syn))                 # L1 -> DN synapses
        self.dn_pre_l1 = cp.asarray(l1_index[brain.pl_pre[dn_syn]])
        deep = (l1_index[post] >= 0) & ~is_dn[post]
        self.deep_sel = cp.asarray(np.flatnonzero(deep))                  # -> L1 synapses
        self.deep_post_l1 = cp.asarray(l1_index[post[deep]])
        self.deep_pre_k = self.pre_k[self.deep_sel]
        self.n_deep = int(deep.sum())
        assert self.n_flies <= 16
        self.k_sel = self.deep_sel.astype(cp.int32)
        self.k_pre = self.deep_pre_k.astype(cp.int32)
        self.k_post = self.deep_post_l1.astype(cp.int32)
        self.k_fos = self._fos.astype(cp.int32)
        self.sign = self.sign.astype(cp.float32)
        self.wmax = self.wmax.astype(cp.float32)
        self._kernel = cp.RawModule(code=_DEEP_SRC).get_function("deep_update")

    def _learn(self, tgt, learn, eta):
        import cupyx

        sel = self.dn_sel
        err = tgt[:, self.post_group[sel]] - self.rate[:, self.post_k[sel]]      # (slots, DN synapses)
        # retrograde error for each L1 neuron, through its current synapses onto the DNs
        w = self.brain.pw[:, sel] / self.brain.p.w_syn
        self.apply(eta * self.trace[:, self.pre_k[sel]] * err, learn, sel)
        delta = cp.zeros((self.B, len(self.l1)), cp.float32)
        cupyx.scatter_add(delta, (slice(None), self.dn_pre_l1), w * err)
        # fused kernel: coef = per-slot rate / number of learning slots of its fly
        eta_b = np.broadcast_to(cp.asnumpy(cp.asarray(eta, cp.float32)).ravel(), (self.B,)) * (self.eta_deep / self.p.eta) * self.deep_scale
        n_act = np.bincount(self.fly_of_slot, weights=learn, minlength=self.n_flies)
        coef = np.where(learn, eta_b / np.maximum(n_act[self.fly_of_slot], 1), 0.0).astype(np.float32)
        n = np.int32(self.n_deep)
        self._kernel(((self.n_deep + 255) // 256,), (256,), (
            n, np.int32(self.B), self.k_sel, self.k_pre, self.k_post, self.trace, np.int32(self.trace.shape[1]),
            delta, np.int32(len(self.l1)), self.brain.pw, np.int32(self.brain.pw.shape[1]),
            self.sign, self.wmax, cp.asarray(coef), self.k_fos, np.int32(self.n_flies)))


class BatchReward(_Plastic):
    """Exploratory-Hebbian reward learning (``learning.RewardPlasticity``)."""

    def __init__(self, brain, conn: Connectome, params: LearnParams | None = None, fly_of_slot=None, seed=0):
        self.p = p = params or LearnParams()
        super().__init__(brain, p.cap, fly_of_slot)
        self.rng = np.random.default_rng(seed)
        self.targets = np.unique(np.concatenate([conn.find(t) for t in p.targets]))
        self.upre, pre_inv = np.unique(brain.pl_pre, return_inverse=True)
        self.upost, post_inv = np.unique(brain.pl_post, return_inverse=True)
        self.pre_k, self.post_k = cp.asarray(pre_inv), cp.asarray(post_inv)
        self.d_upre, self.d_upost = cp.asarray(self.upre), cp.asarray(self.upost)
        self.elig = cp.zeros((self.B, len(self.brain.plastic_pos)), cp.float32)
        self.post_mean = cp.zeros((self.B, len(self.upost)), cp.float32)
        self.r_mean = np.zeros(self.B, np.float32)
        self.noise = np.zeros((self.B, len(self.targets)))

    def reset(self, slots):
        s = np.atleast_1d(slots)
        self.elig[cp.asarray(s)] = 0
        self.noise[s] = 0

    def exploration(self, window_ms: float) -> np.ndarray:
        """Extra Poisson rates (slots, len(self.targets)) for the target neurons."""
        a = window_ms / self.p.noise_tau_ms
        self.noise += -a * self.noise + np.sqrt(2 * a) * self.rng.standard_normal(self.noise.shape)
        return np.maximum(self.p.noise_hz * self.noise, 0.0)

    def step(self, counts: cp.ndarray, reward: np.ndarray, learn: np.ndarray, window_ms: float):
        p = self.p
        ae = np.exp(-window_ms / p.tau_elig_ms)
        ap = 1 - np.exp(-window_ms / p.tau_post_ms)
        ar = 1 - np.exp(-window_ms / p.tau_reward_ms)
        c_post = counts[:, self.d_upost].astype(cp.float32)
        dev = (c_post - self.post_mean)[:, self.post_k]
        self.elig = ae * self.elig + counts[:, self.d_upre].astype(cp.float32)[:, self.pre_k] * dev
        self.post_mean += ap * (c_post - self.post_mean)
        adv = reward - self.r_mean
        self.r_mean += ar * adv
        if learn.any():
            self.apply(p.eta * cp.asarray(adv.astype(np.float32))[:, None] * self.elig, learn)
        return adv
