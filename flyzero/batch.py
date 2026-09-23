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
from .instruct import GROUPS, InstructParams
from .learning import LearnParams
from .motor import BUTTONS, MotorParams

READOUT = ["steer_left", "steer_right", "lean_left", "lean_right", "wing_left", "wing_right",
           "accelerate", "brake", "boost"]


def plastic_positions(conn: Connectome, types=("DNa02", "DNa01", "DNg02*", "DNp09", "MDN")) -> np.ndarray:
    """CSR positions (``Brain.data`` layout) of every FlyWire synapse onto the given cell types."""
    targets = np.unique(np.concatenate([conn.find(t) for t in types]))
    is_t = np.zeros(conn.n, bool)
    is_t[targets] = True
    w = sp.csr_matrix(conn.weights, dtype=np.float32)
    w.sort_indices()
    return np.flatnonzero(is_t[w.indices])


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

    def apply(self, dw: cp.ndarray, active: np.ndarray):
        """Average ``dw`` (slots, n_pl) over each fly's active slots; update every slot of the fly."""
        act = cp.asarray(active.astype(np.float32))
        onehot = cp.zeros((self.n_flies, self.B), cp.float32)
        onehot[self._fos, cp.arange(self.B)] = act
        n = onehot.sum(1, keepdims=True)
        per_fly = (onehot @ dw) / cp.maximum(n, 1.0)
        w = self.brain.pw + per_fly[self._fos]
        self.brain.pw[...] = self.sign * cp.clip(self.sign * w, 0.0, self.wmax)

    def state(self, fly: int) -> dict:
        slot = int(np.flatnonzero(self.fly_of_slot == fly)[0])
        return {"pos": self.brain.plastic_pos, "data": cp.asnumpy(self.brain.pw[slot]),
                "w0": cp.asnumpy(self.w0)}

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

    def drift(self) -> np.ndarray:
        """Mean relative change from FlyWire, per slot."""
        return cp.asnumpy(cp.mean(cp.abs(self.brain.pw - self.w0) / self.wmax, 1))


class BatchInstruct(_Plastic):
    """Delta rule towards a teacher's target rates (``instruct.InstructedPlasticity``)."""

    def __init__(self, brain, conn: Connectome, params: InstructParams | None = None, fly_of_slot=None):
        self.p = p = params or InstructParams()
        super().__init__(brain, p.cap, fly_of_slot)
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

    def reset(self, slots):
        s = cp.asarray(np.atleast_1d(slots))
        self.trace[s] = 0
        self.rate[s] = 0

    def step(self, counts: cp.ndarray, targets: np.ndarray | None, learn: np.ndarray, window_ms: float):
        """``targets``: (slots, 8) target rates in ``GROUPS`` order; ``learn``: bool per slot.
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
        err = tgt[:, self.post_group] - self.rate[:, self.post_k]
        dw = p.eta * self.trace[:, self.pre_k] * err * self.mask
        self.apply(dw, learn)
        rate = cp.asnumpy(self.rate)
        out = np.zeros(self.B)
        for k, mem in enumerate(self.group_members):
            if len(mem):
                out += np.abs(targets[:, k] - rate[:, mem].mean(1))
        return np.where(learn, out / len(self.names), 0.0)


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
