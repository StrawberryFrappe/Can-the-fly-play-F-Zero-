"""Instructed learning: the fly learns to drive from a teacher's button presses.

Same plastic synapses as ``learning.py`` (the existing FlyWire synapses onto the motor neurons
that press buttons), but instead of a scalar reward, each motor neuron gets an *instructive
signal*: the firing rate it should have had, given what the teacher pressed. This is the delta
rule (Widrow-Hoff), the classic model of error-driven motor learning, e.g. climbing-fibre
signals teaching the cerebellum:

    trace_i   <- low-pass of presynaptic spikes
    rate_j    <- low-pass of motor neuron j's spikes
    w_ij      <- w_ij + eta * trace_i * (target_j - rate_j)

Dale's law and the per-synapse caps from ``learning.py`` apply: no synapse changes sign or is
created. Everything upstream of the motor neurons is untouched.

Targets follow the button mapping in ``motor.py``:

    LEFT  -> DNa02 left high, DNa02 right low, DNg02 right high, DNg02 left low (and vice versa)
    L / R -> DNa01 left / right high
    B     -> DNp09 high (gas)         Y -> MDN high (brake)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import Connectome


@dataclass
class InstructParams:
    eta: float = 2e-4          # mV per (spike trace x Hz of error)
    tau_pre_ms: float = 50.0   # presynaptic trace
    tau_rate_ms: float = 80.0  # motor neuron rate estimate (= motor readout smoothing)
    high_hz: float = 80.0
    low_hz: float = 0.0
    rest_hz: float = 20.0      # steering neurons when going straight (symmetric)
    cap: float = 3.0


GROUPS = {  # name: (cell type, side)
    "a02L": ("DNa02", "left"), "a02R": ("DNa02", "right"),
    "g02L": ("DNg02*", "left"), "g02R": ("DNg02*", "right"),
    "a01L": ("DNa01", "left"), "a01R": ("DNa01", "right"),
    "gas": ("DNp09", None), "brake": ("MDN", None),
}


def targets_from_buttons(b: dict, p: InstructParams) -> dict:
    hi, lo, rest = p.high_hz, p.low_hz, p.rest_hz
    left, right = b.get("LEFT", False), b.get("RIGHT", False)
    t = {}
    if left and not right:
        t.update(a02L=hi, a02R=lo, g02R=hi, g02L=lo)
    elif right and not left:
        t.update(a02L=lo, a02R=hi, g02R=lo, g02L=hi)
    else:
        t.update(a02L=rest, a02R=rest, g02R=rest, g02L=rest)
    t["a01L"] = hi if b.get("L") else lo
    t["a01R"] = hi if b.get("R") else lo
    t["gas"] = hi if b.get("B") else lo
    t["brake"] = hi if b.get("Y") else lo
    return t


class InstructedPlasticity:
    def __init__(self, brain, conn: Connectome, params: InstructParams | None = None):
        self.p = p = params or InstructParams()
        self.brain = brain
        self.group_idx = {g: conn.find(t, s) for g, (t, s) in GROUPS.items()}
        self.group_of = np.full(brain.n, -1, np.int32)
        names = list(GROUPS)
        for k, g in enumerate(names):
            self.group_of[self.group_idx[g]] = k
        self.names = names
        post_all = brain.indices
        self.pos = np.flatnonzero(self.group_of[post_all] >= 0)
        rows = np.repeat(np.arange(brain.n), np.diff(brain.indptr))
        self.pre = rows[self.pos]
        self.post = post_all[self.pos]
        self.post_group = self.group_of[self.post]
        self.w0 = brain.data[self.pos].copy()
        self.sign = np.sign(self.w0)
        self.wmax = p.cap * np.maximum(np.abs(self.w0), brain.p.w_syn)
        self.trace = np.zeros(brain.n, np.float32)
        self.rate = np.zeros(brain.n, np.float32)
        self.targets = np.concatenate(list(self.group_idx.values()))

    def step(self, counts: np.ndarray, teacher: dict | None, window_ms: float, learn: bool = True):
        p = self.p
        a_pre = 1 - np.exp(-window_ms / p.tau_pre_ms)
        a_rate = 1 - np.exp(-window_ms / p.tau_rate_ms)
        c = counts.astype(np.float32)
        self.trace += a_pre * (c * 1000.0 / window_ms * (p.tau_pre_ms / 1000.0) - self.trace)
        t = self.targets
        self.rate[t] += a_rate * (c[t] * 1000.0 / window_ms - self.rate[t])
        if not learn or teacher is None:
            return 0.0
        tgt = targets_from_buttons(teacher, p)
        tgt_by_group = np.array([tgt[n] for n in self.names], np.float32)
        err = tgt_by_group[self.post_group] - self.rate[self.post]
        w = self.brain.data[self.pos] + p.eta * self.trace[self.pre] * err
        w = self.sign * np.clip(self.sign * w, 0.0, self.wmax)
        self.brain.data[self.pos] = w.astype(np.float32)
        return float(np.mean(np.abs(tgt_by_group - np.array(
            [self.rate[self.group_idx[n]].mean() if len(self.group_idx[n]) else 0 for n in self.names]))))

    def state(self) -> dict:
        return {"pos": self.pos, "data": self.brain.data[self.pos].copy(), "w0": self.w0}

    def load(self, st: dict):
        assert np.array_equal(st["pos"], self.pos)
        self.brain.data[self.pos] = st["data"]

    def drift(self) -> float:
        return float(np.mean(np.abs(self.brain.data[self.pos] - self.w0) / self.wmax))
