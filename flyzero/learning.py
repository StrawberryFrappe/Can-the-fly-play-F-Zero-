"""Reward learning at the fly's motor output: let the brain adapt to F-Zero.

Only synapses that already exist in FlyWire and end on a few descending neurons (by default
DNa02 and DNa01 for steering, DNp09 for walking/gas, MDN for backing up/brake) are plastic.
Each keeps the sign of its presynaptic neurotransmitter (Dale's law) and stays within
[0, cap * original strength]. Every other synapse in the brain is fixed.

The rule is the exploratory-Hebbian ("EH") rule of Hoerzer, Legenstein & Maass (2014,
Cerebral Cortex 24:677), a reward-modulated Hebbian rule of the node-perturbation family used
to model how songbirds learn to sing:

    E_ij <- lambda * E_ij + pre_i(t) * (post_j(t) - <post_j>)        eligibility trace
    w_ij <- w_ij + eta * (R(t) - <R>) * E_ij                           reward gates the change

The descending neurons' own spiking variability, plus a little exploratory noise, does the
exploring. Changes that happened to be followed by more reward than usual are kept.

Reward (per frame; the user's spec: going fast is good, crashing is bad, reversing is horrible):
    + 20 per track segment gained (fast = more segments per second)
    - 60 per track segment lost (driving the wrong way)
    - 100 per full energy bar lost (a guard-beam hit costs ~5)
    + 0.2 x speed / top speed
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .connectome import Connectome

TOP_SPEED = 2100.0


@dataclass
class LearnParams:
    eta: float = 2e-4             # mV per unit of (reward x eligibility)
    tau_elig_ms: float = 500.0    # how long a synapse stays "eligible" after pre/post coincidence
    tau_post_ms: float = 500.0    # running mean of each target's spike count
    tau_reward_ms: float = 1000.0  # running mean reward (the expectation)
    cap: float = 3.0              # |w| <= cap * max(|w0|, one synapse)
    noise_hz: float = 20.0        # exploratory drive to the targets (OU-modulated Poisson)
    noise_tau_ms: float = 300.0
    targets: tuple = ("DNa02", "DNa01", "DNp09", "MDN")
    # reward weights
    w_speed: float = 0.2      # small: speed only counts when it moves you along the track
    w_progress: float = 20.0  # per track segment (59 per lap)
    w_reverse: float = 60.0   # going the wrong way is horrible
    w_crash: float = 100.0    # per full energy bar lost (a wall hit costs ~5)


@dataclass
class Reward:
    """Per-frame reward from telemetry."""
    p: LearnParams
    last_seg: int | None = None
    last_energy: float | None = None
    parts: dict = field(default_factory=dict)

    def __call__(self, info: dict) -> float:
        seg, energy = info["segment"], info["energy"]
        d = 0
        if self.last_seg is not None:
            d = (seg - self.last_seg + 29) % 59 - 29
        lost = max(0.0, (self.last_energy if self.last_energy is not None else energy) - energy)
        self.last_seg, self.last_energy = seg, energy
        speed = min(info["speed"], TOP_SPEED) / TOP_SPEED if info["speed"] != 512 or energy > 0 else 0.0
        self.parts = {"speed": speed, "fwd": max(d, 0), "rev": max(-d, 0), "crash": lost}
        return (self.p.w_speed * speed + self.p.w_progress * max(d, 0)
                - self.p.w_reverse * max(-d, 0) - self.p.w_crash * lost)


class RewardPlasticity:
    def __init__(self, brain, conn: Connectome, params: LearnParams | None = None, seed: int = 0):
        self.p = p = params or LearnParams()
        self.brain = brain
        self.rng = np.random.default_rng(seed)
        targets = np.unique(np.concatenate([conn.find(t) for t in p.targets]))
        self.targets = targets
        is_target = np.zeros(brain.n, bool); is_target[targets] = True
        self.pos = np.flatnonzero(is_target[brain.indices])            # plastic entries in CSR data
        rows = np.repeat(np.arange(brain.n), np.diff(brain.indptr))
        self.pre = rows[self.pos]
        self.post = brain.indices[self.pos]
        self.w0 = brain.data[self.pos].copy()
        self.sign = np.sign(self.w0)
        one = brain.p.w_syn
        self.wmax = p.cap * np.maximum(np.abs(self.w0), one)
        self.elig = np.zeros(len(self.pos), np.float32)
        self.post_mean = np.zeros(brain.n, np.float32)
        self.r_mean = 0.0
        self.noise = np.zeros(len(targets))

    def exploration(self, window_ms: float) -> tuple[np.ndarray, np.ndarray]:
        """Extra Poisson drive to the target neurons (indices, Hz) for the next window."""
        a = window_ms / self.p.noise_tau_ms
        self.noise += -a * self.noise + np.sqrt(2 * a) * self.rng.standard_normal(len(self.noise))
        return self.targets, np.maximum(self.p.noise_hz * self.noise, 0.0)

    def step(self, counts: np.ndarray, reward: float, window_ms: float):
        p = self.p
        ae = np.exp(-window_ms / p.tau_elig_ms)
        ap = 1 - np.exp(-window_ms / p.tau_post_ms)
        ar = 1 - np.exp(-window_ms / p.tau_reward_ms)
        c = counts.astype(np.float32)
        dev = c[self.post] - self.post_mean[self.post]
        self.elig = ae * self.elig + c[self.pre] * dev
        self.post_mean[self.targets] += ap * (c[self.targets] - self.post_mean[self.targets])
        adv = reward - self.r_mean
        self.r_mean += ar * adv
        w = self.brain.data[self.pos] + p.eta * adv * self.elig
        # Dale's law + bounds: a synapse keeps its sign and stays within its cap
        w = self.sign * np.clip(self.sign * w, 0.0, self.wmax)
        self.brain.data[self.pos] = w.astype(np.float32)
        return adv

    def state(self) -> dict:
        return {"pos": self.pos, "data": self.brain.data[self.pos].copy(), "w0": self.w0}

    def load(self, st: dict):
        assert np.array_equal(st["pos"], self.pos)
        self.brain.data[self.pos] = st["data"]

    def drift(self) -> float:
        """Mean relative change of plastic weights from FlyWire."""
        return float(np.mean(np.abs(self.brain.data[self.pos] - self.w0) / self.wmax))
