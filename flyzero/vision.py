"""Motion vision front end: game frames -> T4/T5 elementary motion detectors.

Why not just drive the photoreceptors? Because the first stages of fly vision
(photoreceptors, lamina L1-L3, most medulla cells) are graded-potential,
non-spiking neurons, and the photoreceptor synapse is an inhibitory histamine
synapse. A spiking LIF model built from synapse counts cannot carry that signal:
driving all ~10,600 photoreceptors of the FlyWire brain activates ~2% of the
optic lobe and not a single descending neuron.

So, like most models of fly vision, we compute the front end algorithmically,
up to the first direction-selective neurons, and hand the connectome over from
there. T4 (ON edges) and T5 (OFF edges) come in four subtypes that each prefer
one cardinal direction (Maisak et al. 2013):

    a: front-to-back   b: back-to-front   c: upward   d: downward

Each T4/T5 neuron gets a place in the visual field from its position in the
brain (see ``eyes.retinotopy``). Its Poisson rate is the output of a
Hassenstein-Reichardt correlator for its polarity and preferred direction at
that place. Everything downstream (lobula plate tangential cells, LPLC/LC
projection neurons, central brain, descending neurons) is the connectome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import Connectome
from .eyes import retinotopy


@dataclass
class MotionParams:
    max_rate: float = 150.0   # Hz
    gain: float = 1000.0      # correlator output -> fraction of max rate
    cell: int = 4             # pixels per sampling point (~ommatidial spacing)
    tau_hp: float = 4.0       # frames, high-pass of the ON/OFF split
    tau_delay: float = 2.0    # frames, low-pass "delay" arm of the correlator
    shift: int = 1            # sampling points between correlator arms
    overlap: float = 0.1      # binocular overlap (fraction of screen width)
    y_min: float = 0.0        # ignore the top of the screen (fraction of height), e.g. the sky


# subtype -> direction in eye coordinates ("front" = towards the midline)
SUBTYPES = {"a": "front_to_back", "b": "back_to_front", "c": "up", "d": "down"}


class MotionEye:
    """Same interface as ``eyes.CompoundEye``: ``idx``, ``see(frame) -> rates``."""

    def __init__(self, conn: Connectome, frame_shape: tuple[int, int], params: MotionParams | None = None):
        self.p = p = params or MotionParams()
        self.h, self.w = frame_shape
        self.gh, self.gw = self.h // p.cell, self.w // p.cell

        idx, xs, ys, sides, chan = [], [], [], [], []
        for kind, pol in (("T4", "on"), ("T5", "off")):
            for sub, direction in SUBTYPES.items():
                cells = conn.find(kind + sub)
                if len(cells) == 0:
                    continue
                px, py, side, keep = retinotopy(conn, cells, self.w, self.h, p.overlap)
                cells, px, py, side = cells[keep], px[keep], py[keep], side[keep]
                idx.append(cells); xs.append(px); ys.append(py); sides.append(side)
                chan.append(np.array([f"{pol}:{direction}"] * len(cells)))
        if not idx:
            raise ValueError("connectome has no T4/T5 neurons")
        self.idx = np.concatenate(idx)
        self.px, self.py = np.concatenate(xs), np.concatenate(ys)
        self.side = np.concatenate(sides)
        chan = np.concatenate(chan)
        # screen direction per neuron: front-to-back is leftwards for the left eye
        self._map = {}
        for c in np.unique(chan):
            pol, direction = c.split(":")
            m = chan == c
            for s in ("left", "right"):
                sel = np.flatnonzero(m & (self.side == s))
                if direction == "front_to_back":
                    screen = "left" if s == "left" else "right"
                elif direction == "back_to_front":
                    screen = "right" if s == "left" else "left"
                else:
                    screen = direction
                self._map.setdefault((pol, screen), []).append(sel)
        self._map = {k: np.concatenate(v) for k, v in self._map.items()}
        self.gx = np.clip(self.px // p.cell, 0, self.gw - 1)
        self.gy = np.clip(self.py // p.cell, 0, self.gh - 1)

        self.lp_hp = None
        self.delayed = None
        self.last_rates = np.zeros(len(self.idx), np.float32)

    def describe(self) -> str:
        return (f"{len(self.idx)} T4/T5 motion detectors "
                f"({(self.side == 'left').sum()} left, {(self.side == 'right').sum()} right)")

    def _grid(self, frame: np.ndarray) -> np.ndarray:
        c = self.p.cell
        lum = frame[: self.gh * c, : self.gw * c].astype(np.float32).mean(2) / 255.0
        return lum.reshape(self.gh, c, self.gw, c).mean((1, 3))

    def see(self, frame: np.ndarray) -> np.ndarray:
        p = self.p
        lum = self._grid(frame)
        if self.lp_hp is None:
            self.lp_hp = lum.copy()
            self.delayed = {"on": np.zeros_like(lum), "off": np.zeros_like(lum)}
        a_hp = 1.0 / p.tau_hp
        a_d = 1.0 / p.tau_delay
        self.lp_hp += a_hp * (lum - self.lp_hp)
        hp = lum - self.lp_hp
        motion = {}
        k = p.shift
        for pol, sig in (("on", np.maximum(hp, 0)), ("off", np.maximum(-hp, 0))):
            d = self.delayed[pol]
            # Hassenstein-Reichardt: delayed(A) * B - A * delayed(B), B = A shifted
            right = np.zeros_like(sig); down = np.zeros_like(sig)
            right[:, :-k] = d[:, :-k] * sig[:, k:] - sig[:, :-k] * d[:, k:]
            down[:-k] = d[:-k] * sig[k:] - sig[:-k] * d[k:]
            motion[(pol, "right")] = np.maximum(right, 0)
            motion[(pol, "left")] = np.maximum(-right, 0)
            motion[(pol, "down")] = np.maximum(down, 0)
            motion[(pol, "up")] = np.maximum(-down, 0)
            d += a_d * (sig - d)
        rates = np.zeros(len(self.idx), np.float32)
        for key, sel in self._map.items():
            rates[sel] = motion[key][self.gy[sel], self.gx[sel]]
        rates[self.py < p.y_min * self.h] = 0.0
        self.last_rates = (p.max_rate * np.clip(p.gain * rates, 0, 1)).astype(np.float32)
        return self.last_rates
