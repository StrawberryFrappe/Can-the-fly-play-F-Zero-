"""Descending neurons -> SNES controller.

Descending neurons (DNs) are the brain's output to the ventral nerve cord, so
they are the fly's "hands on the controller". The mapping follows the known
function of each DN type in the walking/flying fly:

=============  =====================================================  ==============
DN type        what it does in the fly                                F-Zero button
=============  =====================================================  ==============
DNa02 (L/R)    walking: unilateral activity turns towards that side     D-pad LEFT/RIGHT
               (Rayshubskiy et al. 2020, Yang et al. 2023)
DNg02 (L/R)    flight: wingbeat amplitude (Namiki et al. 2022). A       D-pad LEFT/RIGHT
               stronger right wing turns the fly left, so right>left
               DNg02 activity steers left (our inference, not measured)
DNa01 (L/R)    larger, sharper heading changes (saccade-like)           L / R (hard lean)
DNp09 (P9)     drives forward walking (Bidaye et al. 2020)             B (accelerate)
MDN            "moonwalker": drives backward walking                   Y (brake)
               (Bidaye et al. 2014)
DNp01 (GF)     giant fiber, fires the escape jump (von Reyn 2014)       A (super jet)
=============  =====================================================  ==============

Firing rates are smoothed with an exponential filter and compared with
thresholds; steering uses the left-right difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .connectome import Connectome

BUTTONS = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]


@dataclass
class MotorParams:
    tau_ms: float = 80.0          # smoothing of DN rates
    steer_threshold: float = 10.0  # Hz, |left - right| DNa02 rate to push the D-pad
    steer_bias: float = 0.0       # Hz added to (left - right), calibrates a resting asymmetry
    lean_threshold: float = 10.0  # Hz, DNa01 difference to lean
    accel_threshold: float = 5.0  # Hz, DNp09 rate to hold the throttle
    brake_threshold: float = 15.0
    boost_threshold: float = 1.0
    groups: dict = field(default_factory=lambda: {
        "steer_left": ("DNa02", "left"),
        "steer_right": ("DNa02", "right"),
        "lean_left": ("DNa01", "left"),
        "lean_right": ("DNa01", "right"),
        "wing_left": ("DNg02*", "left"),
        "wing_right": ("DNg02*", "right"),
        "accelerate": ("DNp09", None),
        "brake": ("MDN", None),
        "boost": ("DNp01", None),
    })


class MotorReadout:
    def __init__(self, conn: Connectome, params: MotorParams | None = None):
        self.p = params or MotorParams()
        self.groups = {name: conn.find(t, side) for name, (t, side) in self.p.groups.items()}
        self.rates = {name: 0.0 for name in self.groups}
        self.buttons = {b: False for b in BUTTONS}

    def describe(self) -> str:
        return "\n".join(
            f"  {name:<12} {t:<6} {side or 'both':<6} {len(self.groups[name]):>3} neurons"
            for name, (t, side) in self.p.groups.items()
        )

    def update(self, counts: np.ndarray, window_ms: float) -> dict[str, bool]:
        a = 1.0 - np.exp(-window_ms / self.p.tau_ms)
        for name, idx in self.groups.items():
            inst = counts[idx].mean() * 1000.0 / window_ms if len(idx) else 0.0
            self.rates[name] += a * (inst - self.rates[name])
        r, p = self.rates, self.p
        steer = (r["steer_left"] - r["steer_right"]) + (r["wing_right"] - r["wing_left"]) + p.steer_bias
        lean = r["lean_left"] - r["lean_right"]
        b = {k: False for k in BUTTONS}
        b["LEFT"] = steer > p.steer_threshold
        b["RIGHT"] = steer < -p.steer_threshold
        b["L"] = lean > p.lean_threshold
        b["R"] = lean < -p.lean_threshold
        b["B"] = r["accelerate"] > p.accel_threshold
        b["Y"] = r["brake"] > p.brake_threshold
        b["A"] = r["boost"] > p.boost_threshold
        self.buttons = b = {k: bool(v) for k, v in b.items()}
        return b
