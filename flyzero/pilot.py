"""The pilot: a privileged F-Zero driver that follows the owner's racing line, read from RAM.

It is **not the fly** and never drives the fly's car. It is the instructor for DAgger-style
lessons: while the fly drives, the pilot says what it would do in the fly's position (a label
for the fly's own learning rule), and it's a reference driver for comparisons.

It reads the car's position and heading from RAM (see docs/fzero-internals.md):

    $7E0B70 u16  x        $7E0B90 u16  y        (map coordinates; Mute City I spans ~300..6200)
    $7E0BE0 u16  heading, 0xC000 per full turn; atan2(dy, dx) = heading / 0xC000 * 2pi - pi/2
    $7E0B20 u16  speed

The racing line is a human lap (x, y and speed per frame). Steering is pure pursuit: aim at the
point on the line a speed-dependent distance ahead; steer (and lean, for sharp errors) towards it.
Gas is held unless the car is much faster than the human was at that point of the line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RAM_X, RAM_Y, RAM_HEADING, RAM_SPEED = 0x0B70, 0x0B90, 0x0BE0, 0x0B20
TURN = 0xC000


def u16(ram, a):
    return int(ram[a]) | int(ram[a + 1]) << 8


def pose(ram) -> tuple[float, float, float, float]:
    """x, y, heading (radians, same frame as atan2(dy, dx)), speed."""
    h = u16(ram, RAM_HEADING) / TURN * 2 * np.pi - np.pi / 2
    return float(u16(ram, RAM_X)), float(u16(ram, RAM_Y)), h, float(u16(ram, RAM_SPEED))


def racing_line(ram_frames: np.ndarray, lap_addr: int = 0x0F53, lap: int | None = None, spacing: float = 16.0):
    """(points (N, 2), speed (N,)) from per-frame WRAM of a human race: its fastest full lap,
    resampled every ``spacing`` map units."""
    laps = ram_frames[:, lap_addr].astype(int)
    cross = np.flatnonzero(np.diff(laps) > 0) + 1
    spans = list(zip(cross[:-1], cross[1:]))
    a, b = min(spans, key=lambda s: s[1] - s[0]) if lap is None else spans[lap]
    xy = np.array([[u16(r, RAM_X), u16(r, RAM_Y)] for r in ram_frames[a:b]], float)
    sp = np.array([u16(r, RAM_SPEED) for r in ram_frames[a:b]], float)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    s = np.arange(0, d[-1], spacing)
    pts = np.c_[np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])]
    return pts, np.interp(s, d, sp)


@dataclass
class PilotParams:
    look_base: float = 160.0     # map units ahead at standstill
    look_per_speed: float = 0.06  # extra look-ahead per unit of speed
    kp: float = 3.0              # steering per radian of heading error to the look-ahead point
    kd: float = 35.0             # damping: steering per radian/frame of turning
    over_speed: float = 1.2      # release gas when faster than this x the human here
    window: int = 60             # points searched around the last match (keeps it on its lap)
    min_speed: float = 300.0     # below this the pilot's steering label is neutral
    lean_start: float = 0.5      # steering command from which the pilot also leans, like the owner
    boost: bool = True           # super jet on the bottom straight (x, y below), from lap 2
    boost_x: tuple = (2600.0, 4200.0)
    boost_y_max: float = 600.0


class Pilot:
    """Continuous steering command (-1..1, beyond +-1 also leaning) turned into D-pad taps by
    sigma-delta modulation: the tap duty cycle equals the command, like quick human tapping."""

    def __init__(self, points: np.ndarray, speed: np.ndarray, params: PilotParams | None = None):
        self.p = params or PilotParams()
        self.pts, self.speed = points, speed
        self.n = len(points)
        self.spacing = float(np.median(np.hypot(*np.diff(points, axis=0).T)))
        self.reset()

    def reset(self):
        self.i = None
        self.last_h = None
        self.acc = np.zeros(2)   # sigma-delta accumulators: steer, lean

    def _nearest(self, x, y):
        if self.i is None:
            cand = np.arange(self.n)
        else:
            cand = (self.i + np.arange(-self.p.window, self.p.window + 1)) % self.n
        d = np.hypot(self.pts[cand, 0] - x, self.pts[cand, 1] - y)
        k = int(cand[np.argmin(d)])
        if self.i is not None and d.min() > 400:   # lost (e.g. after a big bounce): global search
            d = np.hypot(self.pts[:, 0] - x, self.pts[:, 1] - y)
            k = int(np.argmin(d))
        self.i = k
        return k

    def intent(self, ram) -> np.ndarray:
        """steer (-1 left .. +1 right), lean (-1 L .. +1 R), gas, brake, boost: like ``instruct.intent``."""
        x, y, h, v = pose(ram)
        k = self._nearest(x, y)
        ahead = int((self.p.look_base + self.p.look_per_speed * v) / self.spacing)
        tx, ty = self.pts[(k + ahead) % self.n]
        err = float(np.angle(np.exp(1j * (np.arctan2(ty - y, tx - x) - h))))
        rate = 0.0 if self.last_h is None else float(np.angle(np.exp(1j * (h - self.last_h))))
        self.last_h = h
        u = self.p.kp * err - self.p.kd * rate
        if v < self.p.min_speed:   # standing (the READY countdown): steering does nothing, so teach none
            u = 0.0
        steer = float(np.clip(u, -1, 1))
        # lean into sharp turns, like the owner (who leans on ~28% of frames)
        lean = float(np.sign(u) * np.clip((abs(u) - self.p.lean_start) / max(1.0 - self.p.lean_start, 1e-3), 0, 1))
        gas = 0.0 if v > self.p.over_speed * max(self.speed[k], 400) else 1.0
        # super jet like the owner: once a lap from lap 2, on the long straight, with energy to spare
        boost = 0.0
        if self.p.boost and int(ram[0x0F53]) >= 1 and abs(err) < 0.1 and \
                self.p.boost_x[0] < x < self.p.boost_x[1] and y < self.p.boost_y_max and \
                (int(ram[0x00C9]) | int(ram[0x00CA]) << 8) > 1024:
            boost = 1.0
        return np.array([steer, lean, gas, 0.0, boost], np.float32)

    def buttons(self, ram, x: np.ndarray | None = None) -> dict:
        s, lean, gas, brake, boost = self.intent(ram) if x is None else x
        taps = []
        for j, u in enumerate((s, lean)):
            self.acc[j] += u
            t = 1 if self.acc[j] >= 0.5 else -1 if self.acc[j] <= -0.5 else 0
            self.acc[j] -= t
            taps.append(t)
        return {"B": gas > 0.5, "LEFT": taps[0] < 0, "RIGHT": taps[0] > 0, "L": taps[1] < 0,
                "R": taps[1] > 0, "Y": brake > 0.5, "A": boost > 0.5}
