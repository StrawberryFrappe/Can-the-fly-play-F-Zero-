"""Retinotopy, and the naive eye: game frames -> photoreceptor firing rates.

(The default input is ``vision.MotionEye``; see there for why driving the
photoreceptors of a spiking model barely reaches the rest of the brain.)

Each photoreceptor gets a spot in the visual field. FlyWire does not label
ommatidial coordinates, but photoreceptor axons tile the lamina/medulla
retinotopically, so we recover an approximate visual field position from the
neuron's position in the brain: project each eye's photoreceptors onto the two
principal axes of their point cloud, orient the axes so that dorsal is up and
lateral is outward, and stretch them to cover that eye's half of the screen
(the left eye sees the left half, with a little binocular overlap).

A pixel patch around that spot is averaged ("ommatidium") and the rate is a mix
of luminance and luminance change, since fly photoreceptors are strongly
transient and the motion pathways downstream care about change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import Connectome


@dataclass
class EyeParams:
    max_rate: float = 150.0      # Hz, same as the Poisson rate used by Shiu et al.
    luminance_gain: float = 0.3  # share of the rate driven by steady brightness
    change_gain: float = 4.0     # gain on |dL| per frame
    overlap: float = 0.1         # binocular overlap (fraction of screen width)
    blur: int = 3                # ommatidium half-size in (downsampled) pixels


def retinotopy(conn: Connectome, cells: np.ndarray, w: int, h: int, overlap: float = 0.1):
    """Screen pixel (px, py) and eye ('left'/'right') for each neuron in ``cells``.

    Per eye, the neurons' positions are projected onto the two main axes of
    their point cloud; dorsal is up and lateral is outwards. The left eye sees
    the left half of the screen (plus ``overlap``), the right eye the right half.
    Returns ``px, py, side, keep`` where ``keep`` masks neurons that could be
    placed.
    """
    nrn = conn.neurons.iloc[cells]
    pos = nrn[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
    side = nrn["side"].fillna("").to_numpy().astype(object)
    ok = np.isfinite(pos).all(1)
    mid = np.nanmedian(conn.neurons["pos_x"].to_numpy(float))
    # side labels missing? fall back to which side of the midline it is on
    unknown = ~np.isin(side, ["left", "right"])
    side = np.where(unknown, np.where(pos[:, 0] > mid, "left", "right"), side)

    xy = np.full((len(cells), 2), np.nan)  # visual field, 0..1
    for s in ("left", "right"):
        m = (side == s) & ok
        if m.sum() < 3:
            continue
        xy[m] = _field(pos[m], mid)
    keep = np.isfinite(xy).all(1)
    xy = np.nan_to_num(xy)
    o = overlap / 2
    left = side == "left"
    # left eye covers [0, 0.5+o] of the screen, right eye [0.5-o, 1]; lateral = outer edge
    sx = np.where(left, (1 - xy[:, 0]) * (0.5 + o), 0.5 - o + xy[:, 0] * (0.5 + o))
    px = np.clip((sx * w).astype(int), 0, w - 1)
    py = np.clip((xy[:, 1] * h).astype(int), 0, h - 1)
    return px, py, side.astype(str), keep


def _field(pos: np.ndarray, mid: float) -> np.ndarray:
    c = pos - pos.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    uv = c @ vt[:2].T
    # vertical axis = the component best aligned with brain y (FlyWire y grows ventrally)
    cy = [abs(np.corrcoef(uv[:, k], pos[:, 1])[0, 1]) for k in range(2)]
    v_k = int(np.argmax(cy))
    u = uv[:, 1 - v_k]
    v = uv[:, v_k]
    if np.corrcoef(v, pos[:, 1])[0, 1] < 0:
        v = -v
    lateral = np.abs(pos[:, 0] - mid)
    if np.corrcoef(u, lateral)[0, 1] < 0:
        u = -u
    norm = lambda a: (a - a.min()) / max(np.ptp(a), 1e-9)  # noqa: E731
    return np.c_[norm(u), norm(v)]  # u: 0 = medial, 1 = lateral; v: 0 = dorsal (top)


class CompoundEye:
    """Drives the photoreceptors directly (see ``vision`` for why that barely works)."""

    def __init__(self, conn: Connectome, frame_shape: tuple[int, int], params: EyeParams | None = None):
        self.p = params or EyeParams()
        self.h, self.w = frame_shape
        idx = conn.photoreceptors()
        if len(idx) == 0:
            raise ValueError("connectome has no photoreceptors to plug the game into")
        px, py, side, keep = retinotopy(conn, idx, self.w, self.h, self.p.overlap)
        self.idx, self.px, self.py, self.side = idx[keep], px[keep], py[keep], side[keep]
        self.prev = None
        self.last_rates = np.zeros(len(self.idx), np.float32)

    def describe(self) -> str:
        return (f"{len(self.idx)} photoreceptors "
                f"({(self.side == 'left').sum()} left, {(self.side == 'right').sum()} right)")

    def see(self, frame: np.ndarray) -> np.ndarray:
        """frame: HxWx3 uint8 -> Poisson rate (Hz) per photoreceptor."""
        lum = frame.astype(np.float32).mean(2) / 255.0
        b = self.p.blur
        if b > 0:  # box blur via an integral image = ommatidial acceptance angle
            ii = np.pad(lum, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            y0 = np.clip(self.py - b, 0, self.h); y1 = np.clip(self.py + b + 1, 0, self.h)
            x0 = np.clip(self.px - b, 0, self.w); x1 = np.clip(self.px + b + 1, 0, self.w)
            area = (y1 - y0) * (x1 - x0)
            sample = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / area
        else:
            sample = lum[self.py, self.px]
        change = np.zeros_like(sample) if self.prev is None else np.abs(sample - self.prev)
        self.prev = sample
        drive = self.p.luminance_gain * sample + self.p.change_gain * change
        self.last_rates = (self.p.max_rate * np.clip(drive, 0, 1)).astype(np.float32)
        return self.last_rates
