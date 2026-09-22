"""Compound eyes: turn game frames into photoreceptor firing rates.

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


class CompoundEye:
    def __init__(self, conn: Connectome, frame_shape: tuple[int, int], params: EyeParams | None = None):
        self.p = params or EyeParams()
        self.h, self.w = frame_shape
        self.idx = conn.photoreceptors()
        if len(self.idx) == 0:
            raise ValueError("connectome has no photoreceptors to plug the game into")
        nrn = conn.neurons.iloc[self.idx]
        pos = nrn[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
        side = nrn["side"].fillna("").to_numpy()
        ok = np.isfinite(pos).all(1)
        mid = np.nanmedian(conn.neurons["pos_x"].to_numpy(float))
        # side labels missing? fall back to which side of the midline it is on
        unknown = ~np.isin(side, ["left", "right"])
        side = np.where(unknown, np.where(pos[:, 0] > mid, "left", "right"), side)

        self.side = side
        self.xy = np.full((len(self.idx), 2), np.nan)  # visual field, 0..1
        for s in ("left", "right"):
            m = (side == s) & ok
            if m.sum() < 3:
                continue
            self.xy[m] = self._retinotopy(pos[m], s, mid)
        keep = np.isfinite(self.xy).all(1)
        self.idx, self.xy, self.side = self.idx[keep], self.xy[keep], self.side[keep]

        o = self.p.overlap / 2
        left = self.side == "left"
        # left eye covers [0, 0.5+o] of the screen, right eye [0.5-o, 1]; lateral = outer edge
        sx = np.where(left, (1 - self.xy[:, 0]) * (0.5 + o), 0.5 - o + self.xy[:, 0] * (0.5 + o))
        self.px = np.clip((sx * self.w).astype(int), 0, self.w - 1)
        self.py = np.clip((self.xy[:, 1] * self.h).astype(int), 0, self.h - 1)
        self.prev = None
        self.last_rates = np.zeros(len(self.idx), np.float32)

    @staticmethod
    def _retinotopy(pos: np.ndarray, side: str, mid: float) -> np.ndarray:
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
