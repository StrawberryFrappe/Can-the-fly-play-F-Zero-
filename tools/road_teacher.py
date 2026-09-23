import numpy as np

def road_mask(band):
    b = band.astype(int)
    sat = b.max(2) - b.min(2)
    lum = b.mean(2)
    return (sat < 36) & (lum > 95) & (lum < 205)   # greys: road incl. speckled stretches, not white beams

class Teacher:
    """Visual-only lane keeping: steer towards the middle of the grey road, near and far."""
    def __init__(self, k_far=1.0, thr=0.06, lean_thr=9.0, clip=0.6, rows_near=(160, 205), rows_far=(110, 150)):
        self.near = np.arange(*rows_near, 5); self.far = np.arange(*rows_far, 5)
        self.k_far, self.thr, self.lean_thr, self.clip = k_far, thr, lean_thr, clip
    def centre(self, frame, rows):
        m = road_mask(frame[rows])
        cnt = m.sum(1); ok = cnt > 10
        if not ok.any():
            return 0.0
        cols = np.arange(frame.shape[1])
        return float(np.mean(((m * cols).sum(1)[ok] / cnt[ok] - 128) / 128))
    def signal(self, frame):
        s = self.centre(frame, self.near) + self.k_far * self.centre(frame, self.far)
        return float(np.clip(s, -self.clip, self.clip))
    def __call__(self, frame):
        s = self.signal(frame)
        return {"B": True, "LEFT": s < -self.thr, "RIGHT": s > self.thr,
                "L": s < -self.lean_thr, "R": s > self.lean_thr}, s


class GuardedTeacher(Teacher):
    """Visual lane keeping + a privileged rule: if the track counter runs backwards, turn around."""
    def __init__(self, turn="LEFT", **kw):
        super().__init__(**kw)
        self.turn = turn; self.hist = []; self.reversing = 0
    def act(self, frame, info):
        b, s = self(frame)
        self.hist.append(info.get("segment", 0)); self.hist = self.hist[-90:]
        if len(self.hist) == 90:
            d = (self.hist[-1] - self.hist[0] + 29) % 59 - 29
            if d < -1: self.reversing = 90
        if self.reversing > 0:
            self.reversing -= 1
            b = {"B": True, self.turn: True, self.turn[0]: True}
        return b, s, self.reversing > 0
