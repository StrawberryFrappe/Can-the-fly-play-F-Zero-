"""Dashboard frames: the game, what the fly sees, what its DNs do, which buttons are down."""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_W = 300
BG = (14, 14, 22)
FG = (225, 225, 235)
DIM = (120, 120, 140)
ACCENT = (240, 60, 160)
ON = (250, 210, 50)

LABELS = {
    "steer_left": "DNa02 L  steer <",
    "steer_right": "DNa02 R  steer >",
    "lean_left": "DNa01 L  lean L",
    "lean_right": "DNa01 R  lean R",
    "accelerate": "DNp09    gas (B)",
    "brake": "MDN      brake (Y)",
    "boost": "DNp01 GF boost (A)",
}


class Dashboard:
    def __init__(self, eye, motor, brain_name: str, scale: int = 2):
        self.eye, self.motor, self.scale = eye, motor, scale
        self.brain_name = brain_name
        self.font = ImageFont.load_default()
        self.activity = deque(maxlen=PANEL_W - 24)

    def render(self, frame: np.ndarray, t_ms: float, spikes: int, n_active: int, info: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        s = self.scale
        game = Image.fromarray(frame).resize((w * s, h * s), Image.NEAREST)
        H = h * s
        img = Image.new("RGB", (w * s + PANEL_W, H), BG)
        img.paste(game, (0, 0))
        d = ImageDraw.Draw(img)
        x0 = w * s + 12
        y = 8
        d.text((x0, y), "FLY BRAIN x F-ZERO", fill=ACCENT, font=self.font); y += 14
        d.text((x0, y), self.brain_name, fill=DIM, font=self.font); y += 12
        d.text((x0, y), f"t = {t_ms / 1000:7.2f} s neural time", fill=FG, font=self.font); y += 12
        d.text((x0, y), f"{spikes:6d} spikes / frame, {n_active} neurons", fill=FG, font=self.font); y += 18

        # the fly's eye view: every photoreceptor as a dot where it looks
        ew, eh = PANEL_W - 24, int((PANEL_W - 24) * h / w)
        d.text((x0, y), "photoreceptor input", fill=DIM, font=self.font); y += 12
        d.rectangle([x0, y, x0 + ew, y + eh], fill=(0, 0, 0))
        r = self.eye.last_rates / max(self.eye.p.max_rate, 1)
        ex = x0 + self.eye.px * ew // w
        ey = y + self.eye.py * eh // h
        eye_img = np.zeros((eh + 1, ew + 1, 3), np.uint8)
        col = np.stack([r * 255, r * 120 + 30 * (self.eye.side == "left"), 60 + r * 195], 1).clip(0, 255)
        eye_img[ey - y, ex - x0] = col.astype(np.uint8)
        img.paste(Image.fromarray(eye_img), (x0, y))
        y += eh + 10

        # descending neuron rates
        d.text((x0, y), "descending neurons (Hz)", fill=DIM, font=self.font); y += 12
        bw = PANEL_W - 24 - 110
        for name, label in LABELS.items():
            rate = self.motor.rates.get(name, 0.0)
            n = len(self.motor.groups.get(name, []))
            d.text((x0, y), label if n else label + " (n/a)", fill=FG if n else DIM, font=self.font)
            frac = min(rate / 100.0, 1.0)
            d.rectangle([x0 + 110, y + 2, x0 + 110 + bw, y + 9], outline=(50, 50, 70))
            d.rectangle([x0 + 110, y + 2, x0 + 110 + int(frac * bw), y + 9], fill=ACCENT)
            d.text((x0 + 112 + bw - 30, y), f"{rate:5.1f}", fill=FG, font=self.font)
            y += 12
        y += 6

        # controller
        b = self.motor.buttons
        d.text((x0, y), "controller", fill=DIM, font=self.font); y += 14
        cx, cy = x0 + 40, y + 24
        for key, (dx, dy) in {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}.items():
            bx, by = cx + dx * 16, cy + dy * 16
            d.rectangle([bx - 7, by - 7, bx + 7, by + 7], fill=ON if b.get(key) else (60, 60, 80))
        fx = x0 + 180
        for key, (dx, dy) in {"X": (0, -1), "Y": (-1, 0), "A": (1, 0), "B": (0, 1)}.items():
            bx, by = fx + dx * 18, cy + dy * 18
            d.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=ON if b.get(key) else (60, 60, 80))
            d.text((bx - 3, by - 6), key, fill=BG if b.get(key) else FG, font=self.font)
        for key, bx in (("L", x0 + 10), ("R", x0 + 200)):
            d.rectangle([bx, y - 6, bx + 40, y], fill=ON if b.get(key) else (60, 60, 80))
        y += 62

        # population activity trace
        self.activity.append(spikes)
        d.text((x0, y), "whole-brain spikes per frame", fill=DIM, font=self.font); y += 12
        th = 40
        if self.activity:
            a = np.array(self.activity, float)
            a = a / max(a.max(), 1)
            pts = [(x0 + i, y + th - int(v * th)) for i, v in enumerate(a)]
            if len(pts) > 1:
                d.line(pts, fill=(90, 200, 250))
        y += th + 8
        for k, v in info.items():
            if y > H - 12:
                break
            v = f"{v:.1f}" if isinstance(v, float) else str(v)
            d.text((x0, y), f"{k}: {v}", fill=FG, font=self.font); y += 12
        return np.asarray(img)
