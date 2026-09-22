"""Play F-Zero yourself and record your inputs, so the fly (or a neural network) can learn from you.

    flyzero record --rom "F-Zero (USA).sfc" --out my_races.npz

A window opens, the usual menu macro runs (Grand Prix, Blue Falcon, Knight League, Beginner,
Mute City I), and you get control at the start line. Every frame's buttons are saved, plus a
few checkpoints of the game state, so the race can be replayed exactly on another machine
running the same emulator core (stable-retro's snes9x). Only your inputs are saved, not the
ROM or any video.

Keys (keyboard / typical gamepad mapping):

    arrows   steer            X  accelerate (B)      Z  brake (Y)
    C        super jet (A)    A  lean left (L)       S  lean right (R)
    Enter    pause (START)    Backspace  restart the race     Esc  save and quit
    F        toggle 1x / 2x window scale

Everything you play is saved: each attempt is one "race" in the file. Only needs numpy,
pyglet and stable-retro (no connectome, no brain).
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from .games import FZero, RAM_LAP, RAM_SEGMENT
from .motor import BUTTONS

KEYMAP = {  # pyglet key name -> SNES button
    "LEFT": "LEFT", "RIGHT": "RIGHT", "UP": "UP", "DOWN": "DOWN",
    "X": "B", "Z": "Y", "C": "A", "V": "X", "A": "L", "S": "R",
    "ENTER": "START", "RETURN": "START",
}
CHECK_EVERY = 60  # frames between sync checkpoints (lap, segment)


def buttons_to_mask(pressed: dict) -> np.ndarray:
    return np.array([pressed.get(b, False) for b in BUTTONS], np.uint8)


def mask_to_buttons(mask: np.ndarray) -> dict:
    return {b: bool(v) for b, v in zip(BUTTONS, mask)}


class Session:
    """Game + recording logic, independent of the window (so it can be tested headless)."""

    def __init__(self, rom: str):
        self.rom = rom
        self.game = FZero(rom)
        self.races = []
        self.sha1 = hashlib.sha1(Path(rom).read_bytes()).hexdigest()
        self.new_race()

    def new_race(self):
        self.frame = self.game.reset()
        self.start_state = bytes(self.game.em.get_state())
        self.masks = []
        self.checks = []

    def step(self, pressed: dict) -> np.ndarray:
        mask = buttons_to_mask(pressed)
        self.frame = self.game.step(pressed)
        self.masks.append(mask)
        if len(self.masks) % CHECK_EVERY == 0:
            ram = self.game.ram()
            self.checks.append((len(self.masks), int(ram[RAM_LAP]), int(ram[RAM_SEGMENT])))
        return self.frame

    def finish_race(self):
        if len(self.masks) > 60:
            info = self.game.info
            self.races.append({"masks": np.array(self.masks, np.uint8),
                               "checks": np.array(self.checks, np.int32).reshape(-1, 3),
                               "laps": info.get("lap", 0), "segment": info.get("segment", 0),
                               "energy": info.get("energy", 0.0)})
            print(f"race {len(self.races)}: {len(self.masks)} frames, laps {info.get('lap', 0)}, "
                  f"energy {info.get('energy', 0.0)}")

    def save(self, out: str):
        self.finish_race()
        data = {"rom_sha1": self.sha1, "buttons": np.array(BUTTONS), "n_races": len(self.races),
                "menu": np.array(repr(FZero.menu))}
        for i, r in enumerate(self.races):
            for k, v in r.items():
                data[f"race{i}_{k}"] = v
        np.savez_compressed(out, **data)
        print(f"saved {len(self.races)} race(s) to {out}")


def play(rom: str, out: str, scale: int = 3, max_frames: int = 0):
    import pyglet
    from pyglet.window import key

    s = Session(rom)
    h, w = s.frame.shape[:2]
    win = pyglet.window.Window(w * scale, h * scale, caption="F-Zero: teach the fly  (Esc = save & quit)")
    keys = key.KeyStateHandler()
    win.push_handlers(keys)
    state = {"scale": scale, "quit": False}

    @win.event
    def on_key_press(symbol, modifiers):
        if symbol == key.ESCAPE:
            state["quit"] = True
            return pyglet.event.EVENT_HANDLED
        if symbol == key.BACKSPACE:
            s.finish_race()
            s.new_race()
        if symbol == key.F:
            state["scale"] = 2 if state["scale"] != 2 else 3
            win.set_size(w * state["scale"], h * state["scale"])

    @win.event
    def on_close():
        state["quit"] = True

    print(__doc__.split("Keys")[1])
    frame_time = 1.0 / FZero.fps
    next_t = time.perf_counter()
    while not state["quit"]:
        win.dispatch_events()
        pressed = {btn: bool(keys[getattr(key, name)]) for name, btn in KEYMAP.items()
                   if hasattr(key, name)}
        frame = s.step(pressed)
        img = pyglet.image.ImageData(w, h, "RGB", np.ascontiguousarray(frame[::-1]).tobytes())
        win.clear()
        img.get_texture().blit(0, 0, width=w * state["scale"], height=h * state["scale"])
        win.flip()
        next_t += frame_time
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.perf_counter()
        if max_frames and len(s.masks) >= max_frames:
            state["quit"] = True
    win.close()
    s.save(out)


def replay(rom: str, recording: str, race: int = 0, on_frame=None) -> dict:
    """Re-run a recorded race here; returns frames' count and whether the checkpoints matched."""
    d = np.load(recording)
    masks = d[f"race{race}_masks"]
    checks = {int(f): (int(lap), int(seg)) for f, lap, seg in d[f"race{race}_checks"]}
    game = FZero(rom)
    game.reset()
    mismatches = 0
    for i, m in enumerate(masks, start=1):
        buttons = mask_to_buttons(m)
        frame = game.step(buttons)
        if on_frame:
            on_frame(frame, buttons, game.info)
        if i in checks:
            ram = game.ram()
            if (int(ram[RAM_LAP]), int(ram[RAM_SEGMENT])) != checks[i]:
                mismatches += 1
    return {"frames": len(masks), "checkpoints": len(checks), "mismatches": mismatches,
            "laps": game.info.get("lap"), "segment": game.info.get("segment")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom", required=True)
    ap.add_argument("--out", default="my_races.npz")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    play(a.rom, a.out, a.scale, a.max_frames)


if __name__ == "__main__":
    main()
