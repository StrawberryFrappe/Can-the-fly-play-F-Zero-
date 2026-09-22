"""Play F-Zero yourself and record your inputs, so the fly (or a neural network) can learn from you.

    flyzero record --rom "F-Zero (USA).sfc" --out my_races.npz

A window opens, the usual menu macro runs (Grand Prix, Blue Falcon, Knight League, Beginner,
Mute City I), and you get control at the start line. Every frame's buttons are saved, plus a
few checkpoints of the game state, so the race can be replayed exactly on another machine
running the same emulator core (stable-retro's snes9x). Only your inputs are saved, not the
ROM or any video.

Controls. Xbox controller (first one found):

    D-pad / left stick  steer      A  gas          B  super jet     X  brake
    LB / RB             lean L / R Start  pause    View/Back  restart the race

Keyboard (works at the same time):

    arrows   steer            X  accelerate (B)      Z  brake (Y)
    C        super jet (A)    A  lean left (L)       S  lean right (R)
    Enter    pause (START)    Backspace  restart the race     Esc  save and quit
    F        toggle window size

Controllers number their buttons differently per OS. Run with --controller-test to see what
each button reports, then pass e.g. --map "A=0,B=1,X=2,LB=4,RB=5,START=7,BACK=6".

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

# Xbox controller: button index (Linux xpad / xinput order) -> what it does in F-Zero
XBOX_DEFAULT = {"A": 0, "B": 1, "X": 2, "Y": 3, "LB": 4, "RB": 5, "BACK": 6, "START": 7}
XBOX_TO_SNES = {"A": "B", "B": "A", "X": "Y", "LB": "L", "RB": "R", "START": "START"}


def parse_map(text: str | None) -> dict:
    m = dict(XBOX_DEFAULT)
    for part in (text or "").split(","):
        if "=" in part:
            k, v = part.split("=")
            m[k.strip().upper()] = int(v)
    return m


def pad_to_buttons(buttons, hat_x: float, hat_y: float, stick_x: float, mapping: dict) -> dict:
    """Xbox controller state -> SNES buttons (the user's layout)."""
    def down(name):
        i = mapping.get(name)
        return i is not None and i < len(buttons) and bool(buttons[i])
    out = {snes: down(xbox) for xbox, snes in XBOX_TO_SNES.items()}
    out["LEFT"] = hat_x < -0.5 or stick_x < -0.5 or down("DPAD_LEFT")
    out["RIGHT"] = hat_x > 0.5 or stick_x > 0.5 or down("DPAD_RIGHT")
    out["UP"] = hat_y > 0.5 or down("DPAD_UP")
    out["DOWN"] = hat_y < -0.5 or down("DPAD_DOWN")
    return out


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


def open_pad(window=None):
    import pyglet

    try:
        pads = pyglet.input.get_joysticks()
    except Exception as e:  # no /dev/input, no permission, ...
        print(f"can't read controllers ({e}): keyboard only")
        return None
    if not pads:
        print("no controller found: keyboard only")
        return None
    pad = pads[0]
    try:
        pad.open(window)
    except Exception as e:
        print(f"can't open {pad.device.name} ({e}): keyboard only")
        return None
    print(f"controller: {pad.device.name}")
    return pad


def play(rom: str, out: str, scale: int = 3, max_frames: int = 0, pad_map: str | None = None):
    import pyglet
    from pyglet.window import key

    s = Session(rom)
    h, w = s.frame.shape[:2]
    win = pyglet.window.Window(w * scale, h * scale, caption="F-Zero: teach the fly  (Esc = save & quit)")
    keys = key.KeyStateHandler()
    win.push_handlers(keys)
    pad = open_pad(win)
    mapping = parse_map(pad_map)
    state = {"scale": scale, "back_was_down": False}

    def restart():
        s.finish_race()
        s.new_race()

    @win.event
    def on_key_press(symbol, modifiers):
        if symbol == key.ESCAPE:
            pyglet.app.exit()
            return pyglet.event.EVENT_HANDLED
        if symbol == key.BACKSPACE:
            restart()
        if symbol == key.F:
            state["scale"] = 2 if state["scale"] != 2 else 3
            win.set_size(w * state["scale"], h * state["scale"])

    @win.event
    def on_close():
        pyglet.app.exit()

    @win.event
    def on_draw():
        img = pyglet.image.ImageData(w, h, "RGB", np.ascontiguousarray(s.frame[::-1]).tobytes())
        win.clear()
        img.get_texture().blit(0, 0, width=win.width, height=win.height)

    def tick(dt):
        pressed = {btn: bool(keys[getattr(key, name)]) for name, btn in KEYMAP.items() if hasattr(key, name)}
        if pad is not None:
            p = pad_to_buttons(pad.buttons, pad.hat_x, pad.hat_y, pad.x, mapping)
            pressed = {b: pressed.get(b, False) or p.get(b, False) for b in set(pressed) | set(p)}
            back = mapping.get("BACK") is not None and mapping["BACK"] < len(pad.buttons) \
                and bool(pad.buttons[mapping["BACK"]])
            if back and not state["back_was_down"]:
                restart()
            state["back_was_down"] = back
        s.step(pressed)
        if max_frames and len(s.masks) >= max_frames:
            pyglet.app.exit()

    print(__doc__.split("Controls.")[1].split("Everything")[0])
    pyglet.clock.schedule_interval(tick, 1.0 / FZero.fps)
    pyglet.app.run()
    win.close()
    s.save(out)


def controller_test(seconds: float = 60.0):
    """Print what the controller reports, to fix the mapping with --map."""
    import pyglet

    win = pyglet.window.Window(360, 120, caption="controller test: press buttons (Esc to quit)")
    pad = open_pad(win)
    if pad is None:
        return
    last = {}

    def tick(dt):
        now = {"buttons": [i for i, v in enumerate(pad.buttons) if v],
               "hat": (pad.hat_x, pad.hat_y), "stick": (round(pad.x, 1), round(pad.y, 1))}
        if now != last:
            print(now, flush=True)
            last.clear(); last.update(now)

    pyglet.clock.schedule_interval(tick, 1 / 30)
    pyglet.clock.schedule_once(lambda dt: pyglet.app.exit(), seconds)
    pyglet.app.run()


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
    ap.add_argument("--map", help='controller button numbers, e.g. "A=0,B=1,X=2,LB=4,RB=5"')
    ap.add_argument("--controller-test", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.controller_test:
        controller_test()
    else:
        play(a.rom, a.out, a.scale, a.max_frames, a.map)


if __name__ == "__main__":
    main()
