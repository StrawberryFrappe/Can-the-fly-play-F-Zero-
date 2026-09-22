"""Play F-Zero yourself and record your inputs, so the fly (or a neural network) can learn from you.

    flyzero record --rom "F-Zero (USA).sfc" --out my_races.npz

A window opens, the usual menu macro runs (Grand Prix, Blue Falcon, Knight League, Beginner,
Mute City I), and you get control at the start line. Every frame's buttons are saved, plus a
few checkpoints of the game state, so the race can be replayed exactly on another machine
running the same emulator core (snes9x). Only your inputs are saved, not the ROM or any video.

Windows: stable-retro has no Windows build, so point --core at a snes9x libretro core
(snes9x_libretro.dll from RetroArch's "cores" folder or the libretro buildbot), or just put the
.dll next to where you run the command.

Controls. Xbox controller (first one found; on Windows read through XInput):

    D-pad / left stick  steer      RT or A  gas    LT or X  brake    B  super jet
    LB / RB             lean L / R Start  pause    View/Back  restart the race

Keyboard (works at the same time):

    arrows   steer            X  accelerate (B)      Z  brake (Y)
    C        super jet (A)    A  lean left (L)       S  lean right (R)
    Enter    pause (START)    Backspace  restart the race     Esc  save and quit
    F        toggle window size

On Linux/macOS controllers number their buttons differently; run with --controller-test to see
what each button reports, then pass e.g. --map "A=0,B=1,X=2,LB=4,RB=5,START=7,BACK=6".
(Windows uses XInput, where the layout is fixed.)

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


def pad_to_buttons(buttons, hat_x: float, hat_y: float, stick_x: float, mapping: dict,
                   lt: float = 0.0, rt: float = 0.0) -> dict:
    """Xbox controller state -> SNES buttons (the user's layout). Triggers are 0..1."""
    def down(name):
        i = mapping.get(name)
        return i is not None and i < len(buttons) and bool(buttons[i])
    out = {snes: down(xbox) for xbox, snes in XBOX_TO_SNES.items()}
    out["B"] = out["B"] or rt > 0.3   # RT = gas
    out["Y"] = out["Y"] or lt > 0.3   # LT = brake
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

    def __init__(self, rom: str, core: str | None = None):
        self.rom = rom
        self.game = FZero(rom, core=core)
        print(f"emulator: {self.game.backend}")
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


# XInput (Windows): fixed layout, separate analog triggers
XI_BUTTONS = {"DPAD_UP": 0x0001, "DPAD_DOWN": 0x0002, "DPAD_LEFT": 0x0004, "DPAD_RIGHT": 0x0008,
              "START": 0x0010, "BACK": 0x0020, "LB": 0x0100, "RB": 0x0200,
              "A": 0x1000, "B": 0x2000, "X": 0x4000, "Y": 0x8000}
XI_ORDER = list(XI_BUTTONS)  # index = position in this list, used as the "mapping"


def xinput_to_buttons(wbuttons: int, lt: int, rt: int, lx: int) -> tuple[dict, bool]:
    """Raw XInput state -> (SNES buttons, View/Back pressed)."""
    flags = [bool(wbuttons & XI_BUTTONS[k]) for k in XI_ORDER]
    mapping = {k: i for i, k in enumerate(XI_ORDER)}
    out = pad_to_buttons(flags, 0.0, 0.0, lx / 32767.0, mapping, lt / 255.0, rt / 255.0)
    return out, bool(wbuttons & XI_BUTTONS["BACK"])


class XInputPad:
    """Xbox controller on Windows via xinput1_4/xinput9_1_0 (no pyglet needed)."""

    def __init__(self):
        import ctypes as C

        class Gamepad(C.Structure):
            _fields_ = [("wButtons", C.c_ushort), ("bLeftTrigger", C.c_ubyte), ("bRightTrigger", C.c_ubyte),
                        ("sThumbLX", C.c_short), ("sThumbLY", C.c_short),
                        ("sThumbRX", C.c_short), ("sThumbRY", C.c_short)]

        class State(C.Structure):
            _fields_ = [("dwPacketNumber", C.c_uint), ("Gamepad", Gamepad)]

        for dll in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
            try:
                self.lib = getattr(C.windll, dll)
                break
            except OSError:
                continue
        else:
            raise OSError("no XInput dll")
        self.state = State()
        self.C = C
        self.index = next((i for i in range(4) if self._poll(i)), None)
        if self.index is None:
            raise OSError("no XInput controller connected")

    def _poll(self, i):
        return self.lib.XInputGetState(i, self.C.byref(self.state)) == 0

    def read(self) -> tuple[dict, bool]:
        if not self._poll(self.index):
            return {}, False
        g = self.state.Gamepad
        return xinput_to_buttons(g.wButtons, g.bLeftTrigger, g.bRightTrigger, g.sThumbLX)


def open_pad(window=None):
    import sys

    import pyglet

    if sys.platform == "win32":
        try:
            pad = XInputPad()
            print(f"controller: XInput pad #{pad.index}")
            return pad
        except OSError as e:
            print(f"XInput: {e}; trying DirectInput")

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


def play(rom: str, out: str, scale: int = 3, max_frames: int = 0, pad_map: str | None = None,
         core: str | None = None):
    import pyglet
    from pyglet.window import key

    s = Session(rom, core)
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
        if isinstance(pad, XInputPad):
            p, back = pad.read()
        elif pad is not None:
            # triggers: separate axes (Linux: z = LT, rz = RT); measured against their resting value
            z, rz = getattr(pad, "z", 0.0), getattr(pad, "rz", 0.0)
            rest = state.setdefault("trigger_rest", (z, rz))
            lt = max(0.0, (z - rest[0]) / max(1.0 - rest[0], 1e-6))
            rt = max(0.0, (rz - rest[1]) / max(1.0 - rest[1], 1e-6))
            p = pad_to_buttons(pad.buttons, pad.hat_x, pad.hat_y, pad.x, mapping, lt, rt)
            back = mapping.get("BACK") is not None and mapping["BACK"] < len(pad.buttons) \
                and bool(pad.buttons[mapping["BACK"]])
        if pad is not None:
            pressed = {b: pressed.get(b, False) or p.get(b, False) for b in set(pressed) | set(p)}
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
        if isinstance(pad, XInputPad):
            now = {"snes": sorted(k for k, v in pad.read()[0].items() if v)}
        else:
            now = {"buttons": [i for i, v in enumerate(pad.buttons) if v],
                   "hat": (pad.hat_x, pad.hat_y), "stick": (round(pad.x, 1), round(pad.y, 1)),
                   "triggers z/rz": (round(getattr(pad, "z", 0), 1), round(getattr(pad, "rz", 0), 1))}
        if now != last:
            print(now, flush=True)
            last.clear(); last.update(now)

    pyglet.clock.schedule_interval(tick, 1 / 30)
    pyglet.clock.schedule_once(lambda dt: pyglet.app.exit(), seconds)
    pyglet.app.run()


def replay(rom: str, recording: str, race: int = 0, on_frame=None, core: str | None = None) -> dict:
    """Re-run a recorded race here; returns frames' count and whether the checkpoints matched."""
    d = np.load(recording)
    masks = d[f"race{race}_masks"]
    checks = {int(f): (int(lap), int(seg)) for f, lap, seg in d[f"race{race}_checks"]}
    game = FZero(rom, core=core)
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
    ap.add_argument("--core", help="snes9x libretro core (.dll/.so/.dylib) instead of stable-retro")
    ap.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.controller_test:
        controller_test()
    else:
        play(a.rom, a.out, a.scale, a.max_frames, a.map, a.core)


if __name__ == "__main__":
    main()
