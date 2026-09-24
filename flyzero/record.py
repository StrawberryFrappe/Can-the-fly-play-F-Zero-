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
    LB / RB             lean L / R Start  pause    View/Back  back to the Mute City grid

Keyboard (works at the same time):

    arrows   steer            X  accelerate (B)      Z  brake (Y)
    C        super jet (A)    A  lean left (L)       S  lean right (R)
    Enter    pause (START)    Backspace  back to the Mute City grid     Esc  save and quit
    F        toggle window size

On Linux/macOS controllers number their buttons differently; run with --controller-test to see
what each button reports, then pass e.g. --map "A=0,B=1,X=2,LB=4,RB=5,START=7,BACK=6".
(Windows uses XInput, where the layout is fixed.)

Everything you play is saved: each attempt is one "race" in the file. To race the whole Grand
Prix, just keep driving through the results screens. --mute-city: after each finish you're put
straight back on the Mute City I grid (back-to-back runs). --queen-league: the menus pick the
Queen League (Mute City II, Port Town I, Red Canyon I, White Land I, White Land II). Game sound plays if `sounddevice` is
installed (--no-audio to mute). Only needs numpy,
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

    def __init__(self, rom: str, core: str | None = None, league: str = "knight"):
        self.rom = rom
        self.league = league
        self.game = FZero(rom, core=core, league=league)
        print(f"emulator: {self.game.backend}")
        self.races = []
        self.sha1 = hashlib.sha1(Path(rom).read_bytes()).hexdigest()
        self.power_on = bytes(self.game.em.get_state())
        self.new_race()

    def new_race(self):
        """Back to the first grid of the league from power-on (to race the whole Grand Prix, just
        keep driving through the results screens instead)."""
        self.game.em.set_state(self.power_on)
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
                               "start_state": np.frombuffer(self.start_state, np.uint8),
                               "checks": np.array(self.checks, np.int32).reshape(-1, 3),
                               "laps": info.get("lap", 0), "segment": info.get("segment", 0),
                               "energy": info.get("energy", 0.0)})
            print(f"race {len(self.races)}: {len(self.masks)} frames, laps {info.get('lap', 0)}, "
                  f"energy {info.get('energy', 0.0)}")

    def save(self, out: str):
        self.finish_race()
        data = {"rom_sha1": self.sha1, "buttons": np.array(BUTTONS), "n_races": len(self.races),
                "menu": np.array(repr(self.game.menu)), "league": np.array(self.league),
                "check_lap_addr": RAM_LAP}
        out = Path(out)
        if out.exists():  # never overwrite earlier sessions: my_races.npz -> my_races_2.npz, ...
            k = 2
            while out.with_name(f"{out.stem}_{k}{out.suffix}").exists():
                k += 1
            out = out.with_name(f"{out.stem}_{k}{out.suffix}")
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


class AudioOut:
    """Plays the game's sound: resampled to the device rate, kept at most ~0.1 s behind."""

    def __init__(self, in_rate: float, out_rate: int = 48000, start: bool = True):
        import collections
        import threading

        self.in_rate, self.out_rate = in_rate, out_rate
        self.chunks = collections.deque()
        self.queued = 0
        self.lock = threading.Lock()
        self.stream = None
        if start:
            import sounddevice as sd

            self.stream = sd.OutputStream(samplerate=out_rate, channels=2, dtype="int16",
                                          callback=self._callback, latency="low")
            self.stream.start()

    def push(self, samples: np.ndarray):
        if len(samples) == 0:
            return
        n_out = int(round(len(samples) * self.out_rate / self.in_rate))
        t_in = np.arange(len(samples))
        t_out = np.linspace(0, len(samples) - 1, n_out)
        res = np.stack([np.interp(t_out, t_in, samples[:, ch]) for ch in (0, 1)], 1).astype(np.int16)
        with self.lock:
            self.chunks.append(res)
            self.queued += len(res)
            while self.queued > self.out_rate // 10 and len(self.chunks) > 1:  # drop, don't lag
                self.queued -= len(self.chunks.popleft())

    def _callback(self, outdata, frames, time_info, status):
        out = np.zeros((frames, 2), np.int16)
        filled = 0
        with self.lock:
            while filled < frames and self.chunks:
                c = self.chunks[0]
                take = min(frames - filled, len(c))
                out[filled:filled + take] = c[:take]
                filled += take
                if take == len(c):
                    self.chunks.popleft()
                else:
                    self.chunks[0] = c[take:]
                self.queued -= take
        outdata[:] = out

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


def open_audio(game, enabled: bool = True):
    if not enabled:
        return None
    try:
        out = AudioOut(game.audio_rate())
        game.collect_audio = True
        print("audio: on")
        return out
    except Exception as e:  # no sounddevice / no audio device / PortAudio missing
        print(f"audio off ({e.__class__.__name__}: {e}); `pip install sounddevice` (Linux: also libportaudio2) to enable")
        return None


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
         core: str | None = None, audio: bool = True, league: str = "knight", loop_first: bool = False):
    """``loop_first``: after each finish, save the race and go straight back to the league's first
    grid (Mute City I for Knight League), for back-to-back runs of one course."""
    import pyglet
    from pyglet.window import key

    s = Session(rom, core, league)
    h, w = s.frame.shape[:2]
    win = pyglet.window.Window(w * scale, h * scale, caption="F-Zero: teach the fly  (Esc = save & quit)")
    keys = key.KeyStateHandler()
    win.push_handlers(keys)
    pad = open_pad(win)
    sound = open_audio(s.game, audio)
    mapping = parse_map(pad_map)
    state = {"scale": scale, "back_was_down": False, "finished_for": 0}

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
        if loop_first and s.game.info.get("lap", 0) >= 5:
            state["finished_for"] += 1
            if state["finished_for"] > 240:  # 4 s of the results screen, then the next run
                print(f"race {len(s.races) + 1} finished, back to the grid")
                state["finished_for"] = 0
                restart()
        if sound is not None:
            sound.push(s.game.pop_audio())
        if max_frames and len(s.masks) >= max_frames:
            pyglet.app.exit()

    print(__doc__.split("Controls.")[1].split("Everything")[0])
    pyglet.clock.schedule_interval(tick, 1.0 / FZero.fps)
    pyglet.app.run()
    if sound is not None:
        sound.close()
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


def replay_all(rom: str, recording: str, on_frame=None, core: str | None = None) -> list[dict]:
    """Re-run every race in a recording here and check its sync checkpoints.

    Races with a saved start state start from it. Older recordings (without one) are replayed
    in order in one emulator, like the session that made them: each race starts where the
    previous one left off, followed by the menu macro, which is what the old restart did.
    ``on_frame(race, frame, buttons, info)`` sees every frame."""
    d = np.load(recording)
    lap_addr = int(d["check_lap_addr"]) if "check_lap_addr" in d else 0x0CF3  # older recordings
    league = str(d["league"]) if "league" in d else "knight"
    game = FZero(rom, core=core, league=league)
    power_on = bytes(game.em.get_state())
    results = []
    for race in range(int(d["n_races"])):
        if "league" in d:
            # every race of a new-style recording starts from power-on + the menu macro. Rebuild
            # that here rather than loading the stored save state: save states from a different
            # snes9x build (e.g. the Windows .dll) don't load cleanly, button inputs replay fine
            game.em.set_state(power_on)
            game.reset()
        elif f"race{race}_start_state" in d:
            game.em.set_state(d[f"race{race}_start_state"].tobytes())
            game.frame_no, game.info = 0, {}
            game._last_move = game._empty = 0
        else:
            game.reset()
        start = bytes(game.em.get_state())
        masks = d[f"race{race}_masks"]
        checks = {int(f): (int(lap), int(seg)) for f, lap, seg in d[f"race{race}_checks"]}
        mismatches = 0
        for i, m in enumerate(masks, start=1):
            buttons = mask_to_buttons(m)
            frame = game.step(buttons)
            if on_frame:
                on_frame(race, frame, buttons, game.info)
            if i in checks:
                ram = game.ram()
                if (int(ram[lap_addr]), int(ram[RAM_SEGMENT])) != checks[i]:
                    mismatches += 1
        results.append({"race": race, "start_state": start, "frames": len(masks), "checkpoints": len(checks),
                        "mismatches": mismatches, "laps": game.info.get("lap"),
                        "segment": game.info.get("segment")})
    return results


def replay(rom: str, recording: str, race: int = 0, on_frame=None, core: str | None = None) -> dict:
    """Check one race (replays the whole recording, since old files are sequential)."""
    return replay_all(rom, recording, on_frame, core)[race]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom", required=True)
    ap.add_argument("--out", default="my_races.npz")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--map", help='controller button numbers, e.g. "A=0,B=1,X=2,LB=4,RB=5"')
    ap.add_argument("--controller-test", action="store_true")
    ap.add_argument("--core", help="snes9x libretro core (.dll/.so/.dylib) instead of stable-retro")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--mute-city", action="store_true", help="back-to-back Mute City I runs")
    ap.add_argument("--league", choices=["knight", "queen", "king"], default=None,
                   help="which Grand Prix the menus pick (default knight)")
    ap.add_argument("--class", dest="klass", choices=["beginner", "standard", "expert"], default="beginner",
                   help="difficulty class (default beginner)")
    ap.add_argument("--queen-league", action="store_true", help="same as --league queen")
    ap.add_argument("--king-league", action="store_true", help="same as --league king")
    ap.add_argument("--first-race", action="store_true",
                   help="back-to-back runs of the league's first race (knight: Mute City I, queen: "
                        "Mute City II, king: Mute City III)")
    ap.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.controller_test:
        controller_test()
    else:
        play(a.rom, a.out, a.scale, a.max_frames, a.map, a.core, not a.no_audio,
             (a.league or ("queen" if a.queen_league else "king" if a.king_league else "knight"))
                    + ("" if a.klass == "beginner" else "/" + a.klass),
                    a.mute_city or a.first_race)


if __name__ == "__main__":
    main()
