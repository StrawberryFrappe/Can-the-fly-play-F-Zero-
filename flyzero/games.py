"""Games the fly can play.

* ``FZero``      - the real thing: F-Zero (SNES) running in snes9x via stable-retro.
                   You supply your own ROM dump.
* ``MockRacer``  - a tiny pseudo-3D racer drawn in NumPy, F-Zero colours, same
                   buttons. Used for tests and for playing without a ROM.

Both expose ``reset() -> frame``, ``step(buttons) -> frame``, ``info`` and
``fps``. Frames are HxWx3 uint8 (224x256 like the SNES).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from .motor import BUTTONS

SNES_FPS = 60.0988


# F-Zero (USA) work RAM ($7E0000-) addresses, found by searching RAM traces
RAM_SPEED = 0x0B20    # u16, ~2000 at full speed in the Blue Falcon
RAM_SEGMENT = 0x0D00  # u8, track segment of the player, 0 .. ~58 on Mute City I, resets at the line
RAM_LAP = 0x0F53      # u8, laps completed (0..5; 5 = race finished)
RAM_POWER = 0x00C9    # u16, energy (POWER), 2048 = full; steady while the HUD bar blinks
FINISH_RANK = 3       # SAFE rank on the last lap (Knight League, Beginner)
RAM_RANK = 0x0DC8     # u8, race position (as the HUD's RANK; checked against SAFE at every lap line)


class FZero:
    """F-Zero on snes9x (stable-retro core), controlled frame by frame.

    Getting from power-on to the start line: the default ``menu`` macro skips the
    intro with START, then confirms every menu with B (Grand Prix -> Blue
    Falcon -> Beginner -> Knight League). B is also the accelerator, so extra
    presses during the countdown are harmless. For anything fancier, get to the
    grid once, save a state with ``--save-state`` and use ``--state`` later.
    """

    fps = SNES_FPS
    menu = [("wait", 240), ("START", 6), ("wait", 120), ("START", 6), ("wait", 90)] + [
        step for _ in range(8) for step in (("B", 6), ("wait", 54))
    ]
    LEAGUES = ("knight", "queen", "king")

    # Practice mode, Mute City I, one rival (the default Blue Falcon), no Grand Prix rank rule:
    # title -> PRACTICE -> car -> rival -> course select (Mute City I) -> START
    practice_menu = [("wait", 240), ("DOWN", 6), ("wait", 20), ("START", 6), ("wait", 120), ("B", 6),
                     ("wait", 60), ("B", 6), ("wait", 60), ("START", 6), ("wait", 100)]

    CLASSES = ("beginner", "standard", "expert")

    @classmethod
    def menu_for(cls, league: str = "knight") -> list:
        """Menu macro for a league, optionally with a class: "knight", "queen/expert", ... The league
        screen is up after the first B (car confirmed), the class line after the second."""
        if league == "practice":
            return list(cls.practice_menu)
        league, _, klass = league.partition("/")
        k = cls.LEAGUES.index(league)
        c = cls.CLASSES.index(klass or "beginner")
        m = list(cls.menu)
        at = m.index(("B", 6)) + 2  # after the first B and its wait
        m = m[:at] + [step for _ in range(k) for step in (("DOWN", 6), ("wait", 14))] + m[at:]
        at2 = [i for i, st in enumerate(m) if st == ("B", 6)][1] + 2   # after the second B
        return m[:at2] + [step for _ in range(c) for step in (("DOWN", 6), ("wait", 14))] + m[at2:]

    def __init__(self, rom: str | Path, state: str | Path | None = None, skip_menu: bool = False,
                 core: str | Path | None = None, league: str = "knight"):
        """``core``: path to a snes9x libretro core (.dll/.so/.dylib) to use instead of
        stable-retro, e.g. on Windows. Used automatically if stable-retro isn't installed."""
        rom = Path(rom)
        if not rom.exists():
            raise FileNotFoundError(rom)
        if core is None:
            try:
                import stable_retro
            except ImportError:
                from .libretro import find_core

                core = find_core()
                if core is None:
                    raise RuntimeError("no emulator: install stable-retro, or pass --core with a "
                                       "snes9x_libretro core (.dll on Windows)") from None
        if core is not None:
            from .libretro import Libretro

            self.em = Libretro(core, rom)
            self.data = None
            self.backend = f"libretro ({Path(core).name})"
        else:
            if rom.suffix.lower() != ".sfc":  # stable-retro picks the core by extension
                self._tmp = tempfile.TemporaryDirectory()
                dst = Path(self._tmp.name) / "fzero.sfc"
                shutil.copy(rom, dst)
                rom = dst
            self.em = stable_retro.RetroEmulator(str(rom))
            self.data = stable_retro.data.GameData()
            self.em.configure_data(self.data)
            self.backend = "stable-retro"
        self.menu = self.menu_for(league)
        self.state = Path(state).read_bytes() if state else None
        self.skip_menu = skip_menu or state is not None
        self.frame_no = 0
        self.info: dict = {}
        self.collect_audio = False
        self._audio = []

    def audio_rate(self) -> float:
        return float(self.em.get_audio_rate())

    def pop_audio(self) -> np.ndarray:
        """Game sound since the last call (stereo int16), if ``collect_audio`` is on."""
        out = np.concatenate(self._audio) if self._audio else np.zeros((0, 2), np.int16)
        self._audio = []
        return out

    def _press(self, buttons: dict[str, bool]) -> np.ndarray:
        mask = np.array([buttons.get(b, False) for b in BUTTONS], np.uint8)
        self.em.set_button_mask(mask, 0)
        self.em.step()
        a = self.em.get_audio()  # always drain, so the buffer can't grow
        if self.collect_audio:
            self._audio.append(np.asarray(a, np.int16).reshape(-1, 2))
        self.frame_no += 1
        return self.em.get_screen().copy()

    def reset(self, on_frame=None) -> np.ndarray:
        self.frame_no = 0
        self.info = {}
        self._last_move = 0
        self._empty = 0
        if self.state is not None:
            self.em.set_state(self.state)
        frame = self._press({})
        if not self.skip_menu:
            for what, n in self.menu:
                for _ in range(n):
                    frame = self._press({} if what == "wait" else {what: True})
                    if on_frame:
                        on_frame(frame)
        return frame

    def save_state(self, path: str | Path):
        Path(path).write_bytes(bytes(self.em.get_state()))

    def ram(self) -> np.ndarray:
        if self.data is None:
            return self.em.wram()
        self.data.update_ram()
        return np.frombuffer(bytes(self.data.memory.blocks[0x7E0000]), np.uint8)

    @staticmethod
    def racing(frame: np.ndarray) -> bool:
        """The race HUD is up (both edges of the POWER bar are white): not a menu/results screen."""
        return bool((frame[22, 175] >= 240).all() and (frame[22, 240] >= 240).all())

    @staticmethod
    def energy(frame: np.ndarray) -> float:
        """POWER bar fill (0..1), read from the HUD."""
        bar = frame[22, 176:240].astype(int)
        return float(((bar[:, 0] > 180) & (bar[:, 2] > 180) & (bar[:, 1] < 235)).mean())

    def step(self, buttons: dict[str, bool]) -> np.ndarray:
        frame = self._press(buttons)
        ram = self.ram()
        lap, seg = int(ram[RAM_LAP]), int(ram[RAM_SEGMENT])
        speed = int(ram[RAM_SPEED]) | int(ram[RAM_SPEED + 1]) << 8
        # POWER from RAM: the HUD bar blinks when low, so reading it gives 0, 0.15, 0, ...
        energy = min(1.0, (int(ram[RAM_POWER]) | int(ram[RAM_POWER + 1]) << 8) / 2048.0)
        # no track progress for a while = crashed out ("YOU LOST" leaves speed stuck at 512)
        if seg != self.info.get("segment") or lap != self.info.get("lap"):
            self._last_move = self.frame_no
        stalled = self.frame_no - getattr(self, "_last_move", 0)
        self._empty = self._empty + 1 if energy < 0.02 else 0  # POWER empty: exploding
        # race HUD gone for a second mid-race: "YOU LOST" (explosion or rank out)
        self._no_hud = 0 if self.racing(frame) else getattr(self, "_no_hud", 0) + 1
        out = lap < 5 and self.frame_no > 300 and self._no_hud > 60
        self.info = {"frame": self.frame_no, "lap": lap, "segment": seg, "speed": speed,
                     "energy": round(energy, 3), "stalled": stalled, "rank": int(ram[RAM_RANK]),
                     "done": lap >= 5 or stalled > 600 or self._empty > 90 or out,
                     # a real finish: all 5 laps AND within the SAFE rank at the line (3rd on the
                     # last lap); crossing it lower ends in "YOU LOST" like a crash
                     "finished": lap >= 5 and int(ram[RAM_RANK]) <= FINISH_RANK}
        return frame


class MockRacer:
    """A pseudo-3D (Mode-7-ish) racetrack with F-Zero-style handling.

    Not F-Zero, just something to drive with the same buttons: B accelerates,
    Y brakes, LEFT/RIGHT steer, L/R lean (sharper steering), A boosts. Hitting
    the guard beams bounces you back and drains energy; at zero you are out.
    """

    fps = SNES_FPS
    H, W = 224, 256
    HORIZON = 72

    def __init__(self, seed: int = 0, length: int = 4000):
        rng = np.random.default_rng(seed)
        # track = curvature per segment, smooth random bends
        raw = rng.normal(0, 1, length // 50 + 2)
        curv = np.repeat(raw, 50)[:length]
        k = np.ones(80) / 80
        self.curv = np.convolve(np.r_[curv, curv[:80]], k, "same")[:length] * 0.9
        self.length = length
        rows = np.arange(self.H - self.HORIZON)
        self.depth = 1.0 / (rows + 1) * 60  # distance of each scanline below the horizon
        self.reset()

    def reset(self, on_frame=None) -> np.ndarray:
        self.pos = 0.0
        self.x = 0.0          # lateral offset, road half-width = 1
        self.speed = 0.0
        self.energy = 100.0
        self.boosts = 0
        self.lap_time = 0
        self.info = {}
        return self.render()

    def step(self, buttons: dict[str, bool]) -> np.ndarray:
        dt = 1 / self.fps
        top = 3.0
        if buttons.get("B"):
            self.speed += 1.2 * dt * (1 - self.speed / top)
        else:
            self.speed -= 0.3 * dt
        if self.energy <= 0:
            self.speed = 0.0
        if buttons.get("Y"):
            self.speed -= 2.0 * dt
        if buttons.get("A") and self.energy > 30 and self.boosts < 3:
            self.speed += 0.8
            self.boosts += 1
        steer = buttons.get("RIGHT", False) - buttons.get("LEFT", False)
        lean = buttons.get("R", False) - buttons.get("L", False)
        seg = int(self.pos) % self.length
        self.x += dt * (1.6 * steer + 1.0 * lean) * min(self.speed, 1.5)
        self.x -= dt * self.curv[seg] * self.speed ** 2 * 0.9  # centrifugal drift
        hit = abs(self.x) > 1.0
        if hit:  # guard beam: bounce back onto the track, lose speed and energy
            self.x = float(np.sign(self.x) * 0.92)
            self.energy -= 4.0 * max(self.speed, 0.3)
            self.speed *= 0.7
        self.energy = max(self.energy, 0.0)
        self.speed = float(np.clip(self.speed, 0, top + 1))
        self.pos += self.speed * dt * 60
        self.lap_time += 1
        self.info = {
            "distance": self.pos, "speed_kmh": self.speed * 150, "energy": self.energy,
            "wall_hit": hit, "lap": int(self.pos // self.length) + 1,
            "done": self.energy <= 0,
        }
        return self.render()

    def render(self) -> np.ndarray:
        H, W, hz = self.H, self.W, self.HORIZON
        img = np.empty((H, W, 3), np.uint8)
        sky = np.linspace(0, 1, hz)[:, None]
        img[:hz] = (np.array([20, 10, 60]) * (1 - sky) + np.array([200, 90, 160]) * sky)[:, None, :]
        # skyline scrolls with the accumulated curvature so turns are visible
        seg = int(self.pos) % self.length
        heading = np.cumsum(self.curv)[seg] * 0.05
        cols = np.arange(W)
        city = (np.sin((cols + heading * 400) * 0.07) + np.sin((cols + heading * 400) * 0.023)) * 8 + 16
        for c in range(W):
            img[hz - int(city[c]):hz, c] = (40, 30, 80)

        rows = np.arange(H - hz)
        z = self.depth                            # distance ahead of each scanline
        ahead = (seg + (z * 4).astype(int)) % self.length
        # bend: offset grows with the curvature of the track ahead, quadratically with distance
        bend = self.curv[ahead] * (z / 60) ** 2 * 2.5
        scale = (rows + 1) / (H - hz) * W * 0.9   # road half-width in pixels
        centre = W / 2 + (bend * W * 0.25 - self.x * scale)
        dx = np.abs(cols[None, :] - centre[:, None]) / scale[:, None]
        stripe = np.broadcast_to(((self.pos + z * 4) // 6 % 2).astype(bool)[:, None], dx.shape)
        road = np.where(stripe[..., None], np.uint8([30, 110, 60]), np.uint8([20, 90, 50]))  # grass
        road[dx < 1.0] = [90, 90, 110]
        road[(dx < 1.0) & stripe] = [100, 100, 120]
        rail = (dx >= 1.0) & (dx < 1.12)
        road[rail & stripe] = [250, 220, 40]
        road[rail & ~stripe] = [220, 40, 40]
        road[(dx < 0.03) & stripe] = [230, 230, 230]
        img[hz:] = road
        # the car (Blue Falcon-ish)
        cy, cx = H - 28, W // 2
        img[cy:cy + 14, cx - 14:cx + 14] = [40, 80, 220]
        img[cy + 2:cy + 7, cx - 5:cx + 5] = [250, 220, 60]
        img[cy + 14:cy + 18, cx - 16:cx - 10] = [255, 140, 20]
        img[cy + 14:cy + 18, cx + 10:cx + 16] = [255, 140, 20]
        # energy bar
        e = int(np.clip(self.energy, 0, 100) * 0.8)
        img[6:10, 8:88] = [60, 20, 20]
        img[6:10, 8:8 + e] = [240, 60, 160]
        return img
