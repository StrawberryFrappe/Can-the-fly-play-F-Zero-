# F-Zero (SNES) internals: field notes for AI-plays-games projects

Everything here was found empirically while wiring a simulated fly brain into the game.
It covers the ROM, the emulator setup, memory addresses, HUD pixels, menu timings and track
data. Each entry says how it was found and how sure we are, so you can re-verify before
relying on it.

**Confidence:** ✅ verified on screen · 🟡 consistent with the data but not proven · ❓ candidate only

## ROM

| | |
|---|---|
| Title in header | `F-ZERO` |
| Region | USA (header byte `$7FD9` = `0x01`) |
| Mapping | LoROM (`$7FD5` = `0x20`), 512 KB (`$7FD7` = `0x09`), no copier header |
| SHA-1 | `d3efd32b68f1fe37a82db9d9929b7ca7cc1a3af4` |
| MD5 | `6f334790120e1fe1a972ff184d2cfc50` |
| CRC32 | `aa0e31de` |

All addresses below are for this dump. Other regions or revisions may shift them.

## Emulator setup (stable-retro / snes9x)

stable-retro ships no F-Zero integration, so we drive the core directly:

```python
import stable_retro, numpy as np
em = stable_retro.RetroEmulator("fzero.sfc")       # core picked by extension: must be .sfc
BUTTONS = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
em.set_button_mask(np.array([b in pressed for b in BUTTONS], np.uint8), 0)
em.step(); frame = em.get_screen()                   # (224, 256, 3) uint8, 60.0988 fps
state = em.get_state(); em.set_state(state)          # bytes; deterministic replays

# RAM access: attach an empty GameData and read the WRAM block
data = stable_retro.data.GameData(); em.configure_data(data)
data.update_ram()
wram = np.frombuffer(bytes(data.memory.blocks[0x7E0000]), np.uint8)   # 128 KB, $7E0000-$7FFFFF
```

`data.memory.blocks` also exposes `0x000000` (8 KB) and `0x700000` (2 KB, SRAM).
Speed with no agent attached: about **740 emulated frames/s** on one CPU core, including a RAM
read every frame.

**Portability:** button inputs replay bit-exactly across snes9x builds (Windows buildbot .dll →
stable-retro's Linux core, verified on 9 human races), but **save states do not**: a state
saved by one build loads without error in another and then desyncs immediately. Store inputs
plus a reproducible start (power-on + menu macro), not save states.

**Determinism:** replaying the same button sequence from the same save state reproduces the
same run exactly. That made search-based drivers (try each option from a save state, keep the
best) practical.

## Controls

| Button | Action |
|---|---|
| B | accelerate |
| Y | brake |
| A | super jet (boost). Available from lap 2, one per lap earned (the 3 icons bottom right) |
| L / R | lean / bank (sharper turns) |
| ← → | steer |
| START | pause (don't press it mid-race by accident) |

## Menus: power-on to the start line

This timing works from power-on with no inputs before it. It's in `flyzero/games.py` as `FZero.menu`.

| Step | Frames |
|---|---|
| wait (Nintendo logo, intro) | 240 |
| START (skip intro) | 6 |
| wait | 120 |
| START (title → mode select) | 6 |
| wait | 90 |
| 8 × (B for 6 frames, wait 54) | 480 |

The confirmations pick: Grand Prix → Blue Falcon (default) → Knight League / Beginner → Mute City I.
**B is also the accelerator**, so surplus B presses during the countdown do no harm. ✅ (verified
by a contact sheet of every 60th frame).

* Screen order behind the B presses: car spec (confirm Blue Falcon) → **league select** (cursor
  on Knight) → class → track intro. For Queen / King League, press DOWN once / twice on the
  league screen, i.e. after the first B (`FZero.menu_for("queen")`). Verified: Queen opens on
  Mute City II (orange sunset sky), King on Mute City III.
* The macro ends at emulator frame ~942, during the "READY" countdown.
* The race clock starts about 60 frames later.
* We save a state at frame 942 (`--save-state start.state`) and start every experiment from it.

## WRAM map ($7E....)

Found by recording all 128 KB of WRAM every frame and filtering for values with the expected
behaviour. Offsets are from `$7E0000`.

| Address | Size | Meaning | How it behaves | Conf. |
|---|---|---|---|---|
| `$7E0B20` | u16 LE | player speed | 0 at rest; rises to ~2,000–2,150 at Blue Falcon top speed under B; decays when coasting; drops on wall hits | 🟡 (matches HUD km/h trend; the unit→km/h factor is not pinned down, roughly ÷4.6) |
| `$7E0B70` | u16 LE | player **x** (map units) | Mute City I spans x ≈ 340 … 6190. Found by requiring √(Δx²+Δy²) per frame to track `$0B20` speed (r = 0.72, best pair in low WRAM); the (x, y) path draws the track outline | ✅ |
| `$7E0B90` | u16 LE | player **y** (map units) | Mute City I spans y ≈ 260 … 2940 | ✅ |
| `$7E0BE0` | u16 LE | player **heading** | **0xC000 units per full turn** (0 … 0xBFFF). `atan2(dy, dx) = value / 0xC000 · 2π − π/2`, residual 2° against the direction of motion on a human race. RIGHT on the D-pad increases it | ✅ |
| `$7E00C9` | u16 LE | **POWER** (energy), 2048 = full | tracks the HUD bar (r = 0.998) and stays steady while the bar blinks at low energy. Found by correlating low WRAM with the HUD reading on a race that drained to 30% | ✅ |
| `$7E0DC8` | u8 | **race position** (RANK) | the only low-WRAM byte ending at the HUD's RANK 15 on a race that ranked out at the lap-2 line (SAFE 10); the reference pilot sits at 1-3 | 🟡 |
| `$7E0D00` | u8 | player track segment | 0 → 58 over one lap of Mute City I (**59 segments**), resets at the finish line; decreases when driving the wrong way | ✅ |
| `$7E0F53` | u8 | **laps completed** | 0 → 5, +1 at each finish-line crossing; 5 = race finished. Verified on a full human race (crossings at frames 1935, 3730, 5385, 7146, 8859) | ✅ |
| `$7E0CF3` | u8 | lap-related flag, **not** a lap counter | 0→1 at the first lap line, then toggles between 0 and 1 during later laps (an earlier version of these notes got this wrong) | ❓ |
| `$7E097B`, `$7E0A23` | u8 | count up from lap 2 onward and keep counting on the results screen | not a clean lap counter | ❓ |
| `$7E1164`, `$7E1168`, `$7E0D08` | u8 | segment-like counters | track the same range as `$0D00`, slightly offset; likely other racers or sub-positions | ❓ |
| `$7E0055` | u8 | changes 2 → 3 early in the race | probably a game-state/phase byte | ❓ |

**Lap timing reference.** A human race (Blue Falcon, Beginner, no boosts to speak of) finished
5 laps in 8,859 frames (2'27"), about 30 s per lap. A greedy search driver (always B; left/straight/right picked by
maximum speed 40 frames ahead) crossed the line at frame 2,154 after the start-line state. That
was a first lap of about 0'30"8 on Beginner, running 3rd.

**Not found yet:**
* **Race-over flag** (position, heading, energy and rank: see above). The race HUD disappearing is a good proxy for "YOU LOST". Several bytes near `$0C66`/`$0C86`
  collapse at the moment of the explosion ("YOU LOST"), but they aren't identified.

## HUD pixels (frame coordinates, 256×224)

| What | Where | How |
|---|---|---|
| POWER bar | row `y = 22`, `x = 176 … 239` (64 px) | filled pixels are light pink `(248, 200, 248)`; energy = share of pixels with `R > 180, B > 180, G < 235`. White border at `x = 175` and `240` | ✅ |
| Sky colour | `(104, 144, 248)` at the top rows | handy for "is the HUD visible" checks |
| Road surface (Mute City I) | everywhere on the track | exactly `(144, 160, 160)`: segment the road with a colour match. Useful for road-following drivers and training targets | ✅ |

Energy drains on guard-beam contact. At zero you get "POWER DOWN", then the machine explodes
and "YOU LOST" appears (about 1 s later). Afterwards `$0B20` stays stuck at **512**, not 0, so
detect the crash by track progress instead: `FZero.step` reports `done` after 600 frames without
a segment change.

## Race rules that matter for agents

* **Guard beams (the green bumpers)** bounce you back and drain energy. Constant scraping kills
  you in 20–25 s. This is what ended both our first fly run and the greedy search driver.
* **Pit area** (the pink zone on the main straight) refills energy while you drive over it.
* The **SAFE** indicator under RANK is the worst position you may hold at the end of each lap.
  It tightens every lap (15 on lap 1 on Beginner), and finishing a lap below it retires you
  ("RANK OUT"). A slow but safe driver can still lose this way.
* **"REVERSE"** flashes when you drive the wrong way. `$0D00` counts down then.

## Useful code in this repo

* `flyzero/games.py` → `FZero`: emulator wrapper, menu macro, save/load state, telemetry (`info` =
  frame, lap, segment, speed, energy, stalled, done).
* `flyzero/tune.py` → `Progress`: unwrapped track progress from `$0D00` (wrong-way driving counts
  negative). A good fitness signal for RL or evolution.

## Save data and the class menu

| Address | Meaning | How found | Conf. |
|---|---|---|---|
| `$7F49FA` (work RAM) | Master class unlocked (`0xFF`: MASTER appears on the class screen for every league). The game keeps a working copy of its save RAM at `$7F4800`; this is save byte `0x1FA`. Writing the battery RAM itself does nothing until the game reloads it | brute force (each byte set to `0xFF` on the car screen, class line checked after three DOWNs), then confirmed on both the stable-retro and the libretro frontend | ✅ (screen) |

The class line cycles BEGINNER → STANDARD → EXPERT (→ MASTER when unlocked) with DOWN. The class
list is built when the league/class screen opens, so a patch must happen before that (the car
screen works). `FZero.menu_for("knight/master")` does this (`--class master` in `flyzero record`).

**F-Zero's race clock** adds 1.5 hundredths of a second per frame (0.9 × real time at 60.1 fps):
race time = (frames since GO) × 0.015 s. GO is 102 frames after our `start.state`.
