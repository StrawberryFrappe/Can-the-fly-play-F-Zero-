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

* The macro ends at emulator frame ~942, during the "READY" countdown.
* The race clock starts about 60 frames later.
* We save a state at frame 942 (`--save-state start.state`) and start every experiment from it.

## WRAM map ($7E....)

Found by recording all 128 KB of WRAM every frame and filtering for values with the expected
behaviour. Offsets are from `$7E0000`.

| Address | Size | Meaning | How it behaves | Conf. |
|---|---|---|---|---|
| `$7E0B20` | u16 LE | player speed | 0 at rest; rises to ~2,000–2,150 at Blue Falcon top speed under B; decays when coasting; drops on wall hits | 🟡 (matches HUD km/h trend; the unit→km/h factor is not pinned down, roughly ÷4.6) |
| `$7E0D00` | u8 | player track segment | 0 → 58 over one lap of Mute City I (**59 segments**), resets at the finish line; decreases when driving the wrong way | ✅ |
| `$7E0CF3` | u8 | laps completed | 0 → 1 exactly when "4 LAPS LEFT" appears | ✅ |
| `$7E0F53`, `$7E10D5` | u8 | also increment at the lap line | same frame as `$0CF3`; probably lap-related copies/flags | ❓ |
| `$7E1164`, `$7E1168`, `$7E0D08` | u8 | segment-like counters | track the same range as `$0D00`, slightly offset; likely other racers or sub-positions | ❓ |
| `$7E0055` | u8 | changes 2 → 3 early in the race | probably a game-state/phase byte | ❓ |

**Lap timing reference.** A greedy search driver (always B; left/straight/right picked by
maximum speed 40 frames ahead) crossed the line at frame 2,154 after the start-line state. That
was a first lap of about 0'30"8 on Beginner, running 3rd.

**Not found yet:**
* **Energy (POWER).** No clean RAM candidate turned up. Read it from the HUD instead (below).
* **Rank, lateral position on the track, race-over flag.** Several bytes near `$0C66`/`$0C86`
  collapse at the moment of the explosion ("YOU LOST"), but they aren't identified.

## HUD pixels (frame coordinates, 256×224)

| What | Where | How |
|---|---|---|
| POWER bar | row `y = 22`, `x = 176 … 239` (64 px) | filled pixels are light pink `(248, 200, 248)`; energy = share of pixels with `R > 180, B > 180, G < 235`. White border at `x = 175` and `240` | ✅ |
| Sky colour | `(104, 144, 248)` at the top rows | handy for "is the HUD visible" checks |

Energy drains on guard-beam contact. At zero you get "POWER DOWN", then the machine explodes
and "YOU LOST" appears (about 1 s later). Afterwards `$0B20` stays stuck at **512**, not 0, so
detect the crash by track progress instead: `FZero.step` reports `done` after 600 frames without
a segment change.

| Road surface colour (Mute City I) | exactly `(144, 160, 160)` RGB. Useful for road-segmentation drivers and teachers |

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
