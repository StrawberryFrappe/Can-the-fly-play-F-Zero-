# Handoff: where the project stands, and how to continue on a new machine

Written when the work moved from a 4-core, no-GPU cloud container to the owner's laptop (with a
GPU). Read this first, then `README.md` (the lab notebook) and `docs/fzero-internals.md`.

## The goal, and the owner's rules

* **Goal:** a simulated whole fruit-fly brain (FlyWire connectome) that **finishes Mute City I**
  (5 laps, Knight League, Beginner) in F-Zero on the SNES.
* **"A cyborg fly, not a robot fly."** Keep the fly doing the driving. The order agreed with the
  owner:
  1. Pure fly.
  2. Biological corrections.
  3. Learning inside the fly's own synapses.
  4. Only then more "trickery".
  * **No shared control / autopilot crutch.** The owner explicitly rejected it.
  * Always say plainly how much is the fly and how much is ours.
* **Comparison:** a standard neural network (`flyzero baseline`) trained on the same human races
  and given the same exam, "fly vs AI".

## Where things stand (end of the cloud session)

### Nothing finishes a lap yet

Exam = drive Mute City I alone from the start line for 60 s. The score is track segments gained
(59 per lap).

| Learner | Result |
|---|---|
| untrained fly (corrected brain, bias drive) | 1–1.7, stalls or crashes in < 20 s |
| fly, exact-tap lessons, 5 races, 1 epoch | −2 … 4 (one lucky 20 on a single-race run) |
| CNN baseline, 5 races | 3, then stuck |
| fly, intent lessons, all 10 races, 1 epoch | **was finishing when the session ended**, see below |
| CNN rematch, all 10 races + mirroring | stopped at epoch 7/10 (held-out steer ≈ 50%); rerun on GPU, takes minutes |

### Key findings (details in the README lab notebook)

* **The plain Shiu et al. model seizes under whole-field vision.** It counts about 2 M
  dopamine/serotonin/octopamine synapses as fast excitation, and the NT predictor labels Kenyon
  cells dopaminergic. `flyzero/biology.py` corrects this. Always use `corrected(conn)`.
* **The fly sees the road** (visual projection neurons ≈ pixel-level information), and in the
  corrected brain DNa02 left−right correlates with the road's position (r ≈ 0.33). **But its
  instinct points at high-contrast edges (the walls).**
* **The fly's steering inputs can at best predict human steering at r ≈ 0.36** (linear,
  cross-validated). That's a moderate ceiling.
* **Poisson "flight drive" was pure noise** on one-neuron-per-side motor outputs. Use
  `Brain.set_bias` (a steady depolarisation, about 7.8 mV, just under threshold).
* **Lessons lower the error against the teacher** (≈20 → 12 Hz), **but the solo exam doesn't
  improve.** That's compounding error, as the owner predicted: copying exact taps derails on the
  first mistake. Remedies in progress:
  * teach **intent** (smoothed ~0.25 s), not taps;
  * then **practice**: reward learning while driving solo.
* **Owner's driving style** (from the data):
  * lean-heavy turning (L/R on 28% of frames vs D-pad 19%), often lean alone;
  * gas feathering in hard turns (gas off 53% of the time, about 4.5 taps/s);
  * DOWN held on the Mute City jump (the fly has no DOWN neuron yet);
  * boost once per lap on straights (only reflexive for the fly: giant fiber);
  * never brakes;
  * hugs the walls.
* **Exam noise is large.** The same fly swings −4 … 20. Average several drives (the exam now does 3).

### What was running at handoff

`teach` over all 10 races: 4 flies with learning rates 3e-4, 1e-3, 3e-3 and 1e-2, intent
lessons, mirroring, racing frames only, then 40 practice drives each. The lessons epoch (about
3.3 h on 4 cores) was meant to finish and its weights (`taught_fly{k}_epoch1.npz`) be copied
into `runs/results/3_intent_all/` before the container was stopped. **Check that folder.** If
the weights are there, resume with practice only (below). If not, rerun the lessons.

## Setting up the new machine

1. **Python 3.10–3.12, 64-bit.** From the repo folder:
   ```
   pip install -e .
   pip install numba cma
   ```
   numba makes the brain about 6× faster (required in practice); cma is for `flyzero tune`.
2. **PyTorch with CUDA** for the CNN baseline. Pick the CUDA build matching the driver, e.g.
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
   `flyzero baseline` prints `device: cuda` when it works.
3. **Emulator:**
   * Linux/macOS: stable-retro is installed by `pip install -e .`.
   * Windows: put `snes9x_libretro.dll` (plain "Snes9x", x86_64) in the repo folder or pass
     `--core`. See the README.
4. **ROM:** `F-Zero (USA).sfc` (SHA-1 `d3efd32b…`) in the repo folder. Never commit it.
5. **Connectome:** `flyzero download` (about 135 MB, into `~/.cache/flyzero`).
6. **Local start line + lessons.** These are machine-specific: save states don't transfer between
   snes9x builds, but inputs replay exactly.
   ```
   flyzero start-state --rom "F-Zero (USA).sfc" --out start.state
   flyzero lessons --rom "F-Zero (USA).sfc" runs/recordings/01_mute_city.npz runs/recordings/02_knight_gp_first3.npz runs/recordings/03_mute_city.npz runs/recordings/04_mute_city_x4.npz runs/recordings/05_queen_league_gp.npz --out lessons_all.npz
   ```
   `lessons` replays every race and skips any with checkpoint mismatches. All 10 races (about
   145 k frames) replayed with 0 mismatches in the cloud container.

## Continuing

```
# the fly: practice from the taught weights (one process per learning rate; use as many as you have cores)
flyzero teach --rom "F-Zero (USA).sfc" --lessons lessons_all.npz --exam start.state --epochs 0 --practice 40 --etas 1e-3,1e-3,1e-3,1e-3 --init runs/results/3_intent_all/taught_fly1_epoch1.npz --out taught_practice

# or the full thing from scratch (lessons: ~3 h per epoch on 4 cores; faster with more)
flyzero teach --rom "F-Zero (USA).sfc" --lessons lessons_all.npz --exam start.state --epochs 1 --practice 40 --out taught_all

# the comparison network (GPU: minutes)
flyzero baseline --rom "F-Zero (USA).sfc" --lessons lessons_all.npz --exam start.state --epochs 10 --out baseline_all
```

Logs: `<out>/teach_log.jsonl` (one line per epoch / practice drive, with exams every 5 drives).

### The GPU

Only the CNN uses it. The fly simulation is CPU-bound: each learner is one process, so it
scales with cores, not with the GPU. A GPU port of `Brain.run` is possible (15 M synapses fit
easily), but it's work, and the emulator steps once per frame anyway.

## Ideas not tried yet (roughly in order of promise)

1. **Practice results.** If reward practice after intent lessons helps, run it longer / tune
   `--reward-eta`.
2. **Instructor corrections (DAgger).** The fly drives, and a teacher labels what it should have
   done. The look-ahead search instructor in the old scratch scripts managed ~1.5 laps; it would
   need improving (roll-outs with a better base policy) to label reliably.
3. **Plasticity one layer deeper**: synapses onto the ~4.5 k neurons feeding the motor neurons.
   Still the fly's own wiring, more capacity. The information ceiling there is higher (VPN
   r ≈ 0.41).
4. **Teach boost** (A → giant fiber DNp01), and find **landing-response descending neurons** for
   DOWN on jump plates (the owner holds DOWN in the air to land smoothly).
5. **More human laps** (`flyzero record --mute-city`). The owner is willing; the Queen League
   (`--queen-league`) was recorded once, messy but real.

## Gotchas that cost time

* **One emulator per process** (stable-retro and the libretro frontend both). Use subprocesses.
* **`$7E0CF3` is not the lap counter** (it's a flag). **`$7E0F53`** is.
* **After "YOU LOST", speed reads 512, not 0.** Detect crashes by track progress (`FZero.step`
  does).
* **`pgrep -f "some text"` inside a shell loop matches its own command line.** A launcher waited
  forever because of this.
* **The POWER-bar reading flickers by 1 px.** `learning.Reward` counts damage as new lows only.
* **Old multi-race recordings** (made before start states were stored) must be replayed in
  order; `replay_all` does that. The old in-recorder "restart" ran the menu macro from wherever
  the game was, which advanced the Grand Prix.
