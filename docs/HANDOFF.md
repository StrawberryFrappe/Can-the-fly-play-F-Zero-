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

## Where things stand (laptop session, 2026-09-23/24)

### Mute City I is finished, by the augmented fly

| Driver | Mute City I, Grand Prix, Knight League, Beginner |
|---|---|
| **augmented fly** (natural fly + implant) | **finishes top 3 in 4 of 16 races** (2nd, 2nd, 3rd, 3rd); first finish 3rd, 2'35"34 |
| augmented fly, Practice mode (no rank rule) | finishes 11 of 16 races |
| natural fly (its own synapses only) | best drive 3 laps; about 1 lap on average; never finished |
| CNN, DAgger with the same pilot | best 157 segments (2.7 laps) |
| CNN on the owner's races | 3 segments |
| pilot (teacher, reads RAM) | 2nd, 2'20"61 |
| owner (human), 6 Beginner races | 1st every time, 2'10"-2'13" (laps ~1,690 frames) |

A finish means top 3 at the final line: the last lap's SAFE rank is 3, and crossing lower ends in
"YOU LOST". Replays (bit-exact from `start.state`) are in `runs/results/milestones/`; the
results screen of the first finish is `GP_BEGINNER_FINISH_results_screen.png`.

**The augmented fly is not the fly alone.** An implant (`flyzero/implant.py`) does the
decoding:
* **Read:** electrodes on 2,918 of the natural fly's neurons: the DNs' inputs, visual projection
  neurons and the most steering-related L2 neurons, no descending neurons. No pixels, no RAM.
* **Decide:** a small MLP trained by DAgger + DART from the pilot.
* **Write:** a current clamp drives the fly's own descending neurons, giant-fiber pulses fire the
  boost, and the same readout presses the buttons, tapped like the pilot's.

**The natural fly** (`python -m flyzero.school dagger`) learns only in FlyWire synapses:
* the motor DNs' inputs and one layer up, with a retrograde error through its own synapses;
* intrinsic excitability of the DNs;
* consolidated (slow) weights for exams.

Its best checkpoint is `work/r3g/fly0_f800000.npz` (not committed, 20 MB). It plateaus at about
1 lap; the SAFE rank needs a pilot-like pace.

### What the owner's observations turned out to be (README 20)

Blinking energy bar → POWER from RAM; barely leaning → pilot leans; side to side → teacher taps
but the readout holds (hold vs tap); boost after the line (owner's tip) → boost labels on the
straight in every lap.

### Big Blue (next track), in progress

* `bigblue.state`: the Big Blue grid, reached by replaying the first Mute City finish and pressing
  through the results.
* `runs/pilot/big_blue_line.npz`: racing line from the owner's GP recording, 84 segments per lap,
  with pilot settings; the pilot wins Big Blue.
* The augmented run `work/bb1` gets about 1.5 laps so far.

### Recording flags for the owner

`flyzero record --league knight|queen|king [--first-race]` (back-to-back first races),
`--king-league`, `--queen-league`.

## Where things stood at the end of the cloud session (history)

### Nothing finishes a lap yet

Exam = drive Mute City I alone from the start line for 60 s. The score is track segments gained
(59 per lap).

| Learner | Result |
|---|---|
| untrained fly (corrected brain, bias drive) | 1–1.7, stalls or crashes in < 20 s |
| fly, exact-tap lessons, 5 races, 1 epoch | −2 … 4 (one lucky 20 on a single-race run) |
| CNN baseline, 5 races | 3, then stuck |
| fly, intent lessons, all 10 races, 1 epoch (3 drives averaged) | **best fly 0 (eta 3e-4): mean 4.7, best 7**; others 0.3–2.7. Error vs teacher 9–10 Hz (taps gave 12–18) |
| + 5 practice drives | no clear change (fly 0: 3.7, fly 1: −0.3) |
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

### State at handoff

`teach` over all 10 races finished its lessons epoch: 4 flies with learning rates 3e-4, 1e-3,
3e-3 and 1e-2, intent lessons, mirroring, racing frames only. It was stopped after the first 5
of 40 practice drives. All weights and logs are in `runs/results/3_intent_all/`:
* `taught_fly{k}_epoch1.npz`: after the lessons;
* `taught_fly{0,1}_practice5.npz`: after 5 practice drives.

The most promising starting point is **`taught_fly0_epoch1.npz`** (lowest learning rate, best
exam). Resume with practice only (below).

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

## Continuing (GPU fleet, laptop)

```
# natural fly: DAgger with the pilot, deep plasticity, consolidation (see README 12-21)
python -m flyzero.school dagger --init work/r3g/fly0_f800000.npz --batch 4 --etas 3e-4 --eta-deep 0.01 \
    --eta-bias 5e-5 --consolidate 20000 --focus 3 --p-grid 0.3 --drives 8 --out work/n7
# augmented fly (implant): continue a run, or --eval N to only race
python -m flyzero.implant --host work/r3g/fly0_f800000.npz --batch 4 --l2-top 2000 --host-drives --taps \
    --dart 0.5 --resume work/aug15 --out work/aug16
python -m flyzero.implant ... --resume work/aug15_src --eval 16 --exam start.state --out work/eval
# Big Blue: --line runs/pilot/big_blue_line.npz --exam bigblue.state
# watch any saved race in 3D:   python -m flyzero.live --replay runs/results/milestones/<file>.npz
```

Two processes (4 slots each) fit a GTX 1650 + 7 GB RAM. Watch GPU memory: a second-layer natural
run (1.8 GB) next to a large implant OOMs.

## Continuing (the cloud-session commands, CPU)

```
# the fly: practice from the taught weights (one process per learning rate; use as many as you have cores)
flyzero teach --rom "F-Zero (USA).sfc" --lessons lessons_all.npz --exam start.state --epochs 0 --practice 40 --etas 1e-3,1e-3,1e-3,1e-3 --init runs/results/3_intent_all/taught_fly0_epoch1.npz --out taught_practice

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
   done. The look-ahead search instructor (`tools/search_teacher.py`) managed ~1.5 laps; it would
   need improving (roll-outs with a better base policy) to label reliably.
3. **Plasticity one layer deeper**: synapses onto the ~4.5 k neurons feeding the motor neurons.
   Still the fly's own wiring, more capacity. The information ceiling there is higher (VPN
   r ≈ 0.41).
4. **Teach boost** (A → giant fiber DNp01), and find **landing-response descending neurons** for
   DOWN on jump plates (the owner holds DOWN in the air to land smoothly).
5. **More human laps** (`flyzero record --mute-city`). The owner is willing; the Queen League
   (`--queen-league`) was recorded once, messy but real.

## Gotchas that cost time

* **Replays need one empty frame after loading the state** (`live._start`), like
  `EmulatorPool.load` does; otherwise they desync.
* **`pkill -f` kills your own shell** (the command line matches). Kill by exact PID.
* **Spawned workers re-import the main module:** no module-level CuPy/torch in scripts that spawn
  emulator workers (320 MB per worker otherwise).
* **High learning rates make training look good**: the plasticity steers online. Judge by frozen
  (consolidated) weights.
* **Teacher and student need the same hands** (hold vs tap).
* **Implant data collected while the implant drives teaches it to read its own commands**
  (causal confusion): collect with the host fly driving (`--host-drives`) or the pilot (`--dart`).

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
