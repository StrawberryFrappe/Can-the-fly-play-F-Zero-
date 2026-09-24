# Can the fly play F-Zero?

A spiking simulation of the **whole *Drosophila* brain** (FlyWire v783 connectome:
138,639 neurons, 15.1 M connections, 54.5 M synapses) at the controls of **F-Zero on the SNES**.

Short answer: **yes, badly.** It holds the throttle, steers from what it sees, pinballs
between the guard beams, sometimes drives the wrong way for a bit, and now and then its
escape neuron (the giant fiber) fires the super jet.

```
 game frame ──► motion detectors ──► 138k-neuron LIF brain ──► descending neurons ──► SNES pad
 (snes9x)       T4/T5 (12,246)       FlyWire connectome        DNa02 / DNp09 / ...   (B, ←, →, A…)
```

## Is this a real fly?

No. No living animal is involved, and the program is not a fly. Here is what it actually is.

**Where the data comes from.** In a study published in 2018, researchers imaged the brain of a
single adult female fruit fly with an electron microscope. The fly was a laboratory specimen, and its brain had been
chemically preserved and cut into about 7,000 ultrathin slices, a standard procedure in
neuroanatomy ([Zheng et al. 2018](https://doi.org/10.1016/j.cell.2018.06.019)). Over several
years the [FlyWire](https://flywire.ai) consortium traced every neuron and synapse in those images
([Dorkenwald et al. 2024](https://doi.org/10.1038/s41586-024-07558-y)). The result is a
*connectome*: a table of which neuron connects to which, and through how many synapses. It is a
wiring diagram. Structurally it is a spreadsheet, closer to a street map than to a living thing.

**What the program does with it.** Each of the 138,639 neurons becomes a *leaky integrate-and-fire*
unit: a single number (a voltage) that rises when it receives input, slowly leaks back, and "fires"
when it crosses a threshold. A firing unit adds a small amount to the voltage of the units it
connects to, in proportion to the synapse counts in the table. That is the entire model. The
program repeats this update for every unit, 10,000 times per simulated second, using ordinary
array arithmetic.

**What it leaves out.** Compared with a real nervous system, the model has:

* no body, muscles, senses or environment, apart from the game image we feed in and the buttons we read out;
* no neuromodulators (dopamine, serotonin, octopamine), hormones or metabolism;
* no learning or memory: connection strengths never change (the optional reward-learning
  experiment, step 7 below, changes only ~7,300 synapses onto four motor neuron types);
* no dendritic computation, no gap junctions and no graded (non-spiking) signalling. Much of early
  fly vision relies on graded signalling, which is why we compute that stage separately;
* neuron properties that are identical for all 138,639 cells, whereas real neurons differ widely.

**Why it still works a little.** Wiring alone captures a surprising amount of how information
flows. [Shiu et al. 2024](https://www.nature.com/articles/s41586-024-07763-9) showed that this
exact model predicts which neurons take part in feeding and grooming behaviours, and experiments
in real flies confirmed the predictions. In this project the same wiring routes visual motion to
the turning neurons and looming stimuli to the escape neuron, as in the animal. Those are
properties of the circuit diagram, in the same way a road map tells you which towns are connected
without containing any traffic.

**Could it feel anything?** There is no scientific basis for thinking so. The simulation lacks
nearly everything that makes a nervous system part of a living organism: a body to keep alive,
internal states, chemistry, and the ability to change with experience. It is a numerical
experiment on a diagram, of the same kind as a weather or traffic simulation, and it stops existing
in any meaningful sense when the script exits. Questions about the moral status of future, far more
complete brain emulations are real and worth discussing. This project is nowhere near that
territory.

## How it works

| Stage | What | Where |
|---|---|---|
| Game | F-Zero (USA) on snes9x via stable-retro, stepped one frame at a time. Each game frame = 16.6 ms of neural time, so the game waits for the brain. | `games.py` |
| Eyes | Hassenstein–Reichardt motion detectors (ON/OFF, 4 directions) drive the fly's own **T4/T5** neurons at their retinotopic positions | `vision.py`, `eyes.py` |
| Brain | Leaky integrate-and-fire model of every neuron. Weights = signed synapse counts. Same model and parameters as [Shiu et al. 2024](https://www.nature.com/articles/s41586-024-07763-9) | `brain.py`, `connectome.py` |
| Hands | Descending neurons → buttons (below) | `motor.py` |

| Descending neuron | In the fly | In F-Zero |
|---|---|---|
| DNa02 left/right | turning towards that side | ← / → |
| DNa01 left/right | turning | L / R lean |
| DNp09 (P9) | forward walking | B, accelerate |
| MDN | backward walking ("moonwalker") | Y, brake |
| DNp01, giant fiber | escape jump | A, super jet |

### What's the fly's and what's ours

* **The fly's:** everything from T4/T5 onwards: lobula plate, lobula, all visual projection
  neurons, the central brain and the descending neurons. Nothing downstream is tuned or trained.
* **Ours:**
  1. The front end up to T4/T5, computed with a standard motion-detector model (see below).
  2. Which descending neuron presses which button.
  3. A tonic 60 Hz Poisson drive to DNp09, like optogenetically activating a fly that wants to
     walk. Otherwise the car never moves. Turn it off with `--drive none`.

### The road to finishing Mute City: a lab notebook

Goal: finish Mute City I (5 laps) while changing the fly as little as possible. Each step
below was taken only after the previous one was shown not to be enough. F-Zero memory
addresses, HUD pixels and menu timings are in [docs/fzero-internals.md](docs/fzero-internals.md).

1. **Calibrate the interface only** (`flyzero tune`). A CMA-ES search over the interface
   settings:
   * visual gain and how much of the upper screen the fly sees;
   * steering threshold, bias and smoothing;
   * the walking drive.

   The brain is untouched. **Result:** best 17 of 59 segments in 30 s, still crashing. The
   search mostly learned to cancel a constant left pull.
2. **Is the road in the brain at all?** Replaying footage open-loop, I decoded the road's
   position on screen (from its exact colour) from neural activity, cross-validated in
   contiguous time blocks:

   | Signal | r |
   |---|---|
   | raw pixels (linear ceiling) | 0.45 |
   | visual projection neurons | 0.41 |
   | all active descending neurons | 0.32 |
   | **DNa02, the fly's steering pair** | **≈ 0** |

   **The fly sees the road, but its steering neurons don't use it.** It is a fly, not a driver.
   (A first attempt used a search driver's jittery button presses as the target. Even pixels
   couldn't predict those, which was the tell that the target was bad, not the brain.)
3. **Better eyes.** Added medulla columnar channels (Mi1/Tm3 ON, Tm1/Tm2/Tm4 OFF, Mi4/Mi9/Tm9
   sustained contrast) and colour (R7 ← blue as a UV proxy, R8 ← green). Little change in
   road information.
4. **The model seizes.** Under strong whole-field input the plain model tips into a
   self-sustaining whole-brain seizure: about 23,000 spikes per frame that continue after the
   input is switched off, with most Kenyon cells firing. Some of the earlier videos, including
   the 60 s run and its constant hard left steering, were largely a seizing brain.
5. **Biological corrections** (`flyzero/biology.py`, each with a literature reference):
   * dopamine, serotonin and octopamine synapses (about 2 M) have no fast effect, because these
     act only through GPCRs in flies;
   * Kenyon cells are cholinergic, not dopaminergic as the transmitter predictor labels them;
   * KC→KC synapses are inhibitory (mAChR-B).

   **No seizures at moderate input.**
6. **Dopamine learning in the mushroom body**, the fly's own reward system (the user's idea:
   reward speed, punish crashes, punish reversing even harder). Blocked: the mushroom body's
   visual input comes through the colour pathway (aMe12, MTe cells), and even with colour eyes
   each visual Kenyon cell gets about 0.3 mV per input spike against a 7 mV threshold. They
   never fire without re-igniting the seizure.
7. **Reward learning at the motor output** (`flyzero learn`, `flyzero/learning.py`). The first
   step that changes the brain:
   * only the ~7,300 existing FlyWire synapses onto DNa02, DNa01, DNp09 and MDN are plastic,
     with transmitter sign kept and strength capped;
   * the rule is exploratory-Hebbian reward modulation (Hoerzer, Legenstein & Maass 2014);
   * the reward is +20 per track segment, −60 per segment backwards, and −100 per energy bar
     lost.

   Everything upstream stays FlyWire-exact, with the corrections from step 5.

8. **Why learning stalled: the corrected fly never steered.** With the corrections, DNa02,
   DNg02 and DNa01 sit at 0 Hz in closed loop, so the fly holds the throttle and grinds along
   the rail. A reward rule can't learn from a silent neuron. A symmetric tonic "flight" drive
   (`flight_hz`) makes them responsive.
9. **Good news and bad news in the corrected brain.** On a clean driving lap, DNa02 left−right
   now correlates with the road's position (r = 0.33; ≈ 0 before the corrections). But the sign
   is backwards for driving: with the road on the right, the fly turns left, toward the
   high-contrast walls. That's plausibly real fly behaviour: flies fixate and approach
   high-contrast edges and objects. "Mirror goggles" (`flip`: swapping left and right between
   brain and pad) didn't rescue it either. Every combination of flip and flight drive made
   1–5 segments of progress in 40 s.
10. **Heat as a lane-departure sense doesn't work in this model.** Heating either antenna
    (TRN_VP2), cooling or humidity sensors all push *left* DNa02 up by about 100 Hz, whichever
    side is stimulated. There's no side-specific avoidance to hook into. Many different inputs
    converge on the left DNa02.

11. **Learning from the owner's races** (`flyzero record` → `lessons` → `teach`). 10 human races
    (about 145 k frames: 7× Mute City I, Big Blue, Sand Ocean, the Queen League Grand Prix)
    teach the motor synapses by instructed learning (delta rule, Dale's law kept).
    * **Exact taps** made the fly match the teacher better in lessons (error 20 → 12 Hz) but not
      drive better solo. That's compounding error: one slip and it's somewhere the teacher never was.
    * **Smoothed intent** (~0.25 s) is matched much better (9–10 Hz). The best fly averages
      4.7 segments (best 7) of 59, still far from a lap.
    * **A standard CNN on the same races does no better** (3 segments, below-baseline held-out
      accuracy with 5 races).

12. **A GPU fleet** (`flyzero/gpu.py`, `fleet.py`). The same LIF model runs many flies at once
    on a laptop GPU (GTX 1650), spike-for-spike identical to the CPU kernel without noise. Each
    fly keeps its own copy of the plastic synapses. With one emulator per subprocess, 8 flies
    drive at ~220 game frames/s in total (the CPU did 25 per core).
13. **A pilot to learn from** (`flyzero/pilot.py`). The car's position and heading are in RAM
    (`$0B70`, `$0B90`, `$0BE0`, see the internals doc). A scripted pilot follows the owner's
    fastest lap and **finishes all 5 laps** (2'46" from our start line; the owner's best is
    2'25"). **It is not the fly and never drives the fly's car.** It only says what it would do
    from wherever the fly is. That's the DAgger instructor the owner's races couldn't be: human
    races only show what to do where a human was.
14. **Where the steering information is.** The pilot drove and the taught fly watched. Its
    steering was then decoded from its neurons (ridge regression, cross-validated in time
    blocks):

    | Neurons (active ones) | r |
    |---|---|
    | L1: direct inputs of the motor DNs (643) | 0.59 |
    | visual projection neurons (481) | 0.65 |
    | L2: inputs of L1 (15,931) | **0.83** |

    **The brain has the information; the motor synapses can't reach enough of it.**
15. **DAgger on the motor synapses** (`python -m flyzero.school dagger`). The fly drives and the
    pilot labels. Curriculum starts are spread along the lap, and after a crash the drive resumes
    a few seconds earlier (training only; exams always start from the grid). **No gain:** after
    600 k fly-frames the error against the pilot stayed at 25 Hz and exams got worse.
    (`runs/results/4_dagger_dn`)
16. **Plasticity one layer deeper** (`batch.BatchInstructDeep`). The ~1.3 M FlyWire synapses onto
    the DNs' 4,559 presynaptic partners also learn. Each such neuron gets the error of the DNs it
    synapses onto, weighted by its own synapses: a retrograde "your targets should fire more /
    less" signal, which amounts to one step of backpropagation through the fly's own wiring.
    Dale's law and caps apply as before.
    * **Result:** the error fell from 25 to 13 Hz in 600 k frames, and crash restarts halved.
    * One fly drove 24 segments, and with its weights half of the exam drives now reach about
      30 segments (half a lap).
    * **Failure modes:**
      * a one-sided steering bias (the track mostly turns one way), now trained against with
        the mirror world;
      * the giant fiber runs away to 300 Hz (so it is instructed too, quiet unless boosting);
      * wall scraping that drains the energy.
17. **Three things that were ours, not the fly's.**
    * **The pilot taught "hold left" at the start.** During the READY countdown the car can't
      move, but it sits off the racing line, so the pilot's label said full left + lean for 6 s.
      The flies learned it, held LEFT through the countdown and hit the wall at GO. Now the
      label is neutral while the car stands. This one fix took the mean exam from about 6 to
      about 45 segments.
    * **Learning rates high enough to steer.** With about 0.15 mV of change per synapse per
      frame, the plasticity itself acted like a controller during training. The weights chased
      the last few seconds of labels, so training error looked good but a frozen snapshot
      didn't generalise.
      * Lowering the rate 10× made that visible: the training error went *up*.
      * Cure: **consolidated weights**, a slow average of each synapse (fast/slow synapses,
        cf. Benna & Fusi 2016). Exams drive with the slow copy.
    * **Replays were off by one frame**, which made a good drive look like a crash.
18. **Intrinsic plasticity and a hard-corner curriculum** (still the fly's own neurons). Each
    instructed DN's excitability follows its error within ±8 mV; it counters the connectome's
    own left/right DNa02 asymmetry. Most crashes happened at one sharp corner (segment ~40), so
    training drives now start more often just before recent crash spots, like an instructor
    drilling a bend.
19. **Fly vs AI** (exam = solo from the grid, full race):

    | Driver | What it learned from | Best drive |
    |---|---|---|
    | untrained corrected fly | — | 1–2 segments |
    | fly, human lessons only (step 11) | owner's races | 7 |
    | CNN, behaviour cloning | owner's 10 races | 3 |
    | CNN, DAgger | pilot (same teacher as the fly) | 157 (2.7 laps) |
    | **fly, deep plasticity + DAgger** | owner's races, then the pilot | **178 (3 laps)** |
    | pilot (reference; reads RAM) | owner's fastest lap | 5 laps |

    With the same teacher, the fly and a standard network are about even so far. Neither
    has finished: both run out of energy from wall contact.
20. **What the owner saw when watching it drive live**, and what it turned out to be:
    * *"The energy bar blinks when low, so your readings go 0, 15, 0…"* Our fault: DAgger
      treated a blinking bar as a crash. POWER is now read from RAM (`$00C9`).
    * *"It barely leans."* The pilot only leant when its steering saturated; you lean on 28% of
      frames. It now leans from half steering, and it boosts right after the lap line, where a
      boost is earned (also the owner's tip).
    * *"It moves side to side for no reason."* Also ours:
      * The pilot *taps* and its steering intent is continuous, but the fly's readout *holds* a
        button whenever one side wins. Trained on the pilot's intent, the flies steered on
        ~95% of frames and leant on ~85%.
      * Teaching hold-style labels instead (full steer only when the pilot's |intent| > 0.5)
        is what a button-holding pilot does, and that pilot still finishes. The natural fly
        didn't adapt to the relabelling, though.
    * *"When things go wrong it doesn't know what to do."* Training rewound every crash
      (including fake ones from the blinking bar), so it rarely practised recoveries.
    * One more thing the logs showed: **rank**. The SAFE position tightens every lap (15, 10,
      …, 3 on the last). A drive that is merely alive gets "YOU LOST" at a lap line for being
      too slow. Several drives ended at exactly 119 segments (the lap-2 line).
21. **The natural ladder, finished as agreed:**
    * Reward practice (rung 1) on top of the DAgger fly: harmful (the smallest rate kept 42
      segments mean, the others collapsed to 0–5).
    * Tap-rate readout: understeer (18–28 mean).
    * Readout threshold recalibration: 10 Hz is best (25/20: 41.5; 40/30: 26.4; 60/40: 14.6).
    * Plasticity **two layers deep** (10.3 M synapses onto 83,700 L2 neurons): unstable so
      far.
    * **The natural fly's result: 3 laps in its best drive, about 1 lap on average, never a
      finish.** (It keeps training.)
22. **The augmented fly** (`flyzero/implant.py`). A separate, clearly labelled model, per the
    owner's rule: *"if you do it must be a distinctively different model"*.
    * **Host:** the natural fly, unchanged.
    * **Electrodes:** read the smoothed rates of up to ~4,900 of its neurons (the DNs'
      presynaptic partners, visual projection neurons, and the L2 neurons most related to
      steering). They read no pixels and no RAM.
    * **Implant:** a small MLP on the GPU, trained by DAgger with the pilot as instructor.
    * **Current injection** into the fly's own descending neurons, through a closed-loop
      clamp. The DNs still spike, and the same readout presses the buttons.

    What mattered, in order:

    | Change | Mean segments per exam drive |
    |---|---|
    | L1 + VPN electrodes (1,066) | 42 → 90 over 3 rounds |
    | + L2 electrodes (3,066) | 95 (round 0), then *worse* as the implant drove |
    | − electrodes on descending neurons (reading its own commands: causal confusion) | 155 |
    | + new data only while the host fly drives | 161 |
    | + tap readout: continuous steer/lean, the pilot's own hands | 166 → **220** |

    **In its best exam, 2 of 8 drives completed all 5 laps.** But they crossed the line 7th
    and 6th, and the game's SAFE rank for the last lap is 3rd: **"YOU LOST"**. The pilot
    finishes 2nd (2'20"61, the owner's best is 2'25"). The augmented fly has driven the full
    distance of Mute City I, but it hasn't *finished* it in the game's sense yet. It's about
    15% too slow.

### Earlier findings

* **Photoreceptors don't work as an input.** Driving all 10,582 photoreceptors activates about
  2% of the optic lobe and **zero** descending neurons. Photoreceptors and lamina cells are
  graded, non-spiking cells with inhibitory histamine synapses, and a spiking count-weighted
  model can't carry that. Normal, mirrored and dark footage gave identical steering, so the
  fly was driving blind (`--vision retina` reproduces this).
* **Fed in at T4/T5, the connectome reproduces known fly biology:**
  * One-sided front-to-back motion (T4a/T5a) → same-side DNa02 at about 200 Hz. That's the
    optomotor turning reflex.
  * Looming detectors (LPLC2, LC4, LC6) → giant fiber at 140–270 Hz. That's the escape
    response, which here triggers the boost.
  * LC10a/d → same-side DNa02, the known object-steering pathway.
* **Control:** mirroring the game footage flips the steering bias, and a dark screen returns it
  to baseline. The steering really comes from vision.
* **Built-in bias:** the DNp09 drive leaks into right DNa02 (about 13 Hz vs 7 Hz on the left),
  so the fly has a slight intrinsic right-turn bias.

## Usage

```bash
pip install -e .
flyzero download        # FlyWire data (~135 MB) into ~/.cache/flyzero
flyzero info            # what got wired up

# first run: scripted menus (Grand Prix → Blue Falcon → Knight League, Beginner → Mute City I)
flyzero play --rom "F-Zero (USA).sfc" --save-state start.state --frames 600 --video run.mp4
# later runs start straight from the grid
flyzero play --rom "F-Zero (USA).sfc" --state start.state --frames 3600 --video run.mp4

# no ROM / no data: built-in toy racer + a synthetic toy brain (NOT a fly brain; for testing)
flyzero play --game mock --connectome synthetic --video mock.mp4
```

Useful flags:
* `--drive TYPE[:side]=HZ`: stimulate any cell type (repeatable), or `--drive none`.
* `--vision retina`: drive the photoreceptors instead of T4/T5.
* `--motion-gain`: scale of the motion input.
* `--dt`: brain time step.

It runs at about 4–7 game frames per second on 4 CPU cores. Bring your own ROM, dumped from a
cartridge you own.

## Teach the fly: record your own races

The fly learns best from a good driver. Play F-Zero yourself and send the recording:

```bash
pip install -e .
flyzero record --rom "F-Zero (USA).sfc" --out my_races.npz
```

**On Windows** (native, no WSL). stable-retro has no Windows build, so flyzero loads a snes9x
libretro core directly (`flyzero/libretro.py`, verified bit-identical to stable-retro's core):

1. Install 64-bit Python 3.10+ from python.org, then in the repo folder: `pip install -e .`
2. Get `snes9x_libretro.dll`: it's in RetroArch's `cores` folder if you use RetroArch
   (Online Updater → Core Downloader → Nintendo - SNES / SFC (Snes9x)), or download
   `snes9x_libretro.dll.zip` from https://buildbot.libretro.com/nightly/windows/x86_64/latest/
3. Record:
   ```
   python -m flyzero record --rom "F-Zero (USA).sfc" --core C:\path\to\snes9x_libretro.dll --out my_races.npz
   ```
   (or copy the .dll into the folder you run from and leave out `--core`).

A window opens, the menus run by themselves, and you get control on the Mute City I grid.

| F-Zero | Xbox controller | Keyboard |
|---|---|---|
| steer | D-pad or left stick | arrows |
| accelerate (B) | RT or A | X |
| super jet (A) | B | C |
| brake (Y) | LT or X | Z |
| lean L / R | LB / RB | A / S |
| pause | Start | Enter |
| back to the Mute City grid | View/Back | Backspace |
| save & quit | — | Esc |

On Windows the controller is read through XInput (fixed layout, analog triggers), so there is
nothing to configure. On Linux/macOS controllers number their buttons differently: if a button does the wrong thing, run
`flyzero record --controller-test`, press each button to see its number, then pass them in:
`--map "A=0,B=1,X=2,LB=4,RB=5,START=7,BACK=6"` (the default is the Linux Xbox layout). On Linux
the controller must be readable by your user, which it normally is for the logged-in user.

Handy flags:
* `--mute-city`: back-to-back Mute City I runs. After each finish the race is saved and you're
  put straight back on the grid.
* `--queen-league`: the menus pick the Queen League (Mute City II, Port Town I, Red Canyon I,
  White Land I, White Land II).
* `--league knight|queen|king` (or `--king-league`): pick any Grand Prix; with `--first-race`,
  back-to-back runs of that league's first race (Mute City I / II / III).
* `--class beginner|standard|expert|master`: the difficulty class (default beginner). `master`
  is unlocked on the fly by setting the save-RAM flag an Expert win would set (`$7001FA`), e.g.
  `flyzero record --rom "F-Zero (USA).sfc" --league knight --class expert --out knight_expert.npz`.

Game sound plays while you record (`--no-audio` to mute). To race the whole Grand Prix, just
keep driving through the results screens. Every attempt is kept, and a new session never
overwrites an old file (`my_races_2.npz`, ...). The file holds only your button presses (a few hundred KB) plus sync
checkpoints, not the ROM or any video. `flyzero replay --rom ... my_races.npz` re-runs it and
checks the checkpoints. The emulator is deterministic, so the race replays frame for frame on
any machine with the same stable-retro version (pinned to 1.0.1).

## Tests

```bash
pytest                                  # offline
FLYZERO_ROM=path/to/fzero.sfc pytest    # also boots the real game
```
