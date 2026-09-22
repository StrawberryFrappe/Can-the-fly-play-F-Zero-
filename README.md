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

## Tests

```bash
pytest                                  # offline
FLYZERO_ROM=path/to/fzero.sfc pytest    # also boots the real game
```
