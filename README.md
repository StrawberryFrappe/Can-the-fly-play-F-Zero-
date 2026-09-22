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

### Findings along the way

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
