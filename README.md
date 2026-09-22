# Can the fly play F-Zero?

A leaky integrate-and-fire simulation of the whole *Drosophila* brain (FlyWire v783
connectome, same model and parameters as Shiu et al. 2024, *Nature*) at the controls
of F-Zero on the SNES.

- **Eyes:** game frames → Poisson firing rates of the photoreceptors. Each
  photoreceptor's place in the visual field comes from where it sits in the brain.
- **Brain:** ~140k LIF neurons, signed synapse counts as weights (NumPy/SciPy).
- **Hands:** descending neurons → buttons: DNa02 L/R steer, DNa01 L/R lean,
  DNp09 (P9) accelerate, MDN brake, DNp01 (giant fiber) boost.
- **Game:** snes9x via stable-retro, stepped frame by frame, so 1 game frame = 16.6 ms of
  neural time.

By default DNp09 gets a tonic 60 Hz drive, a bit like optogenetic activation, so the fly
wants to move; `--drive none` turns it off. Steering comes only from vision.

## Usage

```bash
pip install -e .
flyzero download                      # FlyWire connectome + annotations (~GitHub)
flyzero info                          # what got wired up
flyzero play --rom F-Zero.sfc --video run.mp4 --save-state start.state
flyzero play --rom F-Zero.sfc --state start.state --frames 3600 --video run.mp4

# no ROM / no data: built-in toy racer and a synthetic toy brain
flyzero play --game mock --connectome synthetic --video mock.mp4
```

Bring your own ROM, dumped from a cartridge you own. The synthetic brain is **not** a fly
brain. It exists only to test the pipeline offline.
