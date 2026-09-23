# Can the fly play F-Zero?

A simulated whole *Drosophila* brain (FlyWire connectome, LIF model) learning to drive F-Zero (SNES).

**Start with `docs/HANDOFF.md`**: the goal, the owner's rules ("a cyborg fly, not a robot fly";
no autopilot crutches), the current results, and how to continue. Then:
* `README.md`: overview and the step-by-step lab notebook.
* `docs/fzero-internals.md`: RAM addresses, HUD pixels, menu timings, emulator notes.

Conventions:
* Always simulate the corrected brain: `flyzero.biology.corrected(connectome.load())`.
* Motor neurons get a steady bias (`Brain.set_bias`, ~7.8 mV), not Poisson "drive" (noise).
* One emulator per process. Save states aren't portable between snes9x builds; inputs are.
* Never commit the ROM, the snes9x .dll, or `*.state` files. Human recordings live in
  `runs/recordings/` (inputs only).
* Tests: `pytest` (add `FLYZERO_ROM=path/to/rom` to include the emulator tests).
* Report honestly how much of any result is the fly and how much is ours.
