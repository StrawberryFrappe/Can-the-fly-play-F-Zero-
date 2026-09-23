# runs/

* `recordings/`: the owner's F-Zero races, recorded with `flyzero record` (button presses and
  sync checkpoints only, no ROM or video). Rebuild lessons from them with `flyzero lessons`.

  | File | Contents |
  |---|---|
  | `01_mute_city.npz` | Mute City I, 5 laps |
  | `02_knight_gp_first3.npz` | Knight GP: Mute City I, Big Blue, Sand Ocean (old sequential format) |
  | `03_mute_city.npz` | Mute City I (2'25", fastest) |
  | `04_mute_city_x4.npz` | 4 back-to-back Mute City I runs (`--mute-city`) |
  | `05_queen_league_gp.npz` | the whole Queen League GP, incl. two deaths on White Land II |
* `results/`: logs and learned weights of the experiments in the README lab notebook.
  Weight files hold the plastic synapses only (`pos`, `data`, `w0`); load them with
  `InstructedPlasticity.load` / `teach --init`.
