# Weights

* `natural_fly_best_n7_3200k.npz`: the natural fly's best checkpoint so far (mean ~98 track
  segments per Mute City exam drive, best 3 laps, no finish). These are the plastic synapses
  (FlyWire synapses onto the motor DNs and their inputs) plus the DNs' intrinsic biases. Load
  them with `BatchInstruct(...).load` on a `Fleet(..., deep=True)`, or pass them to `--init` /
  `live --weights ... --deep`.
* `natural_fly_host_r3g_800k.npz`: the natural fly that hosts the implant (the augmented fly).
* `implant_mute_city.pt` + `implant_mute_city_normaliser.npz`: the implant that finished Mute City
  I top 3 in 4 of 16 races with that host. Use `python -m flyzero.implant --host
  runs/results/weights/natural_fly_host_r3g_800k.npz --l2-top 2000 --taps --resume <folder with
  implant.pt + normaliser.npz> --eval 16`.
