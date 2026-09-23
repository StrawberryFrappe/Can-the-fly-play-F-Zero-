# tools/: experimental scripts (not part of the package)

* `road_teacher.py`: a visual lane-keeping driver (road = low-saturation grey), with a
  `GuardedTeacher` that uses the track counter to recover from U-turns. Too weak to finish a lap
  on its own.
* `search_teacher.py`: look-ahead instructor. At each chunk it tries 5 moves from a save state,
  rolls each out with the road teacher, and keeps the best by track progress / energy. Usage:
  `python search_teacher.py <chunk> <horizon> <steer_penalty> <tag>` (run from this folder, with
  `start.state` present). Its best setting managed about 1.5 laps before exploding; it's a
  starting point for DAgger-style corrections (docs/HANDOFF.md, idea 2).
