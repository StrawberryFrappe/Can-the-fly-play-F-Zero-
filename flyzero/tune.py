"""Calibrate the fly <-> game interface so the (fixed) brain finishes races.

Only the interface is searched: how strongly the game's motion drives the T4/T5
cells, which part of the screen they see, how descending-neuron rates become
button presses, and the tonic walking drive. The connectome and the neuron model
are never changed. It is the same idea as calibrating a brain-computer
interface decoder, just pointed the other way round.

Fitness = net track progress (segments, 59 per lap on Mute City I, going the
wrong way counts negative) plus a small bonus for energy left.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SEGMENTS = 59  # Mute City I

# name, low, high: the search space (CMA works in [0, 1] and we map linearly)
SPACE = [
    ("log_gain", 2.3, 4.0),     # motion correlator gain, log10
    ("y_min", 0.0, 0.7),        # fraction of the screen (from the top) the fly ignores
    ("steer_threshold", 2.0, 80.0),
    ("steer_bias", -60.0, 60.0),
    ("tau_ms", 20.0, 300.0),
    ("lean_threshold", 5.0, 200.0),
    ("drive_hz", 30.0, 150.0),
]


@dataclass
class Interface:
    log_gain: float = 3.0
    y_min: float = 0.0
    steer_threshold: float = 10.0
    steer_bias: float = 0.0
    tau_ms: float = 80.0
    lean_threshold: float = 10.0
    drive_hz: float = 60.0
    flip: bool = False  # swap left/right at the motor side (a "prism" experiment)

    @classmethod
    def from_unit(cls, x, flip=False):
        kw = {name: lo + float(np.clip(v, 0, 1)) * (hi - lo) for (name, lo, hi), v in zip(SPACE, x)}
        return cls(flip=flip, **kw)

    def to_unit(self):
        return np.array([(getattr(self, n) - lo) / (hi - lo) for n, lo, hi in SPACE])


class Progress:
    """Unwrapped track progress from the per-frame segment number."""

    def __init__(self, segments: int = SEGMENTS):
        self.segments = segments   # per lap: Mute City I 59, Big Blue 84
        self.last = None
        self.total = 0

    def update(self, seg: int, frame: int) -> int:
        if self.last is not None:
            n = self.segments
            d = (seg - self.last + n // 2) % n - n // 2
            self.total += d
        self.last = seg
        return self.total


_CONN = None


def _conn(data_dir=None):
    global _CONN
    if _CONN is None:
        from . import connectome as cx

        _CONN = cx.load("flywire", data_dir or cx.DEFAULT_DATA_DIR)
    return _CONN


def race(iface: Interface, rom: str, state: str, frames: int, seed: int = 0, dt: float = 0.25,
         on_frame=None, verbose: bool = False) -> dict:
    from .brain import Brain, LIFParams
    from .games import FZero
    from .motor import MotorParams, MotorReadout
    from .vision import MotionEye, MotionParams

    conn = _conn()
    game = FZero(rom, state=state)
    frame = game.reset()
    brain = Brain(conn.weights, LIFParams(dt=dt), seed=seed)
    eye = MotionEye(conn, frame.shape[:2], MotionParams(gain=10 ** iface.log_gain, y_min=iface.y_min))
    mp_ = MotorParams(tau_ms=iface.tau_ms, steer_threshold=iface.steer_threshold,
                      steer_bias=iface.steer_bias, lean_threshold=iface.lean_threshold)
    motor = MotorReadout(conn, mp_)
    drive = conn.find("DNp09")
    window = 1000.0 / game.fps
    prog = Progress()
    best = 0
    info = {}
    for i in range(frames):
        rates = eye.see(frame)
        brain.set_input(np.r_[eye.idx, drive], np.r_[rates, np.full(len(drive), iface.drive_hz)])
        counts = brain.run(window)
        buttons = dict(motor.update(counts, window))
        if iface.flip:
            buttons["LEFT"], buttons["RIGHT"] = buttons["RIGHT"], buttons["LEFT"]
            buttons["L"], buttons["R"] = buttons["R"], buttons["L"]
        frame = game.step(buttons)
        info = game.info
        p = prog.update(info["segment"], i)
        best = max(best, p)
        if on_frame:
            on_frame(frame, brain, eye, motor, counts, info, buttons)
        if verbose and i % 300 == 0:
            print(i, p, info, flush=True)
        if info["done"]:
            break
    return {"progress": prog.total, "best": best, "lap": info.get("lap", 0),
            "energy": info.get("energy", 0.0), "frames": i + 1, "finished": info.get("lap", 0) >= 5}


def fitness(res: dict) -> float:
    return res["progress"] + 5.0 * res["energy"] + (100.0 if res["finished"] else 0.0)


def _worker(args):
    x, flip, rom, state, frames, seed = args
    iface = Interface.from_unit(x, flip)
    res = race(iface, rom, state, frames, seed=seed)
    return fitness(res), res, asdict(iface)


def search(rom: str, state: str, frames: int, generations: int, popsize: int, workers: int,
           out: Path, flip: bool = False, x0=None, sigma: float = 0.3, seed: int = 0):
    import cma

    x0 = Interface().to_unit() if x0 is None else x0
    es = cma.CMAEvolutionStrategy(list(x0), sigma, {"popsize": popsize, "bounds": [0, 1], "seed": seed + 1,
                                                    "verbose": -9})
    log = []
    best = (-1e9, None, None)
    ctx = mp.get_context("fork")
    _conn()  # load once, shared with forked workers
    with ctx.Pool(workers) as pool:
        for gen in range(generations):
            xs = es.ask()
            t = time.time()
            results = pool.map(_worker, [(x, flip, rom, state, frames, seed * 1000 + gen * popsize + k)
                                         for k, x in enumerate(xs)])
            es.tell(xs, [-f for f, _, _ in results])
            for f, res, iface in results:
                log.append({"gen": gen, "fitness": f, **res, "iface": iface})
                if f > best[0]:
                    best = (f, res, iface)
            fs = [f for f, _, _ in results]
            print(f"gen {gen:3d}  best {max(fs):7.1f}  median {np.median(fs):7.1f}  "
                  f"overall {best[0]:7.1f} lap {best[1]['lap']} ({time.time() - t:.0f}s)", flush=True)
            out.write_text(json.dumps({"best": {"fitness": best[0], **best[1], "iface": best[2]},
                                       "log": log}, indent=1))
    return best
