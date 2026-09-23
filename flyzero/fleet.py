"""A fleet: B simulated flies on the GPU, each at the controls of its own F-Zero emulator.

Glue between ``gpu.BatchBrain`` (brains), ``pool.EmulatorPool`` (games + eyes), ``batch.BatchMotor``
(buttons) and the batched learning rules. The fly's interface is exactly ``teach._worker``'s: the
corrected FlyWire brain, the motion/medulla/colour eye (``train.VISION``), the steady bias on
the motor neurons, dt = 0.25 ms, one game frame per 16.6 ms of brain time.
"""

from __future__ import annotations

import time

import numpy as np

import cupy as cp

from .tune import Progress

BIAS_TYPES = ("DNa02", "DNg02*", "DNa01", "DNp09")


class Fleet:
    def __init__(self, rom: str, batch: int, seed: int = 0, plastic_types=None,
                 bias_mv: float = 7.8, conn=None, core: str | None = None, line: str | None = None,
                 deep: bool = False, taps: bool = False, steer_span: float = 150.0, lean_span: float = 70.0):
        """``line``: racing-line file for the pilot's labels (e.g. runs/pilot/mute_city_line.npz).
        ``deep``: also make the synapses onto the DNs' presynaptic partners plastic.
        ``plastic_types``: default ``batch.PLASTIC_TYPES`` (the motor DNs and the giant fiber).
        ``taps``: the readout taps the D-pad / shoulders at a rate set by the DNs (``MotorParams``)."""
        from . import connectome as cx
        from .batch import BatchMotor, plastic_positions
        from .biology import corrected
        from .brain import LIFParams
        from .gpu import BatchBrain
        from .pool import EmulatorPool
        from .train import VISION
        from .vision import MotionEye, MotionParams

        self.conn = conn = conn if conn is not None else corrected(cx.load())
        self.B = batch
        from .batch import PLASTIC_TYPES

        self.pos = plastic_positions(conn, plastic_types or PLASTIC_TYPES)
        if deep:
            from .batch import deep_positions

            self.pos = np.union1d(self.pos, deep_positions(conn, self.pos))
        self.brain = BatchBrain(conn.weights, LIFParams(dt=0.25), batch=batch, seed=seed, plastic_pos=self.pos)
        v = dict(VISION)
        self.eye = MotionEye(conn, (224, 256), MotionParams(gain=10 ** v.pop("log_gain"), **v))
        from .motor import MotorParams

        self.motor = BatchMotor(conn, batch, MotorParams(taps=taps, steer_span=steer_span, lean_span=lean_span))
        self.bias_idx = np.concatenate([conn.find(t) for t in BIAS_TYPES])
        self.bias_mv = bias_mv
        self.brain.set_bias(self.bias_idx, bias_mv)
        self.eye_slots = self.brain.slots(self.eye.idx)
        ln = None if line is None else {k: v for k, v in np.load(line).items() if k in ("points", "speed")}
        self.pool = EmulatorPool(rom, self.eye, batch, core=core, line=ln)
        self.window = 1000.0 / 60.0988
        self.extra_idx = np.zeros(0, np.int64)    # optional extra Poisson inputs (exploration, heat)

    def close(self):
        self.pool.close()

    def think(self, rates: np.ndarray, extra: np.ndarray | None = None) -> cp.ndarray:
        """One frame of brain time for every slot; ``rates``: eye rates (B, n_eye)."""
        self.brain.p_ext.fill(0)
        self.brain.p_ext[:, self.eye_slots] = cp.asarray(rates) * np.float32(self.brain.p.dt / 1000.0)
        if extra is not None and len(self.extra_idx):
            j = self.brain.slots(self.extra_idx)
            self.brain.p_ext[:, j] += cp.asarray(extra, cp.float32) * np.float32(self.brain.p.dt / 1000.0)
        return self.brain.run(self.window)

    def reset_slots(self, slots):
        slots = np.atleast_1d(slots)
        self.brain.reset(slots)
        self.motor.reset(slots)

    def exam(self, state: bytes, frames: int, on_frame=None, log_every: int = 0, record: bool = False) -> list[dict]:
        """Every slot drives alone from ``state`` (no learning). Different spiking noise per slot.
        ``record``: each result also holds the drive's inputs (``masks``) and DN rates (``rates``),
        enough to replay it exactly (``live.save_run``)."""
        from .record import buttons_to_mask
        self.brain.reset()
        self.motor.reset()
        rates, infos = self.pool.load([state] * self.B)
        progs = [Progress() for _ in range(self.B)]
        done = np.zeros(self.B, bool)
        res = [None] * self.B
        masks, dn = [[] for _ in range(self.B)], [[] for _ in range(self.B)]
        t0 = time.time()
        for i in range(frames):
            counts = self.think(rates)
            buttons = self.motor.update(counts, self.window)
            if record:
                for k in np.flatnonzero(~done):
                    masks[k].append(buttons_to_mask(buttons[k]))
                    dn[k].append(self.motor.rates[k].astype(np.float16))
            rates, infos = self.pool.step(buttons)
            for k, info in enumerate(infos):
                if done[k]:
                    continue
                progs[k].update(info["segment"], i)
                if info["done"] or i == frames - 1:
                    done[k] = True
                    res[k] = {"progress": progs[k].total, "lap": info["lap"], "frames": i + 1,
                              "energy": info["energy"], "finished": info["lap"] >= 5}
                    if record:
                        res[k]["masks"], res[k]["rates"] = np.array(masks[k]), np.array(dn[k])
            if on_frame:
                on_frame(i, counts, buttons, infos)
            if log_every and (i + 1) % log_every == 0:
                print(f"  frame {i + 1}: {(i + 1) * self.B / (time.time() - t0):.0f} fly-frames/s, "
                      f"progress {[p.total for p in progs]}", flush=True)
            if done.all():
                break
        return res


def summarize(results: list[dict]) -> dict:
    prog = [r["progress"] for r in results]
    return {"mean_progress": round(float(np.mean(prog)), 1), "best": int(max(prog)),
            "worst": int(min(prog)), "laps_best": max(r["lap"] for r in results),
            "finished": sum(r["finished"] for r in results), "n": len(results),
            "mean_frames": int(np.mean([r["frames"] for r in results]))}
