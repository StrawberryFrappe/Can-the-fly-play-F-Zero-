"""Train the fly's motor-output synapses with reward, episode after episode.

    flyzero learn --rom fzero.sfc --state start.state --episodes 40 --workers 4

Each worker is an independent fly (own seed) that keeps its learned synapses across episodes.
Progress goes to <out>/log.jsonl, the best weights of each fly to <out>/fly<k>.npz.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .tune import Progress

VISION = dict(log_gain=2.7, y_min=0.2, transient_gain=4.0, sustained_gain=4.0, chromatic_gain=1.0)


def make_fly(conn, frame_shape, seed, learn_params=None, vision=VISION, dt=0.25):
    from .brain import Brain, LIFParams
    from .learning import LearnParams, RewardPlasticity
    from .motor import MotorParams, MotorReadout
    from .vision import MotionEye, MotionParams

    brain = Brain(conn.weights, LIFParams(dt=dt), seed=seed)
    v = dict(vision)
    eye = MotionEye(conn, frame_shape, MotionParams(gain=10 ** v.pop("log_gain"), **v))
    motor = MotorReadout(conn, MotorParams())
    plast = RewardPlasticity(brain, conn, learn_params or LearnParams(), seed=seed)
    return brain, eye, motor, plast


def episode(game, brain, eye, motor, plast, conn, frames, learn=True, drive_hz=60.0, on_frame=None):
    from .learning import Reward

    frame = game.reset()
    brain.reset()
    eye.lp_hp = None
    reward = Reward(plast.p)
    drive = conn.find("DNp09")
    heat = conn.find("TRN_VP2")  # antennal heat-sensing neurons: damage feels like heat
    hurt_left = 0.0
    window = 1000.0 / game.fps
    prog = Progress()
    total_r = 0.0
    parts = {"speed": 0.0, "fwd": 0, "rev": 0, "crash": 0.0}
    for i in range(frames):
        rates = eye.see(frame)
        nidx, nrate = plast.exploration(window) if learn else (np.zeros(0, int), np.zeros(0))
        hurt_hz = plast.p.hurt_hz if hurt_left > 0 else 0.0
        brain.set_input(np.r_[eye.idx, drive, nidx, heat],
                        np.r_[rates, np.full(len(drive), drive_hz), nrate, np.full(len(heat), hurt_hz)])
        hurt_left -= window
        counts = brain.run(window)
        buttons = motor.update(counts, window)
        frame = game.step(buttons)
        info = game.info
        r = reward(info)
        if reward.parts["crash"] > 0:
            hurt_left = plast.p.hurt_ms
        total_r += r
        for k, v in reward.parts.items():
            parts[k] += v
        if learn:
            plast.step(counts, r, window)
        prog.update(info["segment"], i)
        if on_frame:
            on_frame(frame, counts, info, buttons, r)
        if info["done"]:
            break
    return {"progress": prog.total, "lap": info["lap"], "frames": i + 1, "reward": round(total_r, 1),
            "energy": info["energy"], "drift": round(plast.drift(), 4),
            **{k: round(float(v), 1) for k, v in parts.items()}}


def _worker(args):
    k, rom, state, episodes, frames, out, lp = args
    from . import connectome as cx
    from .biology import corrected
    from .games import FZero
    from .learning import LearnParams

    conn = corrected(cx.load())
    game = FZero(rom, state=state)
    lp = dict(lp)
    lp["eta"] = lp["eta"] * lp.pop("eta_scales")[k % 4]  # a small learning-rate sweep across flies
    brain, eye, motor, plast = make_fly(conn, (224, 256), seed=1000 + k, learn_params=LearnParams(**lp))
    best = -1e9
    for ep in range(episodes):
        t = time.time()
        res = episode(game, brain, eye, motor, plast, conn, frames)
        res.update({"fly": k, "eta": lp["eta"], "episode": ep, "secs": round(time.time() - t)})
        with open(Path(out) / "log.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)
        score = res["progress"]
        if score >= best:
            best = score
            np.savez(Path(out) / f"fly{k}.npz", **plast.state(), episode=ep, progress=score)
        np.savez(Path(out) / f"fly{k}_last.npz", **plast.state(), episode=ep, progress=score)
    return best


def learn(rom, state, episodes, frames, workers, out, learn_params=None):
    from .learning import LearnParams

    Path(out).mkdir(parents=True, exist_ok=True)
    lp = asdict(learn_params or LearnParams())
    lp["targets"] = tuple(lp["targets"])
    lp["eta_scales"] = (1, 3, 10, 30)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        return pool.map(_worker, [(k, rom, state, episodes, frames, out, lp) for k in range(workers)])
