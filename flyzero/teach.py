"""Teach the fly to drive from recorded human races (instructed learning), then test it solo.

    flyzero lessons --rom fzero.sfc my_races.npz more.npz --out lessons.npz   # once
    flyzero teach --rom fzero.sfc --lessons lessons.npz --exam start.state --epochs 4

In each epoch the recorded inputs drive the car through every race while the fly watches and its
motor synapses are nudged towards the teacher's buttons (``instruct.py``). After each epoch the
fly drives Mute City alone (no teacher, no learning) and its track progress is logged.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .tune import Progress


def build_lessons(rom: str, recordings: list[str], out: str):
    """Replay every recording once (in its own process) and save each race's start + inputs."""
    from .record import replay_all

    from .games import FZero

    data, k = {}, 0
    for rec in recordings:
        racing = {}
        results = replay_all(rom, rec, on_frame=lambda race, frame, b, info:
                             racing.setdefault(race, []).append(FZero.racing(frame)))
        for r in results:
            if r["mismatches"]:
                print(f"skipping {rec} race {r['race']}: {r['mismatches']} checkpoint mismatches")
                continue
            data[f"l{k}_state"] = np.frombuffer(r["start_state"], np.uint8)
            data[f"l{k}_masks"] = np.load(rec)[f"race{r['race']}_masks"]
            data[f"l{k}_racing"] = np.array(racing[r["race"]], bool)
            data[f"l{k}_league"] = np.array(str(np.load(rec)["league"]) if "league" in np.load(rec) else "knight")
            data[f"l{k}_name"] = np.array(f"{Path(rec).stem}#{r['race']}")
            print(f"lesson {k}: {Path(rec).stem} race {r['race']}, {r['frames']} frames "
                  f"({int(np.sum(racing[r['race']]))} racing), laps {r['laps']}")
            k += 1
    data["n"] = k
    np.savez_compressed(out, **data)


def _worker(args):
    k, rom, lessons_path, exam_state, epochs, exam_frames, out, eta, flight_hz, mirror = args
    from . import connectome as cx
    from .biology import corrected
    from .brain import Brain, LIFParams
    from .games import FZero
    from .instruct import InstructedPlasticity, InstructParams
    from .motor import MotorParams, MotorReadout
    from .record import mask_to_buttons
    from .train import VISION
    from .vision import MotionEye, MotionParams

    conn = corrected(cx.load())
    L = np.load(lessons_path)
    lessons = [(L[f"l{i}_state"].tobytes(), L[f"l{i}_masks"], str(L[f"l{i}_name"]),
                L[f"l{i}_racing"] if f"l{i}_racing" in L else np.ones(len(L[f"l{i}_masks"]), bool))
               for i in range(int(L["n"]))]
    rng = np.random.default_rng(k)
    SWAP = {"LEFT": "RIGHT", "RIGHT": "LEFT", "L": "R", "R": "L"}
    exam = Path(exam_state).read_bytes()
    game = FZero(rom, skip_menu=True)
    brain = Brain(conn.weights, LIFParams(dt=0.25), seed=100 + k)
    v = dict(VISION)
    eye = MotionEye(conn, (224, 256), MotionParams(gain=10 ** v.pop("log_gain"), **v))
    motor = MotorReadout(conn, MotorParams())
    plast = InstructedPlasticity(brain, conn, InstructParams(eta=eta))
    drive = conn.find("DNp09")
    flight = np.concatenate([conn.find(t) for t in ("DNa02", "DNg02*", "DNa01")])
    window = 1000.0 / game.fps
    log = Path(out) / "teach_log.jsonl"

    def start(state):
        game.em.set_state(state)
        game.frame_no, game.info, game._last_move, game._empty = 0, {}, 0, 0
        brain.reset()
        eye.lp_hp = None
        return game._press({})

    def think(frame):
        rates = eye.see(frame)
        brain.set_input(np.r_[eye.idx, drive, flight],
                        np.r_[rates, np.full(len(drive), 60.0), np.full(len(flight), flight_hz)])
        return brain.run(window)

    def run_exam():
        frame = start(exam)
        prog = Progress()
        agree = 0
        for i in range(exam_frames):
            counts = think(frame)
            plast.step(counts, None, window, learn=False)
            buttons = motor.update(counts, window)
            frame = game.step(buttons)
            prog.update(game.info["segment"], i)
            if game.info["done"]:
                break
        return {"progress": prog.total, "lap": game.info["lap"], "frames": i + 1,
                "energy": game.info["energy"]}

    res = run_exam()
    print(json.dumps({"fly": k, "eta": eta, "epoch": 0, "exam": res}), flush=True)
    for epoch in range(1, epochs + 1):
        t = time.time()
        errs = []
        for state, masks, name, racing in lessons:
            flip = mirror and rng.random() < 0.5  # mirror world: flipped view, swapped buttons
            frame = start(state)
            for m, is_racing in zip(masks, racing):
                teacher = mask_to_buttons(m)
                seen = frame[:, ::-1] if flip else frame
                lesson = {SWAP.get(b, b): v for b, v in teacher.items()} if flip else teacher
                counts = think(np.ascontiguousarray(seen))
                errs.append(plast.step(counts, lesson if is_racing else None, window))
                motor.update(counts, window)
                frame = game.step(teacher)
        res = run_exam()
        rec = {"fly": k, "eta": eta, "epoch": epoch, "err_hz": round(float(np.mean(errs)), 2),
               "drift": round(plast.drift(), 4), "exam": res, "mins": round((time.time() - t) / 60, 1)}
        print(json.dumps(rec), flush=True)
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        np.savez(Path(out) / f"taught_fly{k}_epoch{epoch}.npz", **plast.state())


def teach(rom, lessons, exam_state, epochs, exam_frames, out, etas=(1e-4, 3e-4, 1e-3, 3e-3), flight_hz=30.0,
          mirror=True):
    Path(out).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(etas)) as pool:
        pool.map(_worker, [(k, rom, lessons, exam_state, epochs, exam_frames, out, eta, flight_hz, mirror)
                           for k, eta in enumerate(etas)])
