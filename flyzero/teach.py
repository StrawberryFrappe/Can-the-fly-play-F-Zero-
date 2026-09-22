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
    (k, rom, lessons_path, exam_state, epochs, exam_frames, out, eta, bias_mv, mirror, smooth,
     practice, practice_frames, reward_eta, init) = args
    from . import connectome as cx
    from .biology import corrected
    from .brain import Brain, LIFParams
    from .games import FZero
    from .instruct import InstructedPlasticity, InstructParams, intent, targets_from_intent
    from .motor import BUTTONS
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
    if init:  # continue from an earlier taught fly
        st = np.load(init)
        plast.load({"pos": st["pos"], "data": st["data"]})
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

    # flying and wanting to go: a steady depolarisation of the motor neurons (noise-free), so
    # they sit just around threshold where their input synapses decide what they do
    brain.set_bias(np.r_[flight, drive], bias_mv)

    def think(frame):
        brain.set_input(eye.idx, eye.see(frame))
        return brain.run(window)

    def run_exam(n=3):
        """Mean over n solo drives (spiking noise differs each time): single runs vary a lot."""
        runs = []
        for _ in range(n):
            frame = start(exam)
            prog = Progress()
            for i in range(exam_frames):
                counts = think(frame)
                plast.step(counts, None, window, learn=False)
                buttons = motor.update(counts, window)
                frame = game.step(buttons)
                prog.update(game.info["segment"], i)
                if game.info["done"]:
                    break
            runs.append({"progress": prog.total, "lap": game.info["lap"], "frames": i + 1})
        return {"mean_progress": round(float(np.mean([r["progress"] for r in runs])), 1),
                "best": max(r["progress"] for r in runs), "mean_frames": int(np.mean([r["frames"] for r in runs])),
                "laps": max(r["lap"] for r in runs)}

    res = run_exam()
    print(json.dumps({"fly": k, "eta": eta, "epoch": 0, "exam": res}), flush=True)
    with open(log, "a") as f:
        f.write(json.dumps({"fly": k, "eta": eta, "epoch": 0, "exam": res}) + "\n")
    for epoch in range(1, epochs + 1):
        t = time.time()
        errs = []
        for state, masks, name, racing in lessons:
            flip = mirror and rng.random() < 0.5  # mirror world: flipped view, swapped buttons
            frame = start(state)
            smooth_intent = intent(masks, BUTTONS, smooth) if smooth else None
            for t, (m, is_racing) in enumerate(zip(masks, racing)):
                teacher = mask_to_buttons(m)
                seen = frame[:, ::-1] if flip else frame
                if smooth:
                    x = smooth_intent[t].copy()
                    if flip:
                        x[:2] = -x[:2]
                    lesson = targets_from_intent(x, plast.p)
                else:
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

    if practice:
        # practice: the fly drives alone; reward (progress / crashes / reversing, damage felt as
        # heat) keeps shaping the same motor synapses the lessons shaped
        from .learning import LearnParams, Reward, RewardPlasticity

        rp = RewardPlasticity(brain, conn, LearnParams(eta=reward_eta), seed=k)
        heat = conn.find("TRN_VP2")
        for ep in range(1, practice + 1):
            t = time.time()
            frame = start(exam)
            reward, prog, total, hurt = Reward(rp.p), Progress(), 0.0, 0.0
            for i in range(practice_frames):
                nidx, nrate = rp.exploration(window)
                brain.set_input(np.r_[eye.idx, nidx, heat],
                                np.r_[eye.see(frame), nrate, np.full(len(heat), rp.p.hurt_hz if hurt > 0 else 0.0)])
                hurt -= window
                counts = brain.run(window)
                buttons = motor.update(counts, window)
                frame = game.step(buttons)
                r = reward(game.info)
                if reward.parts["crash"] > 0:
                    hurt = rp.p.hurt_ms
                total += r
                rp.step(counts, r, window)
                prog.update(game.info["segment"], i)
                if game.info["done"]:
                    break
            rec = {"fly": k, "eta": eta, "practice": ep, "progress": prog.total, "frames": i + 1,
                   "reward": round(total, 1), "mins": round((time.time() - t) / 60, 1)}
            if ep % 5 == 0:
                rec["exam"] = run_exam()
                np.savez(Path(out) / f"taught_fly{k}_practice{ep}.npz", **plast.state())
            print(json.dumps(rec), flush=True)
            with open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")


def teach(rom, lessons, exam_state, epochs, exam_frames, out, etas=(1e-4, 3e-4, 1e-3, 3e-3), bias_mv=7.8,
          mirror=True, smooth=15.0, practice=0, practice_frames=2400, reward_eta=5e-4, init=None):
    """``smooth``: teach the teacher's intention smoothed over this many frames (0 = exact taps)."""
    Path(out).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(etas)) as pool:
        pool.map(_worker, [(k, rom, lessons, exam_state, epochs, exam_frames, out, eta, bias_mv, mirror, smooth,
                            practice, practice_frames, reward_eta, init)
                           for k, eta in enumerate(etas)])
