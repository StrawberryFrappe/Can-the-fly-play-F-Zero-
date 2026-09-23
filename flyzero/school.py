"""Training programs for a GPU fleet (``fleet.Fleet``): practice, DAgger lessons, exams.

    python -m flyzero.school practice --init runs/results/3_intent_all/taught_fly0_epoch1.npz ...
    python -m flyzero.school dagger ...

Every program only changes the fly's own plastic synapses (FlyWire synapses onto its motor
neurons, sign and caps kept). Exams: every slot drives alone from the start line, no learning.
Logs go to ``<out>/log.jsonl``, weights to ``<out>/fly{k}_*.npz`` (plastic entries only).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# no module-level CuPy: spawned emulator workers re-import this module and don't need it


def _log(out, rec):
    print(json.dumps(rec), flush=True)
    with open(Path(out) / "log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_exam(fleet, plast, exam_state, frames, flies, drives, save=None):
    """``drives`` solo drives per fly; slots are re-assigned to flies for the exam.
    ``save``: path prefix; each fly's best drive is saved there for replay (``live``)."""
    import cupy as cp
    from .fleet import summarize

    saved = cp.asnumpy(fleet.brain.pw).copy()
    saved_bias = cp.asnumpy(fleet.brain.bias_ext).copy()   # intrinsic plasticity lives here
    per = {}
    order = [f for f in flies for _ in range(drives)]
    for start in range(0, len(order), fleet.B):
        chunk = order[start:start + fleet.B]
        for s in range(fleet.B):
            f = chunk[s] if s < len(chunk) else chunk[0]
            src = np.flatnonzero(plast.fly_of_slot == f)[0]
            fleet.brain.pw[s] = cp.asarray(saved[src])
            fleet.brain.bias_ext[s] = cp.asarray(saved_bias[src])
        res = fleet.exam(exam_state, frames, record=save is not None)
        for s, f in enumerate(chunk):
            per.setdefault(f, []).append(res[s])
    fleet.brain.pw[...] = cp.asarray(saved)
    fleet.brain.bias_ext[...] = cp.asarray(saved_bias)
    if save is not None:
        from .live import save_run

        for f, r in per.items():
            best = max(r, key=lambda x: (x["progress"], -x["frames"]))
            save_run(f"{save}_fly{f}.npz", exam_state, best["masks"], best["rates"], "fly",
                     progress=best["progress"], laps=best["lap"], frames=best["frames"])
            for x in r:
                x.pop("masks"), x.pop("rates")
    return {f: {**summarize(r), "each": [x["progress"] for x in r]} for f, r in per.items()}


def practice(a):
    """Reward practice (``teach`` practice phase), batched: each fly practises on several slots."""
    import cupy as cp
    from .batch import BatchReward
    from .fleet import Fleet
    from .learning import LearnParams, Reward
    from .tune import Progress

    etas = [float(e) for e in a.etas.split(",")]
    fos = np.repeat(np.arange(len(etas)), a.batch // len(etas))
    fleet = Fleet(a.rom, len(fos), seed=a.seed)
    rp = BatchReward(fleet.brain, fleet.conn, LearnParams(), fly_of_slot=fos, seed=a.seed)
    if a.init:
        rp.load(np.load(a.init))
    heat = fleet.conn.find("TRN_VP2")
    fleet.extra_idx = np.r_[rp.targets, heat]
    exam_state = Path(a.exam).read_bytes()
    eta_slot = np.array(etas)[fos]
    res = run_exam(fleet, rp, exam_state, a.exam_frames, range(len(etas)), a.drives)
    _log(a.out, {"drive": 0, "exam": res})
    for ep in range(1, a.drives_total + 1):
        t = time.time()
        fleet.brain.reset()
        fleet.motor.reset()
        rp.reset(np.arange(fleet.B))
        rates, infos = fleet.pool.load([exam_state] * fleet.B)
        rewards = [Reward(rp.p) for _ in range(fleet.B)]
        progs = [Progress() for _ in range(fleet.B)]
        hurt = np.zeros(fleet.B)
        live = np.ones(fleet.B, bool)
        total = np.zeros(fleet.B)
        for i in range(a.practice_frames):
            noise = rp.exploration(fleet.window)
            extra = np.c_[noise, np.repeat((hurt > 0)[:, None] * rp.p.hurt_hz, len(heat), 1)]
            hurt -= fleet.window
            counts = fleet.think(rates, extra)
            buttons = fleet.motor.update(counts, fleet.window)
            rates, infos = fleet.pool.step(buttons)
            r = np.zeros(fleet.B, np.float32)
            for k, info in enumerate(infos):
                if not live[k]:
                    continue
                r[k] = rewards[k](info)
                if rewards[k].parts["crash"] > 0:
                    hurt[k] = rp.p.hurt_ms
                progs[k].update(info["segment"], i)
                if info["done"]:
                    live[k] = False
            total += r
            _step_scaled(rp, counts, r, live.copy(), fleet.window, eta_slot)
            if not live.any():
                break
        rec = {"drive": ep, "progress": [p.total for p in progs], "reward": np.round(total, 1).tolist(),
               "drift": np.round(rp.drift(), 4).tolist(), "secs": round(time.time() - t)}
        if ep % a.exam_every == 0:
            rec["exam"] = run_exam(fleet, rp, exam_state, a.exam_frames, range(len(etas)), a.drives)
            for f in range(len(etas)):
                np.savez_compressed(Path(a.out) / f"fly{f}_drive{ep}.npz", **rp.state(f))
        _log(a.out, rec)
    fleet.close()


def dagger(a):
    """DAgger lessons: the fly drives, the pilot labels what it would do from the fly's own
    position, the fly's motor synapses learn towards the label (delta rule, ``BatchInstruct``).

    The pilot never presses a button of the fly's car. Training drives start from the grid or
    from points along the pilot's own lap (curriculum); when the fly crashes out, its drive
    resumes from its own snapshot a few seconds before (the owner OK'd this for training).
    """
    import cupy as cp
    from .batch import BatchInstruct, BatchInstructDeep, targets
    from .fleet import Fleet
    from .instruct import InstructParams

    etas = [float(e) for e in a.etas.split(",")]
    fos = np.repeat(np.arange(len(etas)), a.batch // len(etas))
    B = len(fos)
    fleet = Fleet(a.rom, B, seed=a.seed, line=a.line, deep=max(float(e) for e in a.eta_deep.split(',')) > 0)
    deep = [float(e) for e in a.eta_deep.split(",")]
    if max(deep) > 0:
        # --eta-deep: relative to each fly's eta; one value, or one per fly
        deep = np.broadcast_to(np.array(deep), (len(etas),))
        ip = BatchInstructDeep(fleet.brain, fleet.conn, InstructParams(eta=etas[0]), fly_of_slot=fos,
                               eta_deep=etas[0], deep_scale=deep[fos], eta_bias=a.eta_bias)
        print(f"deep plasticity: {ip.n_deep} synapses onto {len(ip.l1)} L1 neurons", flush=True)
    else:
        ip = BatchInstruct(fleet.brain, fleet.conn, InstructParams(eta=etas[0]), fly_of_slot=fos, eta_bias=a.eta_bias)
    if a.init:
        ip.load(np.load(a.init))
    exam_state = Path(a.exam).read_bytes()
    eta_slot = cp.asarray(np.array(etas)[fos].astype(np.float32))[:, None]
    rng = np.random.default_rng(a.seed)
    starts = fleet.pool.pilot_drive(exam_state, 12000, a.snap_every)
    print(f"{len(starts)} curriculum starts along the pilot's race", flush=True)

    flips = np.zeros(B, bool)   # mirror world: flipped view, swapped buttons, mirrored labels
    SWAP = {"LEFT": "RIGHT", "RIGHT": "LEFT", "L": "R", "R": "L"}

    def new_start(k):
        fleet.reset_slots(k)
        ip.reset(k)
        flips[k] = rng.random() < a.mirror
        if rng.random() < a.p_grid:
            r, inf = fleet.pool.load([exam_state], [bool(flips[k])], which=[k])
        else:
            r, inf = fleet.pool.restore([starts[rng.integers(len(starts))]], [k], [bool(flips[k])])
        return r[0], inf[0]

    def label(inf, k):
        x = np.asarray(inf["pilot"], np.float32).copy()
        if flips[k]:
            x[:2] = -x[:2]
        return targets(x, ip.p)

    def exam_now(tag):
        return run_exam(fleet, ip, exam_state, a.exam_frames, range(len(etas)), a.drives,
                        save=str(Path(a.out) / f"best_{tag}"))

    _log(a.out, {"frames": 0, "exam": exam_now(0), "etas": etas})
    rates = np.zeros((B, len(fleet.eye.idx)), np.float32)
    infos = [None] * B
    for k in range(B):
        rates[k], infos[k] = new_start(k)
    age = np.zeros(B, int)
    snaps = [[] for _ in range(B)]
    restores = np.zeros(B, int)
    stats = {"frames": 0, "err": [], "restores": 0, "new": 0, "progress": []}
    last_exam, t = 0, time.time()
    total = 0
    while total < a.frames:
        counts = fleet.think(rates)
        buttons = fleet.motor.update(counts, fleet.window)
        buttons = [{SWAP.get(b, b): v for b, v in bt.items()} if flips[k] else bt for k, bt in enumerate(buttons)]
        tgt = np.array([label(inf, k) for k, inf in enumerate(infos)], np.float32)
        learn = np.array([inf.get("racing", False) for inf in infos])
        err = ip.step(counts, tgt, learn, fleet.window, eta=eta_slot)
        stats["err"].append(float(err[learn].mean()) if learn.any() else 0.0)
        rates, infos = fleet.pool.step(buttons)
        age += 1
        total += B
        if (age % a.snap_every == 0).any():
            ks = np.flatnonzero(age % a.snap_every == 0)
            for k, snap in zip(ks, fleet.pool.snapshot(ks)):
                snaps[k] = (snaps[k] + [snap])[-4:]
        for k, inf in enumerate(infos):
            crashed = inf["done"] or inf["energy"] < 0.05
            if crashed and restores[k] < a.max_restores and len(snaps[k]) >= 2:
                restores[k] += 1
                stats["restores"] += 1
                fleet.reset_slots(k)
                ip.reset(k)
                snap = snaps[k][-2]         # a few seconds before the crash
                snaps[k] = snaps[k][:-2]
                r, i2 = fleet.pool.restore([snap], [k], [bool(flips[k])])
                rates[k], infos[k] = r[0], i2[0]
                age[k] = 0                  # a restored drive gets a fresh episode budget
            elif crashed or age[k] >= a.episode_frames:
                stats["new"] += 1
                rates[k], infos[k] = new_start(k)
                age[k], restores[k], snaps[k] = 0, 0, []
        if total - last_exam >= a.exam_every:
            last_exam = total
            rec = {"frames": total, "err_hz": round(float(np.mean(stats["err"])), 2),
                   "restores": stats["restores"], "episodes": stats["new"],
                   "drift": np.round(ip.drift(), 4).tolist(), "mins": round((time.time() - t) / 60, 1),
                   "exam": exam_now(total)}
            stats = {"frames": 0, "err": [], "restores": 0, "new": 0, "progress": []}
            for f in range(len(etas)):
                np.savez_compressed(Path(a.out) / f"fly{f}_f{total}.npz", **ip.state(f))
            _log(a.out, rec)
            # the exam moved every slot: start fresh drives
            for k in range(B):
                rates[k], infos[k] = new_start(k)
            age[:], restores[:] = 0, 0
            snaps = [[] for _ in range(B)]
    fleet.close()


def _step_scaled(rp, counts, reward, learn, window, eta_slot):
    """``BatchReward.step`` with a per-slot learning rate (one rule, several flies)."""
    import cupy as cp
    p = rp.p
    ae = np.exp(-window / p.tau_elig_ms)
    ap = 1 - np.exp(-window / p.tau_post_ms)
    ar = 1 - np.exp(-window / p.tau_reward_ms)
    c_post = counts[:, rp.d_upost].astype(cp.float32)
    dev = (c_post - rp.post_mean)[:, rp.post_k]
    rp.elig = ae * rp.elig + counts[:, rp.d_upre].astype(cp.float32)[:, rp.pre_k] * dev
    rp.post_mean += ap * (c_post - rp.post_mean)
    adv = reward - rp.r_mean
    rp.r_mean += ar * adv
    if learn.any():
        scale = cp.asarray((eta_slot * adv).astype(np.float32))[:, None]
        rp.apply(scale * rp.elig, learn)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("practice")
    pr.add_argument("--rom", default="F-Zero (USA).sfc")
    pr.add_argument("--exam", default="start.state")
    pr.add_argument("--init")
    pr.add_argument("--etas", default="2e-4,5e-4,1e-3,3e-3")
    pr.add_argument("--batch", type=int, default=8)
    pr.add_argument("--drives-total", type=int, default=100)
    pr.add_argument("--practice-frames", type=int, default=2400)
    pr.add_argument("--exam-frames", type=int, default=3600)
    pr.add_argument("--exam-every", type=int, default=10)
    pr.add_argument("--drives", type=int, default=4, help="exam drives per fly")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", required=True)
    dg = sub.add_parser("dagger")
    dg.add_argument("--rom", default="F-Zero (USA).sfc")
    dg.add_argument("--exam", default="start.state")
    dg.add_argument("--line", default="runs/pilot/mute_city_line.npz")
    dg.add_argument("--init")
    dg.add_argument("--etas", default="1e-4,3e-4,1e-3,3e-3")
    dg.add_argument("--batch", type=int, default=8)
    dg.add_argument("--frames", type=int, default=2_000_000, help="total training fly-frames")
    dg.add_argument("--episode-frames", type=int, default=1800)
    dg.add_argument("--snap-every", type=int, default=120)
    dg.add_argument("--max-restores", type=int, default=3)
    dg.add_argument("--p-grid", type=float, default=0.25)
    dg.add_argument("--eta-bias", type=float, default=0.0,
                    help="intrinsic plasticity of the instructed DNs (mV per Hz of error per frame)")
    dg.add_argument("--mirror", type=float, default=0.5, help="share of training drives in the mirror world")
    dg.add_argument("--exam-frames", type=int, default=12000)
    dg.add_argument("--exam-every", type=int, default=200_000)
    dg.add_argument("--drives", type=int, default=2, help="exam drives per fly")
    dg.add_argument("--eta-deep", default="0",
                    help="deep plasticity (synapses onto the DNs' inputs), learning rate relative to eta")
    dg.add_argument("--seed", type=int, default=0)
    dg.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"practice": practice, "dagger": dagger}[a.cmd](a)


if __name__ == "__main__":
    main()
