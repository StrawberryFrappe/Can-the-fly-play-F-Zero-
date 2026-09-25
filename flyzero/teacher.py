"""Teacher v2: a clone of the owner, with the pilot's corrections where the owner never was.

The first pilot (``pilot.py``, pure pursuit) was slower than the owner (2'23" and 3rd, against the
owner's 2'10"-2'13" and 1st) and drove unlike them: D-pad on 42% of frames, where the owner
steers on 14% and leans on 23% (D-pad taps cost speed, leans don't). So the teacher is now
learned from the owner's own races:

1. **Clone** (behaviour cloning): a small network maps what the game's memory says about the car
   relative to the racing line (sideways offset, heading error, how the line bends at 8 distances
   ahead, speed, and how offset and heading are changing) to the owner's buttons at that moment
   (steer, lean, gas, boost). On a held-out race it follows the owner's steering at r = 0.9.
2. **Corrections** (DAgger, Ross et al. 2011): alone, the clone drifts off the owner's line into
   places the owner never was and stalls within a lap. So it drives (with random shoves too),
   and wherever it is off the line the owner-style pilot (``pilot.py``, style "owner", settings
   found by search) labels what to do; it retrains on the owner's frames plus those, a few
   rounds. Owner-like on the line, the pilot's recovery off it.

Like the pilot it reads RAM, so it is **not the fly** and never drives the fly's car: it labels
what to do from wherever the student is. At run time it is plain numpy (no torch in the
emulator workers). It rides in a line file with ``style = "clone"``, and ``Pilot.from_line``
returns it, so every trainer picks it up.

    python -m flyzero.teacher data --out work/teacher/owner.npz runs/recordings/0[134]_*.npz
    python -m flyzero.teacher train --data work/teacher/owner.npz --pilot-params params.json \\
        --out runs/pilot/mute_city_teacher.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pilot import pose

AHEAD = (4, 8, 12, 16, 24, 32, 48, 64)   # line points ahead (16 map units apart) for the bends
LAG = 4                                  # frames over which offset / heading changes are measured
N_IN = 3 + len(AHEAD) + 2


class Track:
    """The racing line, and where a car is relative to it."""

    def __init__(self, points: np.ndarray):
        self.pts = np.asarray(points, float)
        self.n = len(self.pts)
        d = np.roll(self.pts, -1, 0) - self.pts
        self.head = np.arctan2(d[:, 1], d[:, 0])

    def locate(self, ram, k_prev=None):
        """(features without rates (N_IN - 2), nearest line index)."""
        x, y, h, v = pose(ram)
        cand = np.arange(self.n) if k_prev is None else (k_prev + np.arange(-40, 41)) % self.n
        dd = np.hypot(self.pts[cand, 0] - x, self.pts[cand, 1] - y)
        k = int(cand[np.argmin(dd)])
        if dd.min() > 400:   # lost (a big bounce): search the whole line
            k = int(np.argmin(np.hypot(self.pts[:, 0] - x, self.pts[:, 1] - y)))
        th = self.head[k]
        dx, dy = x - self.pts[k, 0], y - self.pts[k, 1]
        lat = -np.sin(th) * dx + np.cos(th) * dy
        eh = np.angle(np.exp(1j * (h - th)))
        bends = [np.angle(np.exp(1j * (self.head[(k + a) % self.n] - th))) for a in AHEAD]
        return np.array([lat / 100.0, eh, v / 2000.0] + bends, np.float32), k


def with_rates(F: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Append the change of offset and heading over LAG frames (0 for a race's first frames)."""
    R = np.zeros((len(F), 2), np.float32)
    for g in np.unique(groups):
        i = np.flatnonzero(groups == g)
        R[i[LAG:]] = F[i[LAG:], :2] - F[i[:-LAG], :2]
    return np.concatenate([F, R * 5], 1)


def mlp_forward(W: dict, x: np.ndarray) -> np.ndarray:
    h = np.maximum(x @ W["w0"].T + W["b0"], 0)
    h = np.maximum(h @ W["w1"].T + W["b1"], 0)
    return h @ W["w2"].T + W["b2"]


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


class OwnerClone:
    """Pilot-compatible teacher: ``intent(ram)`` (steer, lean, gas, brake, boost) and ``buttons``."""

    def __init__(self, points, W: dict):
        self.track, self.W = Track(points), W
        self.reset()

    @classmethod
    def from_line(cls, line) -> "OwnerClone":
        return cls(line["points"], {k: np.asarray(line[f"clone_{k}"], np.float32)
                                    for k in ("w0", "b0", "w1", "b1", "w2", "b2")})

    def reset(self):
        self.k = None
        self.hist = []
        self.acc = np.zeros(3)
        self.x = None

    def features(self, ram) -> np.ndarray:
        f, self.k = self.track.locate(ram, self.k)
        self.hist.append(f[:2])
        if len(self.hist) > LAG + 1:
            del self.hist[0]
        r = (f[:2] - self.hist[0]) * 5 if len(self.hist) > LAG else np.zeros(2, np.float32)
        self.x = np.r_[f, r].astype(np.float32)
        return self.x

    def intent(self, ram) -> np.ndarray:
        o = mlp_forward(self.W, self.features(ram))
        ps, pl = _softmax(o[0:3]), _softmax(o[3:6])
        gas, boost = 1 / (1 + np.exp(-np.clip(o[6:8], -50, 50)))
        return np.array([ps[2] - ps[1], pl[2] - pl[1], gas, 0.0, boost], np.float32)

    def buttons(self, ram, x: np.ndarray | None = None) -> dict:
        s, lean, gas, brake, boost = self.intent(ram) if x is None else x
        taps = []
        for j, u in enumerate((s, lean, gas - 1.0)):   # duty cycles -> taps, like the pilot
            self.acc[j] += u
            t = 1 if self.acc[j] >= 0.5 else -1 if self.acc[j] <= -0.5 else 0
            self.acc[j] -= t
            taps.append(t)
        return {"B": taps[2] >= 0, "LEFT": taps[0] < 0, "RIGHT": taps[0] > 0, "L": taps[1] < 0,
                "R": taps[1] > 0, "Y": brake > 0.5, "A": boost > 0.5}


# ---------------------------------------------------------------- building it (torch, main process)

def owner_data(rom: str, recordings: list[str], line: str, out: str):
    """Replay the owner's races (bit-exact from their inputs); per frame from GO to the 5th lap line:
    features of the state the owner saw and the buttons they then pressed (steer class, lean
    class, gas, boost). ``on_ram`` gets the RAM after each frame, so each frame's buttons pair
    with the previous callback's features."""
    from .record import replay_all

    tr = Track(np.load(line)["points"])
    F, Y, G = [], [], []
    g0 = 0
    for rec in recordings:
        state = {}

        def on_ram(race, ram, b, info):
            s = state.setdefault(race, {"k": None, "go": False, "seen": None})
            if s["seen"] is not None:   # the state before this frame, and the buttons pressed from it
                F.append(s["seen"])
                Y.append([1 if b.get("LEFT") else 2 if b.get("RIGHT") else 0, 1 if b.get("L") else 2 if b.get("R") else 0,
                          int(bool(b.get("B"))), int(bool(b.get("A")))])
                G.append(g0 + race)
            s["go"] = s["go"] or info["speed"] > 0
            if s["go"] and info["lap"] < 5:
                s["seen"], s["k"] = tr.locate(ram, s["k"])
            else:
                s["seen"] = None

        res = replay_all(rom, rec, on_ram=on_ram)
        bad = [r for r in res if r["mismatches"]]
        assert not bad, f"{rec}: replay out of sync {bad}"
        g0 += len(res)
    F, Y, G = np.array(F, np.float32), np.array(Y, np.int64), np.array(G)
    np.savez(out, X=with_rates(F, G), Y=Y, G=G)
    print(f"{len(F)} owner frames from {g0} races", flush=True)


def build_net():
    import torch.nn as nn

    class Clone(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(N_IN, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 8))

        def forward(self, x):
            return self.net(x)

    return Clone()


def soft_owner(Y: np.ndarray) -> np.ndarray:
    T = np.zeros((len(Y), 8), np.float32)
    T[np.arange(len(Y)), Y[:, 0]] = 1
    T[np.arange(len(Y)), 3 + Y[:, 1]] = 1
    T[:, 6], T[:, 7] = Y[:, 2], Y[:, 3]
    return T


def soft_intent(x: np.ndarray) -> np.ndarray:
    s, le = np.clip(x[:, 0], -1, 1), np.clip(x[:, 1], -1, 1)
    return np.stack([1 - abs(s), np.maximum(-s, 0), np.maximum(s, 0), 1 - abs(le), np.maximum(-le, 0),
                     np.maximum(le, 0), x[:, 2], x[:, 4]], 1).astype(np.float32)


def fit(X, T, epochs=25, seed=0) -> dict:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    m = build_net()
    opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-5)
    Xt, Tt = torch.tensor(X), torch.tensor(T)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 512):
            j = perm[b:b + 512]
            o, t = m(Xt[j]), Tt[j]
            loss = (-(t[:, :3] * F.log_softmax(o[:, :3], 1)).sum(1) - (t[:, 3:6] * F.log_softmax(o[:, 3:6], 1)).sum(1)
                    + F.binary_cross_entropy_with_logits(o[:, 6], t[:, 6], reduction="none")
                    + F.binary_cross_entropy_with_logits(o[:, 7], t[:, 7], reduction="none")).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    lin = [layer for layer in m.net if hasattr(layer, "weight")]
    return {f"{k}{i}": (layer.weight if k == "w" else layer.bias).detach().numpy().astype(np.float32)
            for i, layer in enumerate(lin) for k in ("w", "b")}


def _drive(job):
    """One race by the clone (plus random shoves); off-line frames labelled by the pilot."""
    rom, state, line, W, pilot_kw, noise, seed, frames, off_lat, off_eh = job
    from .games import FZero
    from .pilot import Pilot, PilotParams

    game = FZero(rom, skip_menu=True)
    clone = OwnerClone(line["points"], W)
    pilot = Pilot(line["points"], line["speed"], PilotParams(**pilot_kw))
    game.em.set_state(state)
    game.frame_no, game.info, game._last_move, game._empty = 0, {}, 0, 0
    game._press({})
    rng = np.random.default_rng(seed)
    shove, shove_dir, X, T = 0, None, [], []
    for t in range(frames):
        ram = game.ram()
        xc, xp = clone.intent(ram), pilot.intent(ram)
        f = clone.x
        if t > 150 and (abs(f[0]) > off_lat or abs(f[1]) > off_eh):
            X.append(f)
            T.append(xp)
        b = clone.buttons(ram, xc)
        if noise:
            if shove == 0 and rng.random() < noise:
                shove, shove_dir = int(rng.integers(5, 30)), str(rng.choice(["LEFT", "RIGHT"]))
            if shove:
                b["LEFT"] = b["RIGHT"] = False
                b[shove_dir] = True
                shove -= 1
        game.step(b)
        if game.info["done"]:
            break
    info = game.info
    return (np.array(X, np.float32).reshape(-1, N_IN), np.array(T, np.float32).reshape(-1, 5),
            {"frames": t + 1, "lap": info["lap"], "rank": info["rank"], "finished": bool(info["finished"])})


def train(a):
    import multiprocessing as mp

    d = np.load(a.data)
    Xo, To = d["X"], soft_owner(d["Y"])
    line = dict(np.load(a.line))
    pilot_kw = {**json.loads(Path(a.pilot_params).read_text()), "style": "owner"}   # the corrector
    state = Path(a.state).read_bytes()
    W = fit(Xo, To, a.epochs, a.seed)
    Xp, Tp = np.zeros((0, N_IN), np.float32), np.zeros((0, 8), np.float32)
    log = []
    best = (-1.0, None, None)   # (score, round, weights): each round's races test the weights they drove with
    for r in range(a.rounds):
        jobs = [(a.rom, state, line, W, pilot_kw, a.noise if s % 4 else 0.0, 1000 * r + s, a.frames, 0.3, 0.15)
                for s in range(a.races)]
        # one race per worker process: one emulator per process
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
            res = pool.map(_drive, jobs)
        for X, T, _ in res:
            Xp, Tp = np.concatenate([Xp, X]), np.concatenate([Tp, soft_intent(T)])
        races = [x[2] for x in res]
        rec = {"round": r, "clean_top3": sum(x["finished"] for x, j in zip(races, jobs) if not j[5]),
               "shoved_top3": sum(x["finished"] for x, j in zip(races, jobs) if j[5]),
               "races": [(x["lap"], x["rank"], x["frames"]) for x in races], "pilot_samples": len(Xp)}
        print(json.dumps(rec), flush=True)
        log.append(rec)
        # score: finishing clean races first, then how far the shoved races got
        score = 100 * rec["clean_top3"] + 10 * rec["shoved_top3"] + np.mean([x["lap"] for x in races])
        if score > best[0]:
            best = (score, r, W)
        if r < a.rounds - 1:
            W = fit(np.concatenate([Xo, Xp]), np.concatenate([To, Tp]), a.epochs, a.seed + r + 1)
    _, r_best, W = best   # the last refit is untested: never ship weights no race has checked
    out = {**line, "style": np.array("clone"), **{f"clone_{k}": v for k, v in W.items()},
           "clone_log": np.array(json.dumps(log)), "clone_pilot": np.array(json.dumps(pilot_kw)),
           "clone_round": np.array(r_best)}
    np.savez(a.out, **out)
    print(f"teacher (weights of round {r_best}) saved to {a.out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("data", help="owner features + buttons from recordings")
    p.add_argument("recordings", nargs="+")
    p.add_argument("--rom", default="F-Zero (USA).sfc")
    p.add_argument("--line", default="runs/pilot/mute_city_line.npz")
    p.add_argument("--out", required=True)
    p = sub.add_parser("train", help="clone + DAgger corrections -> teacher line file")
    p.add_argument("--data", required=True)
    p.add_argument("--pilot-params", required=True, help="json: owner-style pilot settings (the corrector)")
    p.add_argument("--rom", default="F-Zero (USA).sfc")
    p.add_argument("--state", default="start.state")
    p.add_argument("--line", default="runs/pilot/mute_city_line.npz")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--races", type=int, default=16)
    p.add_argument("--frames", type=int, default=12000)
    p.add_argument("--noise", type=float, default=0.004, help="per-frame chance of a shove (3 of 4 races)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "data":
        owner_data(a.rom, a.recordings, a.line, a.out)
    else:
        train(a)


if __name__ == "__main__":
    main()
