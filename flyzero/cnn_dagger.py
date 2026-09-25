"""Baseline v2: a plain convolutional network (no fly) that drives from the screen.

The comparison for the augmented fly, and the model most likely to finish consistently: it
sees the game picture itself (56x64 colour, now and 4 frames ago) and presses the buttons
directly. No brain, so it trains on many emulators at once, far faster than anything with the
fly in the loop.

Taught by DAgger with the pilot as instructor (Ross et al. 2011): the network drives, the pilot
labels every frame it visits; some episodes the pilot drives with random shoves (DART, Laskey
et al. 2017) so the data also shows how to recover. Labels are the pilot's intent (tap duty
cycles), learned as probabilities; the network samples its buttons from them, so every race
differs a little, like the fly's spiking noise.

    python -m flyzero.cnn_dagger --out work/cnn1 --line runs/pilot/mute_city_line.npz
    python -m flyzero.cnn_dagger --out work/cnn1 --resume work/cnn1 --eval 24     # 24 races
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

H, W = 56, 64   # frame[::4, ::4]
LAG = 4         # the second picture is this many frames old (motion)


def build_net(width: int = 256):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(6, 32, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(), nn.Linear(64 * 7 * 8, width), nn.ReLU())
            self.steer, self.lean = nn.Linear(width, 3), nn.Linear(width, 3)
            self.gas, self.boost = nn.Linear(width, 1), nn.Linear(width, 1)

        def forward(self, x):
            h = self.body(x)
            return self.steer(h), self.lean(h), self.gas(h)[:, 0], self.boost(h)[:, 0]

    return Net()


def soft_targets(x) -> np.ndarray:
    """Pilot intent (steer, lean, gas, brake, boost) -> (none, left, right) x2, gas, boost."""
    x = np.asarray(x, np.float32)
    s, le = np.clip(x[..., 0], -1, 1), np.clip(x[..., 1], -1, 1)
    return np.stack([1 - abs(s), np.maximum(-s, 0), np.maximum(s, 0),
                     1 - abs(le), np.maximum(-le, 0), np.maximum(le, 0),
                     np.clip(x[..., 2], 0, 1), np.clip(x[..., 4], 0, 1)], -1)


def mirror(xb, tb):
    """Mirror world: flip the pictures, swap left and right."""
    xb = xb.flip(-1)
    tb = tb.clone()
    tb[:, [1, 2]] = tb[:, [2, 1]]
    tb[:, [4, 5]] = tb[:, [5, 4]]
    return xb, tb


class Driver:
    """Per-slot picture history and button sampling."""

    def __init__(self, net, dev, n, rng):
        self.net, self.dev, self.n, self.rng = net, dev, n, rng
        self.hist = [[] for _ in range(n)]

    def reset(self, k, view):
        self.hist[k] = [view] * (LAG + 1)

    def push(self, k, view):
        self.hist[k].append(view)
        del self.hist[k][0]

    def obs(self, ks) -> np.ndarray:
        return np.stack([np.concatenate([self.hist[k][-1], self.hist[k][0]], -1) for k in ks])

    def act(self, ks, greedy=False):
        import torch

        x = torch.as_tensor(self.obs(ks), device=self.dev).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            s, le, g, b = self.net(x)
            ps, pl = torch.softmax(s, 1).cpu().numpy(), torch.softmax(le, 1).cpu().numpy()
            pg, pb = torch.sigmoid(g).cpu().numpy(), torch.sigmoid(b).cpu().numpy()
        out = []
        for j in range(len(ks)):
            if greedy:
                si, li, gi, bi = ps[j].argmax(), pl[j].argmax(), pg[j] > 0.5, pb[j] > 0.5
            else:
                si = self.rng.choice(3, p=ps[j] / ps[j].sum())
                li = self.rng.choice(3, p=pl[j] / pl[j].sum())
                gi, bi = self.rng.random() < pg[j], self.rng.random() < pb[j]
            out.append({"LEFT": si == 1, "RIGHT": si == 2, "L": li == 1, "R": li == 2,
                        "B": bool(gi), "A": bool(bi)})
        return out


def run(a):
    import torch
    import torch.nn.functional as F

    from .live import save_run
    from .pool import EmulatorPool
    from .record import buttons_to_mask
    from .tune import Progress

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    line = dict(np.load(a.line))
    segments = int(line.get("segments", 59))
    B = a.batch
    pool = EmulatorPool(a.rom, None, B, line=line, pixels="view")
    start = Path(a.exam).read_bytes()
    net = build_net(a.width).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    drv = Driver(net, dev, B, rng)
    src = Path(a.resume) if a.resume else None
    if src and (src / "cnn.pt").exists():
        net.load_state_dict(torch.load(src / "cnn.pt", map_location=dev, weights_only=True))
        print(f"resumed {src / 'cnn.pt'}", flush=True)

    def exam(n_races, record=True, tag=""):
        """Every slot races alone from the grid, the network sampling its buttons."""
        net.eval()
        results = []
        while len(results) < n_races:
            m = min(B, n_races - len(results))
            ks = list(range(m))
            _, infos = pool.load([start] * m, which=ks)
            for k in ks:
                drv.reset(k, infos[k]["view"])
            progs = [Progress(segments) for _ in ks]
            masks = [[] for _ in ks]
            live = set(ks)
            res = {}
            for i in range(a.exam_frames):
                act = list(live)
                bts = drv.act(act, greedy=a.greedy)
                outs = pool._all([("step", b, False) for b in bts], act)
                for k, b, o in zip(act, bts, outs):
                    info = o[1]
                    drv.push(k, info["view"])
                    masks[k].append(buttons_to_mask(b))
                    progs[k].update(info["segment"], i)
                    if info["done"] or i == a.exam_frames - 1:
                        res[k] = {"progress": progs[k].total, "lap": info["lap"], "frames": i + 1,
                                  "rank": info["rank"], "energy": info["energy"],
                                  "laps5": info["lap"] >= 5, "finished": bool(info["finished"])}
                        live.discard(k)
                if not live:
                    break
            for k in ks:
                r = res[k]
                j = len(results)
                if record and (r["finished"] or r["laps5"]):
                    kind = "finish" if r["finished"] else "5laps"
                    save_run(out / f"{kind}{tag}_{j}.npz", start, np.array(masks[k]), np.zeros((len(masks[k]), 9)),
                             "cnn", **r)
                results.append(r)
        prog = [r["progress"] for r in results]
        return {"n": len(results), "finished": sum(r["finished"] for r in results),
                "laps5": sum(r["laps5"] for r in results), "mean_progress": round(float(np.mean(prog)), 1),
                "ranks_at_5": [r["rank"] for r in results if r["laps5"]],
                "finish_frames": [r["frames"] for r in results if r["finished"]], "each": prog}

    if a.eval:
        rep = exam(a.eval, tag="_eval")
        print(json.dumps({"eval": rep}), flush=True)
        (out / "eval.json").write_text(json.dumps(rep))
        pool.close()
        return rep

    # dataset: disk-backed reservoir of (picture pair, pilot's soft targets)
    X = np.lib.format.open_memmap(out / "views.npy", "w+", np.uint8, (a.cap, H, W, 6))
    T = np.zeros((a.cap, 8), np.float32)
    n = seen = 0

    def add(views, targets):
        nonlocal n, seen
        for v, t in zip(views, targets):
            seen += 1
            if n < a.cap:
                j = n
                n += 1
            else:
                j = int(rng.integers(0, seen))
                if j >= a.cap:
                    continue
            X[j], T[j] = v, t

    # mid-race starts (the pilot's own race, snapshots every 300 frames): later laps get data too
    snaps = pool.pilot_drive(start, 14000, 300)
    print(f"{len(snaps)} mid-race starts from the pilot's race", flush=True)

    def train(epochs):
        net.train()
        Tt = torch.as_tensor(T[:n], device=dev)
        for _ in range(epochs):
            perm = np.random.permutation(n)
            for b0 in range(0, n, 256):
                idx = np.sort(perm[b0:b0 + 256])
                xb = torch.as_tensor(X[idx], device=dev).permute(0, 3, 1, 2).float() / 255.0
                tb = Tt[idx]
                flip = torch.rand(len(idx), device=dev) < 0.5
                if flip.any():
                    xf, tf = mirror(xb[flip], tb[flip])
                    xb, tb = xb.clone(), tb.clone()
                    xb[flip], tb[flip] = xf, tf
                s, le, g, bo = net(xb)
                loss = -(tb[:, 0:3] * F.log_softmax(s, 1)).sum(1).mean() \
                    - (tb[:, 3:6] * F.log_softmax(le, 1)).sum(1).mean() \
                    + F.binary_cross_entropy_with_logits(g, tb[:, 6]) \
                    + a.w_boost * F.binary_cross_entropy_with_logits(bo, tb[:, 7])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        return float(loss.detach())

    def begin(k):
        """New episode in slot k: from the grid or a mid-race start; pilot (DART) or network."""
        if rng.random() < a.p_grid:
            _, inf = pool.load([start], which=[k])
        else:
            _, inf = pool.restore([snaps[int(rng.integers(len(snaps)))]], [k])
        drv.reset(k, inf[0]["view"])
        return {"pilot": rng.random() < a.dart, "age": 0}

    def collect(frames):
        net.eval()
        eps = [begin(k) for k in range(B)]
        done = 0
        while done < frames:
            stud = [k for k in range(B) if not eps[k]["pilot"]]
            pil = [k for k in range(B) if eps[k]["pilot"]]
            msgs, which = [], []
            if stud:
                for k, b in zip(stud, drv.act(stud)):
                    msgs.append(("step", b, False))
                    which.append(k)
            for k in pil:
                msgs.append(("step_pilot", a.dart_noise, False))
                which.append(k)
            views, targets = [], []
            for k, o in zip(which, pool._all(msgs, which)):
                info = o[1]
                drv.push(k, info["view"])   # picture and label both of the state after this step
                if info["racing"] and info["speed"] > 0:
                    views.append(drv.obs([k])[0])
                    targets.append(soft_targets(info["pilot"]))
                eps[k]["age"] += 1
                if info["done"] or eps[k]["age"] >= a.episode_frames:
                    eps[k] = begin(k)
            add(views, targets)
            done += len(which)

    t0 = time.time()
    best = -1.0
    log = open(out / "log.jsonl", "a")
    collect(a.round_frames * (1 if src else 2))   # round 0: a double helping (mostly the pilot's DART drives)
    for r in range(a.rounds + 1):
        loss = train(a.epochs)
        rep = exam(a.exam_races, tag=f"_r{r}")
        score = rep["finished"] * 1000 + rep["mean_progress"]
        rec = {"round": r, "dataset": n, "seen": seen, "loss": round(loss, 3), "mins": round((time.time() - t0) / 60, 1),
               "exam": rep}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save(net.state_dict(), out / "cnn_last.pt")
        if score > best:
            best = score
            torch.save(net.state_dict(), out / "cnn.pt")
        if r < a.rounds:
            collect(a.round_frames)
    pool.close()
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom", default="F-Zero (USA).sfc")
    ap.add_argument("--exam", default="start.state", help="grid state (races start here)")
    ap.add_argument("--line", default="runs/pilot/mute_city_line.npz", help="racing line + pilot settings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", help="directory with cnn.pt to start from")
    ap.add_argument("--batch", type=int, default=8, help="emulators")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--round-frames", type=int, default=100_000)
    ap.add_argument("--episode-frames", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--cap", type=int, default=400_000, help="dataset size (disk: ~21 KB per sample)")
    ap.add_argument("--p-grid", type=float, default=0.3, help="share of episodes that start on the grid")
    ap.add_argument("--dart", type=float, default=0.3, help="share of episodes the pilot drives (with shoves)")
    ap.add_argument("--dart-noise", type=float, default=0.004, help="per-frame chance of a shove")
    ap.add_argument("--exam-races", type=int, default=8)
    ap.add_argument("--exam-frames", type=int, default=16000)
    ap.add_argument("--eval", type=int, default=0, help="only race this many times (with --resume)")
    ap.add_argument("--greedy", action="store_true", help="most likely buttons instead of sampling")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-boost", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
