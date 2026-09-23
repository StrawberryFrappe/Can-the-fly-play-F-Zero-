"""The augmented fly: a FlyWire fly with an artificial neural implant. NOT the fly alone.

Only used once the natural ladder (lessons, DAgger, deeper plasticity, practice) had plateaued,
as agreed with the owner. Everything here is ours and is reported separately:

* **Host**: the fly's brain exactly as the natural ladder left it (the corrected FlyWire model
  with its own learned synapses), running on the GPU fleet as usual.
* **Electrodes (read)**: smoothed spike rates of ~1,100 of its neurons: the presynaptic partners
  of its motor DNs ("L1") and its visual projection neurons, the ones that fire while driving.
  The implant sees nothing else: no pixels, no RAM.
* **Implant**: a small multilayer perceptron (PyTorch, on the GPU) from those rates to an intent
  (steer, lean, gas, boost). It is trained by DAgger with the pilot as instructor, on the whole
  aggregated dataset (ordinary supervised learning, many epochs).
* **Optogenetics (write)**: a closed-loop current clamp. For each motor DN group an integral
  controller adjusts an injected current (mV of steady depolarisation, like ``Brain.set_bias``)
  until the group fires at the rate the implant asks for (the same target rates the natural
  fly's lessons used). The DNs still spike, and the same readout turns them into buttons.

    python -m flyzero.implant --host work/r3g/fly0_f800000.npz --out work/aug
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CLAMP_GROUPS = ["a02L", "a02R", "g02L", "g02R", "a01L", "a01R", "gas", "brake", "gf"]


def electrodes(conn, decode_file: str | None = "work/decode_data.npz", l2_top: int = 0) -> np.ndarray:
    """L1 (inputs of the motor DNs) and visual projection neurons; with a recording of the fly
    driving, only the ones that fired (the others carry nothing and cost memory)."""
    from .batch import plastic_positions
    import scipy.sparse as sp

    w = sp.csr_matrix(conn.weights, dtype=np.float32)
    rows = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
    l1 = np.unique(rows[plastic_positions(conn)])
    sc = conn.neurons["super_class"].fillna("").to_numpy()
    idx = np.union1d(l1, np.flatnonzero(sc == "visual_projection"))
    if decode_file and Path(decode_file).exists():
        d = np.load(decode_file)
        active = d["keep"][d["X"].sum(0) > 20]
        idx = np.intersect1d(idx, active)
        if l2_top:   # plus the L2 neurons (inputs of L1) whose rates follow the pilot's steering best
            keep, X = d["keep"], d["X"]
            cols = np.searchsorted(keep, np.setdiff1d(d["l2"], idx))
            cols = cols[X[:, cols].sum(0) > 20]
            steer = np.convolve(d["Y"][:, 0], np.ones(15) / 15, "same")
            k = np.ones(5) / 5
            r = np.array([abs(np.corrcoef(np.convolve(X[:, c].astype(float), k, "same"), steer)[0, 1]) for c in cols])
            idx = np.union1d(idx, keep[cols[np.argsort(-np.nan_to_num(r))[:l2_top]]])
    # no electrodes on descending neurons: the clamp drives the motor DNs, and reading them (or
    # DNs they excite) lets the implant copy its own last command (causal confusion; the first
    # L2 implant's exams got worse while its training loss kept falling)
    return idx[sc[idx] != "descending"]


class Clamp:
    """Closed-loop current injection into the motor DN groups, per slot."""

    def __init__(self, fleet, gain: float = 0.02, limit: float = 20.0):
        import cupy as cp

        from .batch import GROUPS

        self.fleet, self.gain, self.limit = fleet, gain, limit
        conn, B = fleet.conn, fleet.B
        self.groups = [conn.find(*GROUPS[g]) for g in CLAMP_GROUPS]
        self.slots = [fleet.brain.slots(ix) for ix in self.groups]
        self.base = [fleet.brain.bias_ext[:, j].copy() for j in self.slots]
        self.u = np.zeros((B, len(CLAMP_GROUPS)), np.float32)
        self.rate = np.zeros((B, len(CLAMP_GROUPS)), np.float32)
        self._idx = [cp.asarray(ix) for ix in self.groups]

    def reset(self, slots):
        self.u[np.atleast_1d(slots)] = 0
        self.rate[np.atleast_1d(slots)] = 0

    def step(self, counts, targets: np.ndarray, active: np.ndarray, window_ms: float):
        """``targets``: (B, groups) Hz; ``active``: slots whose implant is switched on."""
        import cupy as cp

        a = 1 - np.exp(-window_ms / 80.0)
        inst = np.stack([cp.asnumpy(counts[:, ix].mean(1)) for ix in self._idx], 1) * (1000.0 / window_ms)
        self.rate += a * (inst - self.rate)
        self.u += self.gain * (targets - self.rate) * active[:, None]
        self.u = np.clip(self.u, -self.limit, self.limit) * active[:, None]
        for g, j in enumerate(self.slots):
            self.fleet.brain.bias_ext[:, j] = self.base[g] + cp.asarray(self.u[:, g])[:, None]


def build_net(n_in: int):
    import torch.nn as nn

    class Implant(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(n_in, 256), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(256, 128), nn.ReLU())
            self.steer, self.lean, self.gas, self.boost = (nn.Linear(128, 3), nn.Linear(128, 3),
                                                           nn.Linear(128, 2), nn.Linear(128, 2))
            self.analog = nn.Linear(128, 2)   # --taps: continuous steer and lean, like the pilot's

        def forward(self, x):
            h = self.body(x)
            return self.steer(h), self.lean(h), self.gas(h), self.boost(h), self.analog(h).tanh()

    return Implant()


def classes(x: np.ndarray) -> np.ndarray:
    """Pilot intent (steer, lean, gas, brake, boost) -> class labels for the implant's heads."""
    s, le, g, _, b = x   # hold-style thresholds (pilot.hold_style): the readout holds buttons
    return np.array([1 if s < -0.5 else 2 if s > 0.5 else 0, 1 if le < -0.5 else 2 if le > 0.5 else 0,
                     int(g > 0.5), int(b > 0.5)], np.int64)


def intent_analog(outs) -> np.ndarray:
    """--taps: continuous steer / lean (tapped by the readout), most likely gas and boost."""
    g, b = (o.argmax(1).cpu().numpy() for o in outs[2:4])
    a = outs[4].cpu().numpy()
    return np.stack([a[:, 0], a[:, 1], g.astype(float), np.zeros(len(g)), b.astype(float)], 1)


def intent_from(outs) -> np.ndarray:
    """Implant outputs -> intent vectors: the most likely class of each head. (Expected values
    turned every "maybe 20% lean" into a target rate that already pressed the button: the first
    augmented fly steered on 94% and leaned on 85% of frames.)"""
    s, le, g, b = (o.argmax(1).cpu().numpy() for o in outs[:4])
    steer = np.where(s == 1, -1.0, np.where(s == 2, 1.0, 0.0))
    lean = np.where(le == 1, -1.0, np.where(le == 2, 1.0, 0.0))
    return np.stack([steer, lean, g.astype(float), np.zeros(len(g)), b.astype(float)], 1)


def run(a):
    import cupy as cp
    import torch
    import torch.nn.functional as F

    from .batch import BatchInstruct, targets
    from .fleet import Fleet, summarize
    from .instruct import InstructParams
    from .live import save_run

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = a.batch
    # --taps: the readout taps at a rate set by the DNs, exactly as the pilot's continuous steering
    # maps onto its duty cycle (160 Hz = full turn, 80 Hz = full lean in the lesson targets)
    fleet = Fleet(a.rom, B, seed=a.seed, line=a.line, deep=True, taps=a.taps, steer_span=150.0, lean_span=70.0)
    host = BatchInstruct(fleet.brain, fleet.conn)
    host.load(np.load(a.host))                     # the natural fly, unchanged from here on
    idx = electrodes(fleet.conn, l2_top=a.l2_top)
    d_idx = cp.asarray(idx)
    print(f"augmented fly: {len(idx)} electrodes, host {a.host}", flush=True)
    clamp = Clamp(fleet)
    ip = InstructParams()
    dev = torch.device("cuda")
    net = build_net(len(idx)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    cap = a.cap
    X = torch.zeros((cap, len(idx)), dtype=torch.float16, device=dev)   # aggregated dataset (GPU)
    Y = torch.zeros((cap, 4), dtype=torch.long, device=dev)
    Ya = torch.zeros((cap, 2), dtype=torch.float32, device=dev)    # the pilot's continuous steer, lean
    n = seen = 0
    mu = sd = None
    ema = cp.zeros((B, len(idx)), cp.float32)
    a_ema = np.float32(1 - np.exp(-fleet.window / 50.0))
    exam_state = Path(a.exam).read_bytes()
    starts = fleet.pool.pilot_drive(exam_state, 12000, 120)
    rng = np.random.default_rng(a.seed)

    def features():
        f = ema * np.float32(1000.0 / fleet.window)
        return f if mu is None else (f - mu) / sd

    def drive(frames, implant_on, collect, grid_only=False, record=False):
        """All B slots drive; returns per-slot results (and recordings)."""
        nonlocal n, ema, seen
        fleet.brain.reset()
        fleet.motor.reset()
        clamp.reset(np.arange(B))
        ema = cp.zeros_like(ema)
        if grid_only or rng.random() < a.p_grid:
            rates, infos = fleet.pool.load([exam_state] * B)
        else:
            rates, infos = fleet.pool.restore([starts[rng.integers(len(starts))] for _ in range(B)], range(B))
        from .tune import Progress
        from .record import buttons_to_mask

        progs = [Progress() for _ in range(B)]
        live = np.ones(B, bool)
        res = [None] * B
        masks, dn = [[] for _ in range(B)], [[] for _ in range(B)]
        for i in range(frames):
            counts = fleet.think(rates)
            ema += a_ema * (counts[:, d_idx].astype(cp.float32) - ema)
            on = implant_on & live
            if implant_on:
                with torch.no_grad():
                    x = torch.as_tensor(features(), device=dev)
                    intent = intent_analog(net(x)) if a.taps else intent_from(net(x))
                tgt = np.array([targets(v, ip) for v in intent], np.float32)
                clamp.step(counts, tgt, on, fleet.window)
            buttons = fleet.motor.update(counts, fleet.window)
            if collect:
                lab = np.array([classes(np.asarray(inf["pilot"])) for inf in infos])
                analog = np.array([np.asarray(inf["pilot"])[:2] for inf in infos], np.float32)
                keep = np.flatnonzero(live & np.array([inf.get("racing", False) for inf in infos]))
                if len(keep):
                    # append while there is room, then replace random old samples (the aggregate
                    # keeps following the implant's own, improving driving)
                    free = min(len(keep), cap - n)
                    where = np.r_[np.arange(n, n + free), rng.integers(0, cap, len(keep) - free)].astype(np.int64)
                    w_t = torch.as_tensor(where, device=dev)
                    X[w_t] = torch.as_tensor(features()[keep], device=dev).half()
                    Y[w_t] = torch.as_tensor(lab[keep], device=dev)
                    Ya[w_t] = torch.as_tensor(analog[keep], device=dev)
                    n += free
                    seen += len(keep)
            if record:
                for k in np.flatnonzero(live):
                    masks[k].append(buttons_to_mask(buttons[k]))
                    dn[k].append(fleet.motor.rates[k].astype(np.float16))
            rates, infos = fleet.pool.step(buttons)
            for k, inf in enumerate(infos):
                if not live[k]:
                    continue
                progs[k].update(inf["segment"], i)
                if inf["done"] or i == frames - 1:
                    live[k] = False
                    res[k] = {"progress": progs[k].total, "lap": inf["lap"], "frames": i + 1,
                              "finished": inf["lap"] >= 5, "rank": inf.get("rank")}
                    if record:
                        res[k]["masks"], res[k]["rates"] = np.array(masks[k]), np.array(dn[k])
            if not live.any():
                break
        return res

    def train(epochs):
        net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for b in range(0, n, 512):
                j = perm[b:b + 512]
                xb, yb = X[j].float(), Y[j]
                outs = net(xb)
                if a.taps:   # steer/lean regressed on the pilot's continuous command; gas, boost classes
                    loss = 4.0 * F.mse_loss(outs[4], Ya[j]) + sum(F.cross_entropy(outs[k], yb[:, k]) for k in (2, 3))
                else:
                    loss = sum(F.cross_entropy(outs[k], yb[:, k]) for k in range(4))
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        return float(loss.detach())

    if a.resume:   # continue a run: its implant, normaliser and dataset
        r0 = Path(a.resume)
        last = r0 / "implant_last.pt"
        net.load_state_dict(torch.load(last if last.exists() else r0 / "implant.pt"))
        net.eval()
        nz = np.load(r0 / "normaliser.npz")
        assert np.array_equal(nz["electrodes"], idx)
        mu, sd = cp.asarray(nz["mu"]), cp.asarray(nz["sd"])
        np.savez(out / "normaliser.npz", mu=nz["mu"], sd=nz["sd"], electrodes=idx)
        if (r0 / "dataset.pt").exists():
            ds = torch.load(r0 / "dataset.pt")
            n = len(ds["Y"])
            X[:n], Y[:n] = ds["X"].to(dev), ds["Y"].to(dev)
            if "Ya" in ds:
                Ya[:n] = ds["Ya"].to(dev)
            del ds
        else:   # no saved dataset: a fresh round driven by the loaded implant
            while n < a.round_frames:
                drive(a.episode_frames, True, True)
        print(f"resumed from {r0}: {n} samples", flush=True)
    else:
        # round 0: the natural fly drives (implant off), the pilot labels: first dataset + normaliser
        t = time.time()
        while n < a.round_frames:
            drive(a.episode_frames, False, True)
        s1 = torch.zeros(X.shape[1], device=dev)
        s2 = torch.zeros(X.shape[1], device=dev)
        for b in range(0, n, 20000):              # in chunks: GPU memory is shared with other flies
            f = X[b:min(n, b + 20000)].float()
            s1 += f.sum(0)
            s2 += (f * f).sum(0)
        mu_t = s1 / n
        sd_t = (s2 / n - mu_t ** 2).clamp(min=0).sqrt() + 1.0
        for b in range(0, n, 20000):
            X[b:min(n, b + 20000)] = ((X[b:min(n, b + 20000)].float() - mu_t) / sd_t).half()
        mu, sd = cp.asarray(mu_t.cpu().numpy()), cp.asarray(sd_t.cpu().numpy())
        np.savez(out / "normaliser.npz", mu=cp.asnumpy(mu), sd=cp.asnumpy(sd), electrodes=idx)
    t = time.time()
    best = -1e9
    for r in range(a.rounds + 1):
        loss = train(a.epochs)
        exam = []
        for _ in range(a.exam_drives // B):
            exam += drive(a.exam_frames, True, False, grid_only=True, record=True)
        top = max(exam, key=lambda x: (x["progress"], -x["frames"]))
        rec = {"round": r, "dataset": n, "loss": round(loss, 3), "mins": round((time.time() - t) / 60, 1),
               "exam": {**summarize(exam), "each": [x["progress"] for x in exam]}}
        print(json.dumps(rec), flush=True)
        with open(out / "log.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        for k, x in enumerate(exam):
            if x["finished"]:
                save_run(out / f"finish_r{r}_{k}.npz", exam_state, x["masks"], x["rates"], "augmented",
                         progress=x["progress"], laps=x["lap"], frames=x["frames"])
        if top["progress"] > best:
            best = top["progress"]
            save_run(out / "best_drive.npz", exam_state, top["masks"], top["rates"], "augmented",
                     progress=top["progress"], laps=top["lap"], frames=top["frames"])
            torch.save(net.state_dict(), out / "implant.pt")
        torch.save(net.state_dict(), out / "implant_last.pt")
        torch.save({"X": X[:n].cpu(), "Y": Y[:n].cpu(), "Ya": Ya[:n].cpu()}, out / "dataset.pt")
        if r == a.rounds:
            break
        target = seen + a.round_frames
        while seen < target:
            # --host-drives: new data only while the natural fly drives (implant off). With the
            # implant driving, DNs it clamps feed back into the electrodes, and later rounds
            # learned to echo their own commands (exams fell from 155 to 67 segments)
            drive(a.episode_frames, not a.host_drives, True)
    fleet.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom", default="F-Zero (USA).sfc")
    ap.add_argument("--exam", default="start.state")
    ap.add_argument("--line", default="runs/pilot/mute_city_line.npz")
    ap.add_argument("--host", required=True, help="the natural fly's weights (deep)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--round-frames", type=int, default=60000, help="new labelled frames per DAgger round")
    ap.add_argument("--episode-frames", type=int, default=2400)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--cap", type=int, default=200000)
    ap.add_argument("--p-grid", type=float, default=0.3)
    ap.add_argument("--exam-frames", type=int, default=16000)
    ap.add_argument("--exam-drives", type=int, default=16)
    ap.add_argument("--host-drives", action="store_true", help="collect new data with the implant off")
    ap.add_argument("--taps", action="store_true", help="continuous steer/lean + tap-rate readout (the pilot's hands)")
    ap.add_argument("--l2-top", type=int, default=0, help="extra electrodes on the most steering-related L2 neurons")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", help="an earlier run's folder: continue with its implant and dataset")
    ap.add_argument("--out", required=True)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
