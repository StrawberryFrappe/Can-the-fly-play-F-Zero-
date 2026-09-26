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

BOOST_P = 0.5
GF_PULSE_HZ = 200.0
CLAMP_GROUPS = ["a02L", "a02R", "g02L", "g02R", "a01L", "a01R", "gas", "brake", "gf"]
# --budget: the implant writes into one control at a time (its DN groups), only while it thinks the
# fly's own choice is wrong; the rest of the time the fly's own DNs decide, with no current injected
CHANNELS = ("steer", "lean", "gas", "boost")
CHANNEL_GROUPS = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3])   # CLAMP_GROUPS -> channel ("gas" = speed: gas + brake)


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

    def __init__(self, fleet, gain: float = 0.02, limit: float = 20.0, tau_ms: float = 80.0):
        import cupy as cp

        from .batch import GROUPS

        self.fleet, self.gain, self.limit, self.tau_ms = fleet, gain, limit, tau_ms
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

    def step(self, counts, targets: np.ndarray, active: np.ndarray, window_ms: float, gate: np.ndarray | None = None):
        """``targets``: (B, groups) Hz; ``active``: slots whose implant is switched on; ``gate``:
        (B, groups) 0/1, groups the implant writes into now (None = all). A closed gate injects
        nothing and holds its controller where it was."""
        import cupy as cp

        a = 1 - np.exp(-window_ms / self.tau_ms)
        inst = np.stack([cp.asnumpy(counts[:, ix].mean(1)) for ix in self._idx], 1) * (1000.0 / window_ms)
        self.rate += a * (inst - self.rate)
        g_on = active[:, None] * (1.0 if gate is None else gate)
        self.u += self.gain * (targets - self.rate) * g_on
        self.u = np.clip(self.u, -self.limit, self.limit) * active[:, None]
        applied = self.u * g_on
        for g, j in enumerate(self.slots):
            self.fleet.brain.bias_ext[:, j] = self.base[g] + cp.asarray(applied[:, g])[:, None]


def save_dataset(out, X, Y, Ya, n, chunk=20000, Yw=None):
    """Chunk by chunk into a memory-mapped file: a whole-array copy in RAM got the process
    OOM-killed on a 7 GB laptop (300k x 2918 fp16 = 1.75 GB)."""
    mm = np.lib.format.open_memmap(out / "dataset_X.npy", "w+", np.float16, (n, X.shape[1]))
    for b in range(0, n, chunk):
        e = min(n, b + chunk)
        mm[b:e] = X[b:e].cpu().numpy()
    mm.flush()
    del mm
    extra = {} if Yw is None else {"Yw": Yw[:n].cpu().numpy()}
    np.savez(out / "dataset_Y.npz", Y=Y[:n].cpu().numpy(), Ya=Ya[:n].cpu().numpy(), **extra)


def load_dataset(r0, X, Y, Ya, dev, chunk=20000, Yw=None) -> int:
    import torch

    mm = np.load(r0 / "dataset_X.npy", mmap_mode="r")
    y = np.load(r0 / "dataset_Y.npz")
    n = min(len(mm), X.shape[0])
    for b in range(0, n, chunk):
        e = min(n, b + chunk)
        X[b:e] = torch.as_tensor(np.asarray(mm[b:e]), device=dev)
    Y[:n] = torch.as_tensor(y["Y"][:n], device=dev)
    Ya[:n] = torch.as_tensor(y["Ya"][:n], device=dev)
    if Yw is not None:
        Yw[:n] = torch.as_tensor(y["Yw"][:n], device=dev) if "Yw" in y else -1
    return n


def build_net(n_in: int, width: int = 256):
    import torch.nn as nn

    class Implant(nn.Module):
        def __init__(self):
            super().__init__()
            h = width // 2
            self.body = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(width, h), nn.ReLU())
            self.steer, self.lean, self.gas, self.boost = (nn.Linear(h, 3), nn.Linear(h, 3),
                                                           nn.Linear(h, 2), nn.Linear(h, 2))
            self.analog = nn.Linear(h, 2)   # --taps: continuous steer and lean, like the pilot's
            self.need = nn.Linear(h, 4)     # --budget: "the fly's own choice is wrong" per channel

        def forward(self, x):
            h = self.body(x)
            return self.steer(h), self.lean(h), self.gas(h), self.boost(h), self.analog(h).tanh(), self.need(h)

    return Implant()


def classes(x: np.ndarray) -> np.ndarray:
    """Pilot intent (steer, lean, gas, brake, boost) -> class labels for the implant's heads."""
    s, le, g, _, b = x   # hold-style thresholds (pilot.hold_style): the readout holds buttons
    return np.array([1 if s < -0.5 else 2 if s > 0.5 else 0, 1 if le < -0.5 else 2 if le > 0.5 else 0,
                     int(g > 0.5), int(b > 0.5)], np.int64)


def intent_analog(outs) -> np.ndarray:
    """--taps: continuous steer / lean (tapped by the readout), most likely gas and boost."""
    import torch

    g = outs[2].argmax(1).cpu().numpy()
    # boost labels are rare (once a lap, ~2% of frames): fire when the implant is fairly sure
    b = (torch.softmax(outs[3], 1)[:, 1] > BOOST_P).cpu().numpy()
    a = outs[4].cpu().numpy()
    return np.stack([a[:, 0], a[:, 1], g.astype(float), np.zeros(len(g)), b.astype(float)], 1)


def load_implant(net, path):
    """Implants from before --budget have no gate head: it starts untrained (and is trained on
    the fly-alone frames of the new data)."""
    import torch

    missing, unexpected = net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    assert not unexpected, unexpected
    if missing:
        print(f"{path}: new heads start untrained: {missing}", flush=True)


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
    pix = a.inputs == "pixels"   # control: the same implant reading the screen instead of the fly's neurons
    fleet = Fleet(a.rom, B, seed=a.seed, line=a.line, deep=True, taps=a.taps, steer_span=150.0, lean_span=70.0,
                  pixels="view" if a.cnn else pix, readout_tau=a.readout_tau)
    host = BatchInstruct(fleet.brain, fleet.conn)
    host.load(np.load(a.host))                     # the natural fly, unchanged from here on
    idx = np.array([-1]) if pix else electrodes(fleet.conn, l2_top=a.l2_top)
    n_in = 3584 if pix else len(idx)
    d_idx = cp.asarray(idx)
    print(f"pixel control: {n_in} pixel inputs (the fly's neurons are not read)" if pix else
          f"augmented fly: {len(idx)} electrodes, host {a.host}", flush=True)
    clamp = Clamp(fleet, gain=a.clamp_gain, tau_ms=a.clamp_tau)
    from .batch import GROUPS as _G

    gf = fleet.conn.find(*_G["gf"])
    fleet.extra_idx = gf
    ip = InstructParams()
    global BOOST_W
    BOOST_W = torch.tensor([1.0, 8.0], device="cuda")
    dev = torch.device("cuda")
    net = build_net(n_in, a.width).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    cap = a.cap
    X = torch.zeros((cap, n_in), dtype=torch.float16, device=dev)   # aggregated dataset (GPU)
    Y = torch.zeros((cap, 4), dtype=torch.long, device=dev)
    Ya = torch.zeros((cap, 2), dtype=torch.float32, device=dev)    # the pilot's continuous steer, lean
    # --budget: was the fly's own choice wrong (per channel)? Only known where the fly acted alone (-1 elsewhere)
    Yw = torch.full((cap, 4), -1, dtype=torch.int8, device=dev)
    n = seen = 0
    budget = float(a.budget)
    theta = np.full(4, 0.5)          # gate thresholds, adapted online so the implant stays within budget
    for src in (a.resume,):          # continue from the thresholds a run converged to
        if src and budget < 1.0 and (Path(src) / "gate.json").exists():
            theta[:] = json.loads((Path(src) / "gate.json").read_text())["theta"]
    a_own = np.float32(1 - np.exp(-1 / 8.0))   # the fly's own recent buttons, ~8 frames
    mu = sd = None
    ema = cp.zeros((B, n_in), cp.float32)
    a_ema = np.float32(1 - np.exp(-fleet.window / 50.0))
    exam_state = Path(a.exam).read_bytes()
    starts = fleet.pool.pilot_drive(exam_state, 12000, 120)
    rng = np.random.default_rng(a.seed)

    cnn = None
    if a.cnn:   # test of the fly's hands: the CNN baseline gives the orders, the clamp and DNs carry them out
        from .cnn_dagger import LAG as CNN_LAG
        from .cnn_dagger import build_net as build_cnn

        cnn = build_cnn(a.cnn_width).to(dev)
        cnn.load_state_dict(torch.load(a.cnn, map_location=dev, weights_only=True))
        cnn.eval()

    def cnn_intent(views):
        x = np.stack([np.concatenate([v[-1], v[0]], -1) for v in views])
        with torch.no_grad():
            s, le, g, b = cnn(torch.as_tensor(x, device=dev).permute(0, 3, 1, 2).float() / 255.0)
            ps, pl = torch.softmax(s, 1).cpu().numpy(), torch.softmax(le, 1).cpu().numpy()
            pg, pb = torch.sigmoid(g).cpu().numpy(), torch.sigmoid(b).cpu().numpy()
        return np.stack([ps[:, 2] - ps[:, 1], pl[:, 2] - pl[:, 1], (pg > 0.5).astype(float), np.zeros(len(pg)),
                         (pb > 0.5).astype(float)], 1)

    def features():
        f = ema * np.float32(1000.0 / fleet.window)
        return f if mu is None else (f - mu) / sd

    def drive(frames, implant_on, collect, grid_only=False, record=False, pilot_noise=None):
        """All B slots drive; returns per-slot results (and recordings). ``pilot_noise``: DART data
        collection, the pilot drives with disturbances while the fly's brain watches (implant off)."""
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

        if cnn is not None:   # --cnn: the baseline's picture history per slot
            views = [[inf["view"]] * (CNN_LAG + 1) for inf in infos]
        progs = [Progress(fleet.segments) for _ in range(B)]
        live = np.ones(B, bool)
        gf_hz = np.zeros(B)
        res = [None] * B
        own = np.zeros((B, 4), np.float32)          # the fly's own steer, lean, gas, boost (smoothed)
        share = np.zeros((B, 4))                    # frames each channel was written by the implant
        racing_frames = np.zeros(B)
        gate_ch = np.ones((B, 4))
        masks, dn = [[] for _ in range(B)], [[] for _ in range(B)]
        for i in range(frames):
            counts = fleet.think(rates, np.repeat(gf_hz[:, None], len(gf), 1))
            if pix:   # features() scales by 1000/window: undo it for pixels
                ema = cp.asarray(np.stack([inf["pix"] for inf in infos]).astype(np.float32)) * \
                    np.float32(fleet.window / 1000.0)
            else:
                ema += a_ema * (counts[:, d_idx].astype(cp.float32) - ema)
            on = implant_on & live
            racing = np.array([inf.get("racing", False) for inf in infos])
            if implant_on:
                with torch.no_grad():
                    x = torch.as_tensor(features(), device=dev)
                    outs = net(x)
                    intent = intent_analog(outs) if a.taps else intent_from(outs)
                    p_need = torch.sigmoid(outs[5]).cpu().numpy()
                    if cnn is not None:   # the orders come from the CNN baseline instead
                        intent = cnn_intent(views)
                tgt = np.array([targets(v, ip) for v in intent], np.float32)
                if budget < 1.0:
                    # write only where the fly is predicted wrong, and within budget: each channel's
                    # threshold rises while the implant is over its share and relaxes (down to
                    # --budget-floor; 0.5 = "more likely wrong than right") while under it
                    gate_ch = (p_need > theta).astype(np.float64)
                    cnt = on & racing
                    if cnt.any():
                        theta[:] = np.clip(theta + a.budget_eta * (gate_ch[cnt].mean(0) - budget), a.budget_floor, 0.999)
                else:
                    gate_ch = np.ones((B, 4))
                share += gate_ch * (on & racing)[:, None]
                racing_frames += on & racing
                clamp.step(counts, tgt, on, fleet.window, gate_ch[:, CHANNEL_GROUPS])
                # the giant fiber: light pulses rather than current (the host's lessons taught it to
                # stay quiet, and +20 mV doesn't make it fire): Poisson stimulation while boosting
                gf_hz = np.where(on & (gate_ch[:, 3] > 0), intent[:, 4] * GF_PULSE_HZ, 0.0)
            buttons = fleet.motor.update(counts, fleet.window)
            act = np.array([[b.get("RIGHT", False) - b.get("LEFT", False), b.get("R", False) - b.get("L", False),
                             b.get("B", False), b.get("A", False)] for b in buttons], np.float32)
            own += a_own * (act - own)
            if collect:
                lab = np.array([classes(np.asarray(inf["pilot"])) for inf in infos])
                analog = np.array([np.asarray(inf["pilot"])[:2] for inf in infos], np.float32)
                if implant_on:   # the fly didn't choose alone: unknown whether its own choice was right
                    wrong = np.full((B, 4), -1, np.int8)
                else:
                    pl = np.array([np.asarray(inf["pilot"]) for inf in infos], np.float32)
                    want = np.stack([np.clip(pl[:, 0], -1, 1), np.clip(pl[:, 1], -1, 1), pl[:, 2], pl[:, 4]], 1)
                    wrong = (np.abs(own - want) > 0.5).astype(np.int8)
                keep = np.flatnonzero(live & racing)
                if len(keep):
                    # append while there is room, then replace random old samples (the aggregate
                    # keeps following the implant's own, improving driving)
                    free = min(len(keep), cap - n)
                    where = np.r_[np.arange(n, n + free), rng.integers(0, cap, len(keep) - free)].astype(np.int64)
                    w_t = torch.as_tensor(where, device=dev)
                    X[w_t] = torch.as_tensor(features()[keep], device=dev).half()
                    Y[w_t] = torch.as_tensor(lab[keep], device=dev)
                    Ya[w_t] = torch.as_tensor(analog[keep], device=dev)
                    Yw[w_t] = torch.as_tensor(wrong[keep], device=dev)
                    n += free
                    seen += len(keep)
            if record:
                for k in np.flatnonzero(live):
                    masks[k].append(buttons_to_mask(buttons[k]))
                    dn[k].append(fleet.motor.rates[k].astype(np.float16))
            if pilot_noise is not None:
                rates, infos = fleet.pool.step_pilot(pilot_noise)
            else:
                rates, infos = fleet.pool.step(buttons)
            if cnn is not None:
                for k, inf in enumerate(infos):
                    views[k] = views[k][1:] + [inf["view"]]
            for k, inf in enumerate(infos):
                if not live[k]:
                    continue
                progs[k].update(inf["segment"], i)
                if inf["done"] or i == frames - 1:
                    live[k] = False
                    res[k] = {"progress": progs[k].total, "lap": inf["lap"], "frames": i + 1,
                              "finished": bool(inf.get("finished")), "laps5": inf["lap"] >= 5,
                              "rank": inf.get("rank")}
                    if implant_on:   # share of racing frames each channel was written by the implant
                        res[k]["implant_share"] = [round(float(v), 3) for v in share[k] / max(racing_frames[k], 1)]
                    if record:
                        res[k]["masks"], res[k]["rates"] = np.array(masks[k]), np.array(dn[k])
            if not live.any():
                break
        return res

    def add_human(path):
        """The owner's races as lessons: the emulators replay the owner's exact inputs while the
        fly's brain watches (implant and clamp off); each racing frame is a sample labelled with
        the owner's smoothed steering / lean (instruct.intent) and gas / boost buttons."""
        nonlocal n, ema   # not ``seen``: rounds are paced by new DAgger frames only
        from .instruct import intent
        from .motor import BUTTONS
        from .record import mask_to_buttons

        L = np.load(path)
        lessons = [(L[f"l{i}_state"].tobytes(), L[f"l{i}_masks"], L[f"l{i}_racing"]) for i in range(int(L["n"]))]
        added = 0
        for start in range(0, len(lessons), B):
            batch = [lessons[(start + k) % len(lessons)] for k in range(B)]
            fleet.brain.reset()
            fleet.motor.reset()
            clamp.reset(np.arange(B))
            ema = cp.zeros_like(ema)
            rates, infos = fleet.pool.load([b[0] for b in batch])
            smooth = [intent(b[1], BUTTONS, 15.0) for b in batch]
            T = max(len(b[1]) for b in batch)
            iB, iA = BUTTONS.index("B"), BUTTONS.index("A")
            for t in range(T):
                counts = fleet.think(rates)
                if pix:
                    ema = cp.asarray(np.stack([inf["pix"] for inf in infos]).astype(np.float32)) * \
                        np.float32(fleet.window / 1000.0)
                else:
                    ema += a_ema * (counts[:, d_idx].astype(cp.float32) - ema)
                fleet.motor.update(counts, fleet.window)
                keep = [k for k in range(B) if t < len(batch[k][1]) and batch[k][2][t] and (start + k) < len(lessons)]
                if keep:
                    lab = np.array([[1 if smooth[k][t][0] < -0.5 else 2 if smooth[k][t][0] > 0.5 else 0,
                                     1 if smooth[k][t][1] < -0.5 else 2 if smooth[k][t][1] > 0.5 else 0,
                                     int(batch[k][1][t][iB]), int(batch[k][1][t][iA])] for k in keep])
                    analog = np.array([smooth[k][t][:2] for k in keep], np.float32)
                    free = min(len(keep), cap - n)
                    where = np.r_[np.arange(n, n + free), rng.integers(0, cap, len(keep) - free)].astype(np.int64)
                    w_t = torch.as_tensor(where, device=dev)
                    X[w_t] = torch.as_tensor(features()[keep], device=dev).half()
                    Y[w_t] = torch.as_tensor(lab, device=dev)
                    Ya[w_t] = torch.as_tensor(analog, device=dev)
                    Yw[w_t] = -1
                    n += free
                    added += len(keep)
                buttons = [mask_to_buttons(batch[k][1][min(t, len(batch[k][1]) - 1)]) for k in range(B)]
                rates, infos = fleet.pool.step(buttons)
        print(f"human lessons: {added} samples from {len(lessons)} races", flush=True)

    def train(epochs):
        net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for b in range(0, n, 512):
                j = perm[b:b + 512]
                xb, yb = X[j].float(), Y[j]
                outs = net(xb)
                if a.taps:   # steer/lean regressed on the pilot's continuous command; gas, boost classes
                    loss = 4.0 * F.mse_loss(outs[4], Ya[j]) + F.cross_entropy(outs[2], yb[:, 2]) + \
                        F.cross_entropy(outs[3], yb[:, 3], weight=BOOST_W)   # boost labels are rare
                else:
                    loss = sum(F.cross_entropy(outs[k], yb[:, k]) for k in range(4))
                w = Yw[j]
                known = w >= 0
                if known.any():   # the gate: where would the fly's own choice be wrong?
                    loss = loss + F.binary_cross_entropy_with_logits(outs[5][known], w[known].float())
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        return float(loss.detach())

    if a.eval:     # exams only: a trained implant, N solo races from --exam, finishes saved
        r0 = Path(a.resume)
        last = r0 / "implant_last.pt"
        load_implant(net, r0 / a.eval_weights if a.eval_weights else (last if last.exists() else r0 / "implant.pt"))
        net.eval()
        nz = np.load(r0 / "normaliser.npz")
        assert np.array_equal(nz["electrodes"], idx)
        mu, sd = cp.asarray(nz["mu"]), cp.asarray(nz["sd"])
        results = []
        for k in range(0, a.eval, B):
            for x in drive(a.exam_frames, True, False, grid_only=True, record=True):
                results.append(x)
                tag = "finish" if x["finished"] else "5laps" if x.get("laps5") else None
                if tag:
                    save_run(out / f"{tag}_{len(results)}.npz", exam_state, x["masks"], x["rates"], "augmented",
                             progress=x["progress"], laps=x["lap"], frames=x["frames"], rank=x.get("rank"))
            rec = {"attempts": len(results), "finished": sum(r["finished"] for r in results),
                   "progress": [r["progress"] for r in results[-B:]], "ranks": [r.get("rank") for r in results[-B:]],
                   "implant_share": [r.get("implant_share") for r in results[-B:]], "theta": theta.round(3).tolist()}
            print(json.dumps(rec), flush=True)
            with open(out / "eval.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        fleet.close()
        return
    if a.resume:   # continue a run: its implant, normaliser and dataset
        r0 = Path(a.resume)
        last = r0 / "implant_last.pt"
        load_implant(net, last if last.exists() else r0 / "implant.pt")
        net.eval()
        nz = np.load(r0 / "normaliser.npz")
        assert np.array_equal(nz["electrodes"], idx)
        mu, sd = cp.asarray(nz["mu"]), cp.asarray(nz["sd"])
        np.savez(out / "normaliser.npz", mu=nz["mu"], sd=nz["sd"], electrodes=idx)
        if (r0 / "dataset_X.npy").exists() and not a.fresh_data:
            n = load_dataset(r0, X, Y, Ya, dev, Yw=Yw)
        elif (r0 / "dataset.pt").exists() and not a.fresh_data:   # older runs
            ds = torch.load(r0 / "dataset.pt", mmap=True)
            n = min(len(ds["Y"]), cap)
            for b in range(0, n, 20000):
                e = min(n, b + 20000)
                X[b:e], Y[b:e] = ds["X"][b:e].to(dev), ds["Y"][b:e].to(dev)
                if "Ya" in ds:
                    Ya[b:e] = ds["Ya"][b:e].to(dev)
            del ds
        else:   # no saved dataset: a fresh round (driven by the host fly with --host-drives)
            while n < a.round_frames:
                drive(a.episode_frames, not a.host_drives, True)
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
    if a.human:
        add_human(a.human)
    for r in range(a.rounds + 1):
        loss = train(a.epochs)
        exam = []
        for _ in range(a.exam_drives // B):
            exam += drive(a.exam_frames, True, False, grid_only=True, record=True)
        top = max(exam, key=lambda x: (x["progress"], -x["frames"]))
        rec = {"round": r, "dataset": n, "loss": round(loss, 3), "mins": round((time.time() - t) / 60, 1),
               "exam": {**summarize(exam), "each": [x["progress"] for x in exam]},
               "implant_share": np.mean([x["implant_share"] for x in exam], 0).round(3).tolist(),
               "gate_labels": int((Yw[:n, 0] >= 0).sum()), "theta": theta.round(3).tolist()}
        print(json.dumps(rec), flush=True)
        with open(out / "log.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        for k, x in enumerate(exam):
            if x["finished"] or x.get("laps5"):   # "finish" = top 3; "5laps" = full distance, ranked out
                tag = "finish" if x["finished"] else "5laps"
                save_run(out / f"{tag}_r{r}_{k}.npz", exam_state, x["masks"], x["rates"], "augmented",
                         progress=x["progress"], laps=x["lap"], frames=x["frames"], rank=x.get("rank"))
        if top["progress"] > best:
            best = top["progress"]
            save_run(out / "best_drive.npz", exam_state, top["masks"], top["rates"], "augmented",
                     progress=top["progress"], laps=top["lap"], frames=top["frames"])
            torch.save(net.state_dict(), out / "implant.pt")
        torch.save(net.state_dict(), out / "implant_last.pt")
        (out / "gate.json").write_text(json.dumps({"budget": budget, "theta": theta.tolist()}))
        save_dataset(out, X, Y, Ya, n, Yw=Yw)
        if r == a.rounds:
            break
        if a.human and a.human_every and (r + 1) % a.human_every == 0:
            add_human(a.human)   # again, so the reservoir keeps some of the owner's driving
        target = seen + a.round_frames
        while seen < target:
            # --host-drives: new data only while the natural fly drives (implant off). With the
            # implant driving, DNs it clamps feed back into the electrodes, and later rounds
            # learned to echo their own commands (exams fell from 155 to 67 segments).
            # --dart: a share of the data is collected with the pilot driving plus disturbances
            if a.dart > 0 and rng.random() < a.dart:
                drive(a.episode_frames, False, True, pilot_noise=a.dart_noise)
            else:
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
    ap.add_argument("--inputs", choices=["neurons", "pixels"], default="neurons",
                    help="pixels: control network reading the screen instead of the fly's neurons")
    ap.add_argument("--width", type=int, default=256, help="implant hidden units")
    ap.add_argument("--l2-top", type=int, default=0, help="extra electrodes on the most steering-related L2 neurons")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--human", help="lessons file (flyzero lessons) of the owner's races to learn from too")
    ap.add_argument("--human-every", type=int, default=3, help="re-add the human races every N rounds")
    ap.add_argument("--dart", type=float, default=0.0, help="share of data drives where the pilot drives (with noise)")
    ap.add_argument("--dart-noise", type=float, default=0.02, help="per-frame chance of starting a disturbance")
    ap.add_argument("--eval", type=int, default=0, help="with --resume: only run this many exam races")
    ap.add_argument("--eval-weights", help="with --eval: implant file inside the --resume folder")
    ap.add_argument("--fresh-data", action="store_true", help="with --resume: keep the implant, start a new dataset")
    ap.add_argument("--resume", help="an earlier run's folder: continue with its implant and dataset")
    ap.add_argument("--budget", type=float, default=1.0,
                    help="most the implant may write: share of racing frames per control (steer, lean, gas, "
                         "boost); the fly's own DNs decide the rest. 1 = always (the earlier augmented flies)")
    ap.add_argument("--budget-eta", type=float, default=0.002, help="gate threshold adaptation per frame")
    ap.add_argument("--budget-floor", type=float, default=0.5,
                    help="lowest gate threshold: 0.5 = write only where the fly is more likely wrong than right; "
                         "lower lets the implant use its whole budget")
    ap.add_argument("--cnn", help="with --eval: a CNN baseline (cnn_dagger) gives the orders through the "
                                  "clamp and the fly's DNs (a test of the output path)")
    ap.add_argument("--cnn-width", type=int, default=256)
    # the write path (ours): how fast the injected current follows the implant, and how much the
    # button readout smooths the DN rates. Defaults = the earlier augmented flies
    ap.add_argument("--clamp-gain", type=float, default=0.02, help="mV per Hz of rate error per frame")
    ap.add_argument("--clamp-tau", type=float, default=80.0, help="ms, the clamp's rate estimate")
    ap.add_argument("--readout-tau", type=float, default=80.0, help="ms, DN rate smoothing of the button readout")
    ap.add_argument("--out", required=True)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
