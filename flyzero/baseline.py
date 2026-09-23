"""The comparison: a small standard neural network taught from the same human races.

Behaviour cloning with a plain convolutional network. It sees the last two frames, downsampled
4x (56x64, colour), and predicts the same buttons the fly controls:

    steer (none / left / right), lean (none / L / R), gas, brake

It gets exactly the same lessons (``flyzero lessons``) and the same solo exam on Mute City as
the fly (``flyzero teach``), so the two can be compared directly. It is deliberately modest
(about 100k parameters), the kind of network people train in an afternoon, not a tuned agent.

    flyzero baseline --rom fzero.sfc --lessons lessons.npz --exam start.state
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .motor import BUTTONS
from .record import mask_to_buttons
from .tune import Progress

IDX = {b: i for i, b in enumerate(BUTTONS)}


def observe(frame: np.ndarray) -> np.ndarray:
    return frame[::4, ::4]  # 56 x 64 x 3


def labels(mask: np.ndarray) -> tuple[int, int, int, int]:
    l, r = mask[IDX["LEFT"]], mask[IDX["RIGHT"]]
    L, R = mask[IDX["L"]], mask[IDX["R"]]
    steer = 1 if l and not r else 2 if r and not l else 0
    lean = 1 if L and not R else 2 if R and not L else 0
    return steer, lean, int(mask[IDX["B"]]), int(mask[IDX["Y"]])


def build_net():
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(6, 16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(), nn.Linear(32 * 7 * 8, 128), nn.ReLU())
            self.steer, self.lean = nn.Linear(128, 3), nn.Linear(128, 3)
            self.gas, self.brake = nn.Linear(128, 2), nn.Linear(128, 2)

        def forward(self, x):
            h = self.body(x)
            return self.steer(h), self.lean(h), self.gas(h), self.brake(h)

    return Net()


def to_tensor(obs_now, obs_prev):
    import torch

    x = np.concatenate([obs_now, obs_prev], axis=-1).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2) if x.ndim == 4 else \
        torch.from_numpy(x).permute(2, 0, 1)[None]


def run(rom: str, lessons_path: str, exam_state: str, out: str, epochs: int = 6,
        exam_frames: int = 3600, seed: int = 0, threads: int = 2, mirror: bool = True):
    import torch
    import torch.nn.functional as F

    from .games import FZero

    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    L = np.load(lessons_path)
    game = FZero(rom, skip_menu=True)

    # 1. dataset: replay every lesson, keep downsampled frames + the teacher's buttons
    obs, y = [], []
    for i in range(int(L["n"])):
        game.em.set_state(L[f"l{i}_state"].tobytes())
        frame = game._press({})
        prev = observe(frame)
        masks = L[f"l{i}_masks"]
        racing = L[f"l{i}_racing"] if f"l{i}_racing" in L else np.ones(len(masks), bool)
        for m, is_racing in zip(masks, racing):
            cur = observe(frame)
            if is_racing:  # menus / results / retry screens are not driving lessons
                obs.append(np.concatenate([cur, prev], -1))
                y.append(labels(m))
            prev = cur
            frame = game._press(mask_to_buttons(m))
    X = np.stack(obs)                     # N x 56 x 64 x 6 uint8
    del obs
    Y = np.array(y, np.int64)
    n = len(X)
    print(f"dataset: {n} frames", flush=True)

    # 2. train (last 10% held out, in time order)
    split = int(n * 0.9)
    net = build_net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    # class weights: rare actions (turns, leans) matter
    wts = [torch.tensor(1.0 / np.maximum(np.bincount(Y[:split, k], minlength=c), 1) ** 0.5, dtype=torch.float32,
                        device=dev) for k, c in enumerate((3, 3, 2, 2))]
    Xt = torch.from_numpy(X).permute(0, 3, 1, 2)
    Yt = torch.from_numpy(Y)
    for ep in range(epochs):
        t = time.time()
        perm = torch.randperm(split)
        net.train()
        for b in range(0, split, 256):
            idx = perm[b:b + 256]
            xb, yb = Xt[idx].to(dev).float() / 255.0, Yt[idx].to(dev)
            if mirror:  # mirror world for half the batch: flip the view, swap left/right
                flip = torch.rand(len(idx), device=dev) < 0.5
                xb[flip] = xb[flip].flip(-1)
                for k in (0, 1):
                    col = yb[flip, k]
                    yb[flip, k] = torch.where(col == 1, 2, torch.where(col == 2, 1, col))
            outs = net(xb)
            loss = sum(F.cross_entropy(o, yb[:, k], weight=wts[k]) for k, o in enumerate(outs))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            accs = []
            for k in range(4):
                preds = []
                for b in range(split, n, 1024):
                    preds.append(net(Xt[b:b + 1024].to(dev).float() / 255.0)[k].argmax(1).cpu())
                accs.append(float((torch.cat(preds) == Yt[split:, k]).float().mean()))
        print(json.dumps({"epoch": ep + 1, "loss": round(float(loss), 3),
                          "heldout_acc": dict(zip(("steer", "lean", "gas", "brake"), [round(a, 3) for a in accs])),
                          "secs": round(time.time() - t)}), flush=True)
    torch.save(net.state_dict(), Path(out) / "baseline_cnn.pt")

    # 3. solo exam, same as the fly's
    game.em.set_state(Path(exam_state).read_bytes())
    game.frame_no, game.info, game._last_move, game._empty = 0, {}, 0, 0
    frame = game._press({})
    prev = observe(frame)
    prog = Progress()
    with torch.no_grad():
        for i in range(exam_frames):
            cur = observe(frame)
            s, le, g, br = (o.argmax(1).item() for o in net(to_tensor(cur, prev).to(dev)))
            prev = cur
            buttons = {"LEFT": s == 1, "RIGHT": s == 2, "L": le == 1, "R": le == 2, "B": g == 1, "Y": br == 1}
            frame = game.step(buttons)
            prog.update(game.info["segment"], i)
            if game.info["done"]:
                break
    res = {"progress": prog.total, "lap": game.info["lap"], "frames": i + 1, "energy": game.info["energy"]}
    print(json.dumps({"exam": res}), flush=True)
    (Path(out) / "baseline_exam.json").write_text(json.dumps(res))
    return res
