"""Emulator pool: one F-Zero emulator per subprocess (stable-retro allows one per process).

Each worker owns a game and a copy of the fly's eye (``MotionEye``, which keeps its own filter
state), so the main process only sends buttons and gets back what the fly sees:

    pool = EmulatorPool(rom, eye, n=8)
    rates, infos = pool.load([state] * 8)          # start positions
    rates, infos = pool.step(list_of_button_dicts)  # one frame for every game
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _pixels(frame, hist):
    """A small picture for a brain-free control network: 28x32 colour + motion (difference of the
    grey image to 4 frames earlier), 3,584 values."""
    small = frame[::8, ::8].astype(np.float32) / 255.0
    grey = small.mean(2)
    hist.append(grey)
    if len(hist) > 5:
        hist.pop(0)
    return np.concatenate([small.ravel(), (grey - hist[0]).ravel()]).astype(np.float16)


def _worker(conn, rom, eye, core, line, pixels=False):
    from .games import FZero
    from .pilot import Pilot

    game = FZero(rom, skip_menu=True, core=core)
    if line is not None:
        from .pilot import PilotParams

        # per-track pilot settings may ride along in the line file (boost_frac, kp, kd, look_base, ...)
        keys = ("boost_frac", "kp", "kd", "look_base", "look_per_speed", "over_speed", "lean_start")
        pilot = Pilot(line["points"], line["speed"],
                      PilotParams(**{k: float(line[k]) for k in keys if k in line}))
    else:
        pilot = None
    flip = False
    frame = None

    def look(fr):
        return eye.see(np.ascontiguousarray(fr[:, ::-1]) if flip else fr)

    hist = []

    def labelled(info):
        """Telemetry plus what the pilot would do here (it never presses anything for the fly)."""
        info = dict(info)
        info["racing"] = FZero.racing(frame)
        if pixels:
            info["pix"] = _pixels(frame, hist)
        if pilot is not None:
            info["pilot"] = pilot.intent(game.ram()).tolist()
        return info

    while True:
        msg = conn.recv()
        cmd = msg[0]
        if cmd == "load":
            _, state, flip = msg
            game.em.set_state(state)
            game.frame_no, game.info, game._last_move, game._empty = 0, {}, 0, 0
            eye.lp_hp = None
            if pilot is not None:
                pilot.reset()
            hist.clear()
            frame = game._press({})
            conn.send((look(frame), labelled(game.info)))
        elif cmd == "step":
            _, buttons, want_frame = msg
            game.collect_audio = want_frame
            frame = game.step(buttons)
            extra = (frame, game.pop_audio()) if want_frame else (None, None)
            conn.send((look(frame), labelled(game.info)) + extra)
        elif cmd == "restore":  # jump to a saved state mid-race, keeping frame/segment bookkeeping
            _, state, info, frame_no, flip = msg
            game.em.set_state(state)
            game.info, game.frame_no = dict(info), frame_no
            game._last_move, game._empty = frame_no, 0
            eye.lp_hp = None
            if pilot is not None:
                pilot.reset()
            hist.clear()
            frame = game._press({})
            conn.send((look(frame), labelled(game.info)))
        elif cmd == "step_pilot":  # DART data collection: the pilot drives, with random disturbances
            _, noise, want_frame = msg
            rng_ = getattr(pilot, "_rng", None) or np.random.default_rng()
            pilot._rng = rng_
            buttons = pilot.buttons(game.ram())
            if getattr(pilot, "_dist", 0) > 0:
                pilot._dist -= 1
                buttons.update(pilot._dist_b)
            elif rng_.random() < noise:          # start a disturbance: a random steer / lean for a bit
                pilot._dist = int(rng_.integers(5, 30))
                d = rng_.choice(["LEFT", "RIGHT"])
                pilot._dist_b = {"LEFT": d == "LEFT", "RIGHT": d == "RIGHT",
                                 "L": d == "LEFT" and rng_.random() < 0.4, "R": d == "RIGHT" and rng_.random() < 0.4}
            frame = game.step(buttons)
            conn.send((look(frame), labelled(game.info), None, None))
        elif cmd == "snapshot":  # (state, info, frame_no) to restore later
            conn.send((bytes(game.em.get_state()), dict(game.info), game.frame_no))
        elif cmd == "pilot_drive":  # the pilot drives from a state; snapshots along the way
            _, state, frames, every = msg
            if pilot is None:
                raise RuntimeError("pilot_drive needs a racing line (EmulatorPool(line=...))")
            game.em.set_state(state)
            game.frame_no, game.info, game._last_move, game._empty = 0, {}, 0, 0
            pilot.reset()
            snaps = []
            for i in range(frames):
                game.step(pilot.buttons(game.ram()))
                if (i + 1) % every == 0:
                    snaps.append((bytes(game.em.get_state()), dict(game.info), game.frame_no))
                if game.info["done"]:
                    break
            conn.send(snaps)
        elif cmd == "save":
            conn.send(bytes(game.em.get_state()))
        elif cmd == "frame":
            conn.send(frame)
        elif cmd == "close":
            conn.close()
            return


class EmulatorPool:
    def __init__(self, rom: str, eye, n: int, core: str | None = None, line: dict | None = None,
                 pixels: bool = False):
        """``line``: racing line (``points``, ``speed``) for the pilot's labels (``info["pilot"]``)."""
        ctx = mp.get_context("spawn")
        self.n = n
        self.pipes, self.procs = [], []
        eye.lp_hp = None
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, rom, eye, core, line, pixels), daemon=True)
            p.start()
            self.pipes.append(a)
            self.procs.append(p)

    def _all(self, msgs, which=None):
        which = list(range(self.n)) if which is None else list(which)
        msgs = list(msgs)
        assert len(msgs) == len(which), "one message per worker"
        for k, m in zip(which, msgs):
            self.pipes[k].send(m)
        return [self.pipes[k].recv() for k in which]

    def load(self, states, flips=None, which=None):
        which = list(range(self.n)) if which is None else list(which)
        flips = flips or [False] * len(which)
        out = self._all([("load", s, f) for s, f in zip(states, flips)], which)
        return np.stack([o[0] for o in out]), [o[1] for o in out]

    def restore(self, items, which, flips=None):
        """``items``: (state, info, frame_no) per worker in ``which``; ``flips``: mirrored view."""
        flips = flips or [False] * len(list(which))
        items = [tuple(it) + (f,) for it, f in zip(items, flips)]
        out = self._all([("restore", *it) for it in items], list(which))
        return np.stack([o[0] for o in out]), [o[1] for o in out]

    def step(self, buttons, frames=False):
        """-> rates, infos (+ frames, audio chunks if ``frames``)."""
        out = self._all([("step", b, frames) for b in buttons])
        rates = np.stack([o[0] for o in out])
        return (rates, [o[1] for o in out]) + (([o[2] for o in out], [o[3] for o in out]) if frames else ())

    def step_pilot(self, noise: float, which=None):
        """DART: the workers' pilots drive (plus random disturbances); for data collection only."""
        which = list(range(self.n)) if which is None else list(which)
        out = self._all([("step_pilot", noise, False)] * len(which), which)
        return np.stack([o[0] for o in out]), [o[1] for o in out]

    def snapshot(self, which):
        return self._all([("snapshot",)] * len(list(which)), which)

    def pilot_drive(self, state, frames, every, worker=0):
        return self._all([("pilot_drive", state, frames, every)], [worker])[0]

    def save(self, which=None):
        return self._all([("save",)] * (self.n if which is None else len(which)), which)

    def close(self):
        for p in self.pipes:
            try:
                p.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for c in self.pipes:
            c.close()
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
