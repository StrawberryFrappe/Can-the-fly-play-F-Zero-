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


def _worker(conn, rom, eye, core):
    from .games import FZero

    game = FZero(rom, skip_menu=True, core=core)
    flip = False
    frame = None

    def look(fr):
        return eye.see(np.ascontiguousarray(fr[:, ::-1]) if flip else fr)

    while True:
        msg = conn.recv()
        cmd = msg[0]
        if cmd == "load":
            _, state, flip = msg
            game.em.set_state(state)
            game.frame_no, game.info, game._last_move, game._empty = 0, {}, 0, 0
            eye.lp_hp = None
            frame = game._press({})
            conn.send((look(frame), dict(game.info)))
        elif cmd == "step":
            _, buttons, want_frame = msg
            frame = game.step(buttons)
            conn.send((look(frame), game.info, frame if want_frame else None))
        elif cmd == "restore":  # jump to a saved state mid-race, keeping frame/segment bookkeeping
            _, state, info, frame_no = msg
            game.em.set_state(state)
            game.info, game.frame_no = dict(info), frame_no
            game._last_move, game._empty = frame_no, 0
            eye.lp_hp = None
            frame = game._press({})
            conn.send((look(frame), dict(game.info)))
        elif cmd == "save":
            conn.send(bytes(game.em.get_state()))
        elif cmd == "frame":
            conn.send(frame)
        elif cmd == "close":
            conn.close()
            return


class EmulatorPool:
    def __init__(self, rom: str, eye, n: int, core: str | None = None):
        ctx = mp.get_context("spawn")
        self.n = n
        self.pipes, self.procs = [], []
        eye.lp_hp = None
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, rom, eye, core), daemon=True)
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

    def restore(self, items, which):
        """``items``: (state, info, frame_no) per worker in ``which``."""
        out = self._all([("restore", *it) for it in items], list(which))
        return np.stack([o[0] for o in out]), [o[1] for o in out]

    def step(self, buttons, frames=False):
        out = self._all([("step", b, frames) for b in buttons])
        rates = np.stack([o[0] for o in out])
        return (rates, [o[1] for o in out]) + (([o[2] for o in out],) if frames else ())

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
