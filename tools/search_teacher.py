"""Look-ahead instructor: tries each move from a save state, keeps the best. Records a full race."""
import sys, time, json, numpy as np
from flyzero.games import FZero
from road_teacher import Teacher
BASE = Teacher(k_far=1.0, thr=0.04)
chunk, horizon, steer_pen = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); tag=sys.argv[4]
OPTS = [{"B": True}, {"B": True, "LEFT": True}, {"B": True, "RIGHT": True},
        {"B": True, "LEFT": True, "L": True}, {"B": True, "RIGHT": True, "R": True}]
import os
g = FZero(os.environ.get('FLYZERO_ROM', 'fzero.sfc'), state='start.state'); g.reset()
def score_after(opt):
    seg0, lap0 = g.info.get('segment', 58), g.info.get('lap', 0)
    e0 = g.info.get('energy', 1.0)
    prog = 0; last = seg0; fr = g.em.get_screen()
    for k in range(horizon):
        if k < chunk: fr = g.step(opt)
        else: fr = g.step(BASE(fr)[0])
        s = g.info['segment']; prog += (s - last + 29) % 59 - 29; last = s
    return prog * 100 + (g.info['energy'] - e0) * 400 + g.info['speed'] / 50.0
acts = []; t0 = time.time(); prev = 0
while True:
    s0 = g.em.get_state(); info0 = dict(g.info); f0 = g.frame_no
    scores = []
    for i, o in enumerate(OPTS):
        g.em.set_state(s0); g.info = dict(info0); g.frame_no = f0
        scores.append(score_after(o) - (steer_pen if i != 0 else 0) - (steer_pen / 2 if i != prev else 0))
    best = int(np.argmax(scores)); prev = best
    g.em.set_state(s0); g.info = dict(info0); g.frame_no = f0
    for _ in range(chunk):
        g.step(OPTS[best]); acts.append(best)
    if len(acts) % 1200 < chunk:
        print(tag, len(acts), g.info, round(time.time() - t0), flush=True)
    if g.info['done'] or len(acts) > 12000:
        break
np.save(f'teacher_acts_{tag}.npy', np.array(acts))
print(tag, 'END', len(acts), g.info, round(time.time() - t0), flush=True)
