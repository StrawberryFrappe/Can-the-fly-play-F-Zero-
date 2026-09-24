"""flyzero command line: download the connectome, inspect it, and let the fly drive."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import connectome as cx
from .brain import Brain, LIFParams
from .eyes import CompoundEye, EyeParams
from .motor import MotorParams, MotorReadout
from .vision import MotionEye, MotionParams


def parse_drive(specs: list[str], conn) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``TYPE[:side]=HZ`` -> neuron indices and Poisson rates ("optogenetic" drive)."""
    idx, rate, desc = [], [], []
    for spec in specs:
        name, hz = spec.split("=")
        cell_type, _, side = name.partition(":")
        found = conn.find(cell_type, side or None)
        if len(found) == 0:
            print(f"warning: --drive {spec}: no {name} neurons in {conn.name}", file=sys.stderr)
            continue
        idx.append(found)
        rate.append(np.full(len(found), float(hz), np.float32))
        desc.append(f"{name}@{float(hz):g}Hz ({len(found)})")
    if not idx:
        return np.zeros(0, np.int64), np.zeros(0, np.float32), []
    return np.concatenate(idx), np.concatenate(rate), desc


def make_game(args):
    from . import games

    if args.game == "mock":
        return games.MockRacer(seed=args.seed)
    if not args.rom:
        sys.exit("--game fzero needs --rom path/to/F-Zero.sfc (bring your own dump)")
    return games.FZero(args.rom, state=args.state, skip_menu=args.skip_menu)


def cmd_play(args):
    t0 = time.time()
    print(f"loading {args.connectome} connectome ...")
    conn = cx.load(args.connectome, args.data_dir, seed=args.seed)
    print(conn.summary())
    print(f"  ({time.time() - t0:.1f}s)")

    game = make_game(args)
    brain = Brain(conn.weights, LIFParams(dt=args.dt), seed=args.seed)
    motor = MotorReadout(conn, MotorParams())
    print("motor readout:\n" + motor.describe())
    drive_idx, drive_rate, drive_desc = parse_drive(args.drive, conn)
    if drive_desc:
        print("tonic drive: " + ", ".join(drive_desc))

    writer = viewer = dash = None
    if args.video or args.show:
        from .viz import Dashboard
    if args.video:
        import imageio.v2 as imageio

        writer = imageio.get_writer(args.video, fps=round(game.fps), codec="libx264",
                                    quality=7, macro_block_size=1)
    if args.show:
        from stable_retro.rendering import SimpleImageViewer

        viewer = SimpleImageViewer()

    menu_frames = []
    frame = game.reset(on_frame=menu_frames.append if writer else None)
    if args.save_state and hasattr(game, "save_state"):
        game.save_state(args.save_state)
        print(f"saved start-line state to {args.save_state}")

    if args.vision == "motion" and len(conn.find("T4a")) == 0:
        print("no T4/T5 neurons in this connectome, falling back to --vision retina")
        args.vision = "retina"
    if args.vision == "motion":
        eye = MotionEye(conn, frame.shape[:2], MotionParams(gain=args.motion_gain))
    else:
        eye = CompoundEye(conn, frame.shape[:2], EyeParams())
    print("eyes: " + eye.describe())
    if args.video or args.show:
        name = conn.name + (" | drive " + ", ".join(drive_desc) if drive_desc else "")
        dash = Dashboard(eye, motor, name)
    if writer:
        for f in menu_frames:  # the menu macro, so the video starts at power-on
            writer.append_data(dash.render(f, 0.0, 0, 0, {"phase": "menus (scripted)"}))

    trace = [] if args.trace else None
    window = 1000.0 / game.fps
    wall = time.time()
    try:
        for i in range(args.frames):
            rates = eye.see(frame)
            brain.set_input(np.r_[eye.idx, drive_idx], np.r_[rates, drive_rate])
            counts = brain.run(window)
            buttons = motor.update(counts, window)
            frame = game.step(buttons)
            if trace is not None:
                trace.append({"buttons": [k for k, v in buttons.items() if v],
                              "rates": {k: round(float(v), 2) for k, v in motor.rates.items()}})

            if dash:
                out = dash.render(frame, brain.t * brain.p.dt, int(counts.sum()),
                                  int((brain.spike_count > 0).sum()), game.info)
                if writer:
                    writer.append_data(out)
                if viewer:
                    viewer.imshow(out)
            if (i + 1) % args.log_every == 0 or i == args.frames - 1:
                pressed = "".join(b if on else "." for b, on in
                                  zip("<>LRBYA", [buttons[k] for k in
                                                  ("LEFT", "RIGHT", "L", "R", "B", "Y", "A")]))
                r = motor.rates
                info = " ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in game.info.items())
                speed = (i + 1) / (time.time() - wall)
                print(f"[{i + 1:5d}] {pressed} steerL={r['steer_left']:5.1f} "
                      f"steerR={r['steer_right']:5.1f} gas={r['accelerate']:5.1f} "
                      f"brake={r['brake']:5.1f} spikes={int(counts.sum()):5d} | {info} "
                      f"| {speed:.1f} fps")
            if game.info.get("done"):
                print(f"out of energy after {i + 1} frames")
                break
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if writer:
            writer.close()
            print(f"wrote {args.video}")
        if trace is not None:
            import json

            Path(args.trace).write_text(json.dumps({"fps": game.fps, "frames": trace}))
            print(f"wrote {args.trace}")


def cmd_tune(args):
    from .tune import search

    best = search(args.rom, args.state, args.frames, args.generations, args.popsize, args.workers,
                  Path(args.out), flip=args.flip, seed=args.seed)
    print("best:", best)


def cmd_learn(args):
    from .learning import LearnParams
    from .train import learn

    print(learn(args.rom, args.state, args.episodes, args.frames, args.workers, args.out,
                LearnParams(eta=args.eta)))


def cmd_download(args):
    print(f"downloading FlyWire v783 connectome to {args.data_dir}")
    cx.download(args.data_dir, force=args.force)


def cmd_info(args):
    conn = cx.load(args.connectome, args.data_dir)
    print(conn.summary())
    print(f"photoreceptors: {len(conn.photoreceptors())}")
    print("motor readout:\n" + MotorReadout(conn).describe())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flyzero", description="Let a simulated fly brain play F-Zero.")
    ap.add_argument("--data-dir", type=Path, default=cx.DEFAULT_DATA_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch the FlyWire connectome + annotations")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_download)

    i = sub.add_parser("info", help="summarize the connectome and the neurons we wire up")
    i.add_argument("--connectome", choices=["flywire", "synthetic"], default="flywire")
    i.set_defaults(func=cmd_info)

    p = sub.add_parser("play", help="let the fly drive")
    p.add_argument("--game", choices=["fzero", "mock"], default="fzero")
    p.add_argument("--rom", help="F-Zero (SNES) ROM, .sfc/.smc")
    p.add_argument("--state", help="emulator save state to start from (skips menus)")
    p.add_argument("--save-state", help="save the state after the menu macro here")
    p.add_argument("--skip-menu", action="store_true", help="don't run the menu macro")
    p.add_argument("--connectome", choices=["flywire", "synthetic"], default="flywire")
    p.add_argument("--vision", choices=["motion", "retina"], default="motion",
                   help="motion: drive T4/T5 motion detectors (default); "
                        "retina: drive photoreceptors (signal dies in the optic lobe)")
    p.add_argument("--motion-gain", type=float, default=MotionParams.gain)
    p.add_argument("--frames", type=int, default=60 * 30, help="game frames to play (60 = 1 s)")
    p.add_argument("--dt", type=float, default=0.1, help="brain time step in ms")
    p.add_argument("--drive", action="append", default=None, metavar="TYPE[:side]=HZ",
                   help="tonic Poisson drive to a cell type, like optogenetic activation "
                        "(default: DNp09=60, a fly that wants to walk). Pass --drive none to disable.")
    p.add_argument("--video", help="write a dashboard video (mp4)")
    p.add_argument("--trace", help="write per-frame buttons and DN rates (json)")
    p.add_argument("--show", action="store_true", help="live window (needs a display)")
    p.add_argument("--log-every", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_play)

    t = sub.add_parser("tune", help="calibrate the interface (brain stays fixed) for race progress")
    t.add_argument("--rom", required=True)
    t.add_argument("--state", required=True, help="start-line save state")
    t.add_argument("--frames", type=int, default=2400)
    t.add_argument("--generations", type=int, default=20)
    t.add_argument("--popsize", type=int, default=8)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--flip", action="store_true", help="swap left/right at the motor side")
    t.add_argument("--out", default="tune.json")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_tune)

    le = sub.add_parser("learn", help="reward-train the fly's motor-output synapses")
    le.add_argument("--rom", required=True)
    le.add_argument("--state", required=True)
    le.add_argument("--episodes", type=int, default=40)
    le.add_argument("--frames", type=int, default=3600)
    le.add_argument("--workers", type=int, default=4)
    le.add_argument("--eta", type=float, default=2e-4)
    le.add_argument("--out", default="learn")
    le.set_defaults(func=cmd_learn)

    r = sub.add_parser("record", help="play F-Zero yourself (keyboard or Xbox controller); "
                                       "your inputs become the fly's lessons")
    r.add_argument("--rom")
    r.add_argument("--out", default="my_races.npz")
    r.add_argument("--scale", type=int, default=3)
    r.add_argument("--map", help='controller button numbers, e.g. "A=0,B=1,X=2,LB=4,RB=5,START=7,BACK=6"')
    r.add_argument("--controller-test", action="store_true", help="show what each controller button reports")
    r.add_argument("--core", help="snes9x libretro core (.dll on Windows) instead of stable-retro")
    r.add_argument("--no-audio", action="store_true", help="don't play the game sound")
    r.add_argument("--mute-city", action="store_true",
                   help="back-to-back Mute City I: after each finish, straight back to the grid")
    r.add_argument("--league", choices=["knight", "queen", "king"], default=None,
                   help="which Grand Prix the menus pick (default knight)")
    r.add_argument("--class", dest="klass", choices=["beginner", "standard", "expert", "master"], default="beginner",
                   help="difficulty class (default beginner)")
    r.add_argument("--queen-league", action="store_true", help="same as --league queen")
    r.add_argument("--king-league", action="store_true", help="same as --league king")
    r.add_argument("--first-race", action="store_true",
                   help="back-to-back runs of the league's first race (knight: Mute City I, queen: "
                        "Mute City II, king: Mute City III)")
    r.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)

    def _record(a):
        from . import record
        if a.controller_test:
            return record.controller_test()
        if not a.rom:
            sys.exit("--rom is required")
        record.play(a.rom, a.out, a.scale, a.max_frames, a.map, a.core, not a.no_audio,
                    (a.league or ("queen" if a.queen_league else "king" if a.king_league else "knight"))
                    + ("" if a.klass == "beginner" else "/" + a.klass),
                    a.mute_city or a.first_race)
    r.set_defaults(func=_record)

    rp = sub.add_parser("replay", help="check that a recording replays exactly here")
    rp.add_argument("--rom", required=True)
    rp.add_argument("recording")
    rp.set_defaults(func=lambda a: [print({k: v for k, v in r.items() if k != "start_state"})
                                    for r in __import__("flyzero.record", fromlist=["replay_all"])
                                    .replay_all(a.rom, a.recording)])

    ls = sub.add_parser("lessons", help="turn recordings into training lessons (verifies each race)")
    ls.add_argument("--rom", required=True)
    ls.add_argument("recordings", nargs="+")
    ls.add_argument("--out", default="lessons.npz")
    ls.set_defaults(func=lambda a: __import__("flyzero.teach", fromlist=["build_lessons"]).build_lessons(
        a.rom, a.recordings, a.out))

    te = sub.add_parser("teach", help="teach the fly from recorded races, exam it solo each epoch")
    te.add_argument("--rom", required=True)
    te.add_argument("--lessons", required=True)
    te.add_argument("--exam", required=True, help="start-line save state for the solo exam")
    te.add_argument("--epochs", type=int, default=1, help="passes over all lessons (0 = practice only)")
    te.add_argument("--exam-frames", type=int, default=3600)
    te.add_argument("--out", default="taught")
    te.add_argument("--etas", default="3e-4,1e-3,3e-3,1e-2",
                    help="lesson learning rates, one fly (= one CPU process) each")
    te.add_argument("--practice", type=int, default=40, help="solo reward-learning drives after the lessons")
    te.add_argument("--practice-frames", type=int, default=2400)
    te.add_argument("--reward-eta", type=float, default=5e-4)
    te.add_argument("--init", help="start from a taught fly's weights (.npz), e.g. to resume")
    te.add_argument("--smooth", type=float, default=15.0, help="teach intent smoothed over N frames (0 = taps)")
    te.add_argument("--no-mirror", action="store_true")
    te.add_argument("--bias-mv", type=float, default=7.8, help="steady depolarisation of the motor neurons")
    te.set_defaults(func=lambda a: __import__("flyzero.teach", fromlist=["teach"]).teach(
        a.rom, a.lessons, a.exam, a.epochs, a.exam_frames, a.out,
        etas=tuple(float(x) for x in a.etas.split(",")), bias_mv=a.bias_mv, mirror=not a.no_mirror,
        smooth=a.smooth, practice=a.practice, practice_frames=a.practice_frames,
        reward_eta=a.reward_eta, init=a.init))

    bl = sub.add_parser("baseline", help="the comparison: a small CNN taught from the same races")
    bl.add_argument("--rom", required=True)
    bl.add_argument("--lessons", required=True)
    bl.add_argument("--exam", required=True)
    bl.add_argument("--epochs", type=int, default=6)
    bl.add_argument("--exam-frames", type=int, default=3600)
    bl.add_argument("--out", default="baseline")
    bl.set_defaults(func=lambda a: __import__("flyzero.baseline", fromlist=["run"]).run(
        a.rom, a.lessons, a.exam, a.out, a.epochs, a.exam_frames))

    ss = sub.add_parser("start-state", help="save the Mute City I start line (run the menus once)")
    ss.add_argument("--rom", required=True)
    ss.add_argument("--out", default="start.state")
    ss.add_argument("--league", default="knight", choices=["knight", "queen", "king", "practice"],
                    help="practice: Practice mode, Mute City I, one rival, no rank rule")
    ss.add_argument("--core", help="snes9x libretro core (.dll on Windows) instead of stable-retro")

    def _start_state(a):
        from .games import FZero
        g = FZero(a.rom, core=a.core, league=a.league)
        g.reset()
        g.save_state(a.out)
        print(f"saved {a.out} ({g.backend}); the exam starts here")
    ss.set_defaults(func=_start_state)

    args = ap.parse_args(argv)
    if getattr(args, "drive", None) is None and args.cmd == "play":
        args.drive = ["DNp09=60"]
    if args.cmd == "play":
        args.drive = [d for d in args.drive if d.lower() != "none"]
    args.func(args)


if __name__ == "__main__":
    main()
