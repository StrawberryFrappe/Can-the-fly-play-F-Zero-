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

    args = ap.parse_args(argv)
    if getattr(args, "drive", None) is None and args.cmd == "play":
        args.drive = ["DNp09=60"]
    if args.cmd == "play":
        args.drive = [d for d in args.drive if d.lower() != "none"]
    args.func(args)


if __name__ == "__main__":
    main()
