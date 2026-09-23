"""Live 3D feed: the fly on a SNES pad, pressing what its descending neurons press, in the browser.

    python -m flyzero.live --weights runs/.../fly0.npz            # the fly drives, live
    python -m flyzero.live --replay runs/.../race.npz              # a recorded race at true speed
    python -m flyzero.live --pilot                                  # the reference pilot drives

Open http://localhost:8765. The page (``web/index.html``) gets one JSON state per game frame
(buttons, DN rates, lap, time, energy, speed) plus the game image (JPEG) and sound over a
WebSocket on port+1. Live mode runs as fast as the brain allows, capped at real time.

Replays hold only inputs and DN rates (``save_run``), and play back bit-exact from the start
line, because the emulator is deterministic.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import io
import json
import threading
import time
from pathlib import Path

import numpy as np

WEB = Path(__file__).parent / "web"
FPS = 60.0988
READOUT = ["steer_left", "steer_right", "lean_left", "lean_right", "wing_left", "wing_right",
           "accelerate", "brake", "boost"]


def save_run(path, start_state: bytes, masks: np.ndarray, rates: np.ndarray, driver: str, **meta):
    """A race as inputs (T, 12) + DN rates (T, 9); replays exactly from ``start_state``."""
    np.savez_compressed(path, start_state=np.frombuffer(start_state, np.uint8), masks=masks.astype(np.uint8),
                        rates=rates.astype(np.float16), driver=np.array(driver), meta=np.array(json.dumps(meta)))


def _start(game, run):
    """Put a saved race's start in the emulator exactly as the drive began: like
    ``EmulatorPool.load``, one frame with nothing pressed comes before the first input."""
    game.em.set_state(run["start_state"].tobytes())
    game.frame_no, game.info, game._last_move, game._empty = 0, {}, 0, 0
    game._press({})
    game.pop_audio()


class Hub:
    """Frames produced by a sim thread, broadcast to every connected browser."""

    def __init__(self):
        self.clients = set()
        self.loop = None
        self.hello = {"hello": True, "driver": "fly", "audio_rate": 32040}

    def publish(self, state: dict, jpeg: bytes | None, audio: np.ndarray | None):
        if self.loop is None or not self.clients:
            return
        msgs = [json.dumps(state)]
        if jpeg is not None:
            msgs.append(b"\x01" + jpeg)
        if audio is not None and len(audio):
            msgs.append(b"\x02" + np.ascontiguousarray(audio, np.int16).tobytes())
        asyncio.run_coroutine_threadsafe(self._send(msgs), self.loop)

    async def _send(self, msgs):
        for ws in list(self.clients):
            try:
                for m in msgs:
                    await ws.send(m)
            except Exception:
                self.clients.discard(ws)

    async def handler(self, ws):
        self.clients.add(ws)
        await ws.send(json.dumps(self.hello))
        try:
            async for _ in ws:
                pass
        finally:
            self.clients.discard(ws)


def jpeg(frame: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "JPEG", quality=88)
    return buf.getvalue()


def _state(i, buttons, rates, info, finish, sim_fps=None):
    st = {"i": i, "buttons": [b for b, v in buttons.items() if v],
          "rates": {k: round(float(v), 1) for k, v in zip(READOUT, rates)},
          "lap": int(info.get("lap", 0)), "race_frames": i, "energy": float(info.get("energy", 1.0)),
          "speed": int(info.get("speed", 0)), "rank": info.get("rank")}
    if finish:
        st["finish_frames"] = finish
    if sim_fps is not None:
        st["sim_fps"] = sim_fps
    return st


class Pacer:
    """Keep playback at no more than real time."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.n = 0

    def wait(self):
        self.n += 1
        ahead = self.t0 + self.n / FPS - time.perf_counter()
        if ahead > 0:
            time.sleep(ahead)
        elif ahead < -0.5:  # fell behind (slow brain): don't try to catch up
            self.t0, self.n = time.perf_counter(), 0


def run_replay(hub: Hub, rom: str, path: str, loop: bool = True):
    from .games import FZero
    from .record import mask_to_buttons

    d = np.load(path)
    hub.hello["driver"] = str(d["driver"])
    game = FZero(rom, skip_menu=True)
    hub.hello["audio_rate"] = game.audio_rate()
    game.collect_audio = True
    while True:
        _start(game, d)
        pace, finish = Pacer(), None
        for i, (m, r) in enumerate(zip(d["masks"], d["rates"].astype(np.float32))):
            b = mask_to_buttons(m)
            frame = game.step(b)
            if finish is None and game.info["lap"] >= 5:
                finish = i + 1
            hub.publish(_state(i + 1, b, r, game.info, finish), jpeg(frame), game.pop_audio())
            pace.wait()
        for _ in range(int(FPS * 4)):   # hold the last frame a moment, then go again
            pace.wait()
        if not loop:
            return


def export(rom: str, path: str, out: str):
    """A saved race -> ``game.mp4`` (with sound) + ``trace.json`` for the published page."""
    import subprocess
    import wave

    import imageio.v2 as imageio
    import imageio_ffmpeg

    from .games import FZero
    from .record import mask_to_buttons

    d = np.load(path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    game = FZero(rom, skip_menu=True)
    _start(game, d)
    game.collect_audio = True
    frames, finish = [], None
    silent = out / "silent.mp4"
    w = imageio.get_writer(silent, fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                           ffmpeg_params=["-vf", "scale=512:448:flags=neighbor", "-pix_fmt", "yuv420p"])
    pcm = []
    for i, (m, r) in enumerate(zip(d["masks"], d["rates"].astype(np.float32))):
        b = mask_to_buttons(m)
        w.append_data(game.step(b))
        pcm.append(game.pop_audio())
        if finish is None and game.info["lap"] >= 5:
            finish = i + 1
        frames.append(_state(i + 1, b, r, game.info, finish))
    w.close()
    with wave.open(str(out / "audio.wav"), "wb") as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(int(round(game.audio_rate())))
        f.writeframes(np.concatenate(pcm).astype(np.int16).tobytes())
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(out / "audio.wav"),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", "-shortest", str(out / "game.mp4")], check=True)
    silent.unlink()
    (out / "audio.wav").unlink()
    (out / "trace.json").write_text(json.dumps({"fps": FPS, "driver": str(d["driver"]), "frames": frames},
                                               separators=(",", ":")))
    print(f"wrote {out}/game.mp4 and trace.json ({len(frames)} frames, finish {finish})")


def run_pilot(hub: Hub, rom: str, state: str, line: str):
    from .games import FZero
    from .pilot import Pilot

    hub.hello["driver"] = "pilot"
    L = np.load(line)
    pilot = Pilot(L["points"], L["speed"])
    game = FZero(rom, state=state)
    hub.hello["audio_rate"] = game.audio_rate()
    game.collect_audio = True
    while True:
        game.reset()
        pilot.reset()
        pace, finish = Pacer(), None
        for i in range(12000):
            b = pilot.buttons(game.ram())
            frame = game.step(b)
            if finish is None and game.info["lap"] >= 5:
                finish = i + 1
            hub.publish(_state(i + 1, b, np.zeros(9), game.info, finish), jpeg(frame), game.pop_audio())
            pace.wait()
            if game.info["done"]:
                break


def run_live(hub: Hub, rom: str, state: str, weights: str | None, deep: bool, driver: str, seed: int,
             out: str | None):
    from .batch import BatchInstruct
    from .fleet import Fleet
    from .record import buttons_to_mask

    hub.hello["driver"] = driver
    w = np.load(weights) if weights else {}
    fleet = Fleet(rom, 1, seed=seed, deep=deep, taps=bool(w.get("taps", False)),
                  steer_span=float(w.get("steer_span", 150.0)), lean_span=float(w.get("lean_span", 70.0)))
    if weights:
        BatchInstruct(fleet.brain, fleet.conn).load(np.load(weights))
    start = Path(state).read_bytes()
    hub.hello["audio_rate"] = 32040.0
    run = 0
    while True:
        fleet.brain.reset()
        fleet.motor.reset()
        rates, infos = fleet.pool.load([start])
        pace, finish = Pacer(), None
        masks, dn = [], []
        t_last, n_last = time.perf_counter(), 0
        sim_fps = 0.0
        for i in range(12000):
            counts = fleet.think(rates)
            buttons = fleet.motor.update(counts, fleet.window)
            rates, infos, frames, audio = fleet.pool.step(buttons, frames=True)
            info = infos[0]
            masks.append(buttons_to_mask(buttons[0]))
            dn.append(fleet.motor.rates[0].copy())
            if finish is None and info["lap"] >= 5:
                finish = i + 1
            if i - n_last >= 30:
                now = time.perf_counter()
                sim_fps, t_last, n_last = (i - n_last) / (now - t_last), now, i
            hub.publish(_state(i + 1, buttons[0], fleet.motor.rates[0], info, finish, sim_fps),
                        jpeg(frames[0]), audio[0])
            pace.wait()
            if info["done"]:
                break
        if out:
            run += 1
            save_run(Path(out) / f"live_{driver}_{run}.npz", start, np.array(masks), np.array(dn), driver,
                     laps=int(info["lap"]), frames=i + 1, weights=weights)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom", default="F-Zero (USA).sfc")
    ap.add_argument("--state", default="start.state")
    ap.add_argument("--port", type=int, default=8765)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--replay", help="a race saved by save_run")
    g.add_argument("--pilot", action="store_true", help="the reference pilot drives")
    ap.add_argument("--weights", help="the fly's plastic synapses (.npz); none = untrained FlyWire fly")
    ap.add_argument("--deep", action="store_true", help="weights include the deeper layer")
    ap.add_argument("--driver", default="fly", choices=["fly", "augmented", "pilot", "cnn"])
    ap.add_argument("--line", default="runs/pilot/mute_city_line.npz")
    ap.add_argument("--save", help="folder to save each live race (inputs + DN rates) for replay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", metavar="DIR", help="with --replay: write game.mp4 + trace.json and exit")
    a = ap.parse_args(argv)
    if a.export:
        return export(a.rom, a.replay, a.export)

    import websockets

    hub = Hub()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    if a.replay:
        target = functools.partial(run_replay, hub, a.rom, a.replay)
    elif a.pilot:
        target = functools.partial(run_pilot, hub, a.rom, a.state, a.line)
    else:
        target = functools.partial(run_live, hub, a.rom, a.state, a.weights, a.deep, a.driver, a.seed, a.save)
    threading.Thread(target=target, daemon=True).start()
    print(f"open http://localhost:{a.port}", flush=True)

    async def serve():
        hub.loop = asyncio.get_running_loop()
        async with websockets.serve(hub.handler, "127.0.0.1", a.port + 1, max_size=None):
            await asyncio.Future()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
