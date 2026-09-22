"""A minimal libretro frontend in pure Python (ctypes), so F-Zero runs anywhere a snes9x core does.

stable-retro (the default backend) has no Windows build. This loads a snes9x libretro core
directly, the same kind of core stable-retro wraps and RetroArch uses:

* Windows: ``snes9x_libretro.dll`` from RetroArch's ``cores`` folder, or from
  https://buildbot.libretro.com/nightly/windows/x86_64/latest/snes9x_libretro.dll.zip
* Linux/macOS: ``snes9x_libretro.so`` / ``.dylib`` (stable-retro ships one in its ``cores`` dir)

It implements only what F-Zero needs: video, one joypad, save states and work-RAM access.
Audio is discarded.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

# libretro.h constants
ENV_GET_CAN_DUPE = 3
ENV_GET_SYSTEM_DIRECTORY = 9
ENV_SET_PIXEL_FORMAT = 10
ENV_GET_VARIABLE = 15
ENV_GET_VARIABLE_UPDATE = 17
ENV_GET_SAVE_DIRECTORY = 31
PIXEL_0RGB1555, PIXEL_XRGB8888, PIXEL_RGB565 = 0, 1, 2
DEVICE_JOYPAD = 1
MEMORY_SYSTEM_RAM = 2

ENV_CB = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
VIDEO_CB = C.CFUNCTYPE(None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t)
AUDIO_CB = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
AUDIO_BATCH_CB = C.CFUNCTYPE(C.c_size_t, C.c_void_p, C.c_size_t)
POLL_CB = C.CFUNCTYPE(None)
STATE_CB = C.CFUNCTYPE(C.c_int16, C.c_uint, C.c_uint, C.c_uint, C.c_uint)


class GameInfo(C.Structure):
    _fields_ = [("path", C.c_char_p), ("data", C.c_void_p), ("size", C.c_size_t), ("meta", C.c_char_p)]


class Libretro:
    """Same surface as stable_retro.RetroEmulator for what flyzero uses."""

    _loaded = False

    def __init__(self, core: str | Path, rom: str | Path):
        if Libretro._loaded:
            raise RuntimeError("one libretro core per process")
        self.lib = C.CDLL(str(core))
        L = self.lib
        L.retro_serialize_size.restype = C.c_size_t
        L.retro_serialize.argtypes = [C.c_void_p, C.c_size_t]
        L.retro_serialize.restype = C.c_bool
        L.retro_unserialize.argtypes = [C.c_void_p, C.c_size_t]
        L.retro_unserialize.restype = C.c_bool
        L.retro_get_memory_data.restype = C.c_void_p
        L.retro_get_memory_data.argtypes = [C.c_uint]
        L.retro_get_memory_size.restype = C.c_size_t
        L.retro_get_memory_size.argtypes = [C.c_uint]
        L.retro_load_game.argtypes = [C.POINTER(GameInfo)]
        L.retro_load_game.restype = C.c_bool

        self.pixel_format = PIXEL_0RGB1555
        self.mask = np.zeros(12, np.uint8)
        self.screen = np.zeros((224, 256, 3), np.uint8)
        self._sysdir = C.c_char_p(str(Path(rom).resolve().parent).encode())
        # keep references to the callbacks, or they get garbage-collected under the core
        self._cbs = [ENV_CB(self._env), VIDEO_CB(self._video), AUDIO_CB(lambda l, r: None),
                     AUDIO_BATCH_CB(lambda d, n: n), POLL_CB(lambda: None), STATE_CB(self._state)]
        L.retro_set_environment(self._cbs[0])
        L.retro_init()
        L.retro_set_video_refresh(self._cbs[1])
        L.retro_set_audio_sample(self._cbs[2])
        L.retro_set_audio_sample_batch(self._cbs[3])
        L.retro_set_input_poll(self._cbs[4])
        L.retro_set_input_state(self._cbs[5])

        self._rom = Path(rom).read_bytes()
        self._rombuf = C.create_string_buffer(self._rom, len(self._rom))
        info = GameInfo(str(rom).encode(), C.cast(self._rombuf, C.c_void_p), len(self._rom), None)
        if not L.retro_load_game(C.byref(info)):
            raise RuntimeError(f"{core} could not load {rom}")
        Libretro._loaded = True
        L.retro_run()  # stable-retro runs one frame on load; match it so recordings replay exactly

    # --- callbacks ---------------------------------------------------------------------------
    def _env(self, cmd, data):
        if cmd == ENV_GET_CAN_DUPE:
            C.cast(data, C.POINTER(C.c_bool))[0] = True
            return True
        if cmd == ENV_SET_PIXEL_FORMAT:
            self.pixel_format = C.cast(data, C.POINTER(C.c_int))[0]
            return True
        if cmd in (ENV_GET_SYSTEM_DIRECTORY, ENV_GET_SAVE_DIRECTORY):
            C.cast(data, C.POINTER(C.c_char_p))[0] = self._sysdir.value
            return True
        if cmd == ENV_GET_VARIABLE_UPDATE:
            C.cast(data, C.POINTER(C.c_bool))[0] = False
            return True
        return False  # everything else: use the core's defaults

    def _video(self, data, width, height, pitch):
        if not data:
            return  # frame dupe
        if self.pixel_format == PIXEL_XRGB8888:
            buf = np.ctypeslib.as_array(C.cast(data, C.POINTER(C.c_uint8)), (height, pitch))
            px = buf[:, : width * 4].reshape(height, width, 4)
            img = px[..., [2, 1, 0]]
        else:
            buf = np.ctypeslib.as_array(C.cast(data, C.POINTER(C.c_uint16)), (height, pitch // 2))
            v = buf[:, :width].astype(np.uint32)
            if self.pixel_format == PIXEL_RGB565:
                r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
                img = np.stack([r << 3, g << 2, b << 3], -1)
            else:
                r, g, b = (v >> 10) & 31, (v >> 5) & 31, v & 31
                img = np.stack([r << 3, g << 3, b << 3], -1)
        if img.shape[1] == 512:  # hi-res frame: fold back to 256 wide
            img = img[:, ::2]
        if img.shape[0] >= 448:
            img = img[::2]
        self.screen = np.ascontiguousarray(img, dtype=np.uint8)

    def _state(self, port, device, index, button_id):
        if port == 0 and device == DEVICE_JOYPAD and button_id < 12:
            return int(self.mask[button_id])
        return 0

    # --- stable-retro compatible API -------------------------------------------------------------
    def set_button_mask(self, mask, player=0):
        # libretro joypad ids use the same order as stable-retro's SNES buttons
        self.mask = np.asarray(mask, np.uint8)

    def step(self):
        self.lib.retro_run()

    def get_screen(self):
        return self.screen

    def get_state(self) -> bytes:
        n = self.lib.retro_serialize_size()
        buf = C.create_string_buffer(n)
        if not self.lib.retro_serialize(buf, n):
            raise RuntimeError("serialize failed")
        return buf.raw

    def set_state(self, state: bytes):
        buf = C.create_string_buffer(bytes(state), len(state))
        if not self.lib.retro_unserialize(buf, len(state)):
            raise RuntimeError("unserialize failed")

    def wram(self) -> np.ndarray:
        ptr = self.lib.retro_get_memory_data(MEMORY_SYSTEM_RAM)
        n = self.lib.retro_get_memory_size(MEMORY_SYSTEM_RAM)
        return np.ctypeslib.as_array(C.cast(ptr, C.POINTER(C.c_uint8)), (n,)).copy()


def find_core() -> str | None:
    """A snes9x core next to the working dir, in $FLYZERO_CORE, or bundled with stable-retro."""
    env = os.environ.get("FLYZERO_CORE")
    if env:
        return env
    for name in ("snes9x_libretro.dll", "snes9x_libretro.so", "snes9x_libretro.dylib"):
        if Path(name).exists():
            return name
    try:
        import importlib.util
        spec = importlib.util.find_spec("stable_retro")
        if spec and spec.origin:
            cands = list((Path(spec.origin).parent / "cores").glob("snes9x_libretro.*"))
            if cands:
                return str(cands[0])
    except Exception:
        pass
    return None
