"""系统音频 — capturing what the computer is playing, for online meetings.

The v3 design said this needed BlackHole, a virtual audio driver the user would
have to install. That was true in April and is not true now: macOS 14.4 added
Core Audio process taps, which need only the "System Audio Recording"
permission and no driver at all.

Verified on this machine 2026-08-16, step by step, because the API is barely
documented and every step had a way to fail quietly:

    tap created                     err=0, id=171
    aggregate device created        err=0, id=172
    visible to PortAudio            index 6, 48000 Hz, 2ch
    samples actually arriving       3.40s captured, peak 0.1674

Two traps cost real time and are worth writing down.

**The aggregate device description is a CFDictionary, so its keys must be
CFStrings.** PyObjC exposes the key constants as `bytes`, and passing those
straight through segfaults the interpreter — no exception, no message, no
output at all. `.decode()` on every key is the whole fix.

**PortAudio will happily pretend to resample this device.** Asking for 16 kHz
mono passes `check_input_settings`, opens a stream, and delivers *exact zeros*.
That is this project's oldest lesson in a new place: a perfect run of zeros is
never a measurement, it is an `if` in somebody's software. So the stream is
opened at the device's own rate and converted here.

The conversion is numpy only. scipy would be the obvious tool and it is even
installed — but it arrives as a dependency of mlx-whisper, which exists only on
Apple Silicon, and 铁律 6 says this has to work on the author's other machine
too.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

from backend.vad import SAMPLE_RATE

log = logging.getLogger(__name__)

AGGREGATE_NAME = "Utter 听记"
AGGREGATE_UID = "com.dylan.utter.listen-aggregate"

#: Anti-aliasing cutoff, below the 8 kHz Nyquist of our 16 kHz target with
#: enough margin that the filter's shoulder does not reach it. Speech has
#: little energy up here; music does, which is why filtering at all matters.
CUTOFF_HZ = 7200.0
TAPS = 63


class SystemAudioError(RuntimeError):
    pass


def pid_to_audio_object(pid: int) -> int | None:
    """A process id as Core Audio knows it, or None.

    Taps are described in terms of audio objects, not pids: handing
    `initMonoMixdownOfProcessesIDs_` a raw pid returns `!obj` (bad object).
    This is the translation step, and it is the only place in the project that
    has to hand PyObjC a raw C buffer — the out-parameter is a 4-byte
    `AudioObjectID`, so a `bytearray(4)` is what it wants.
    """
    import struct

    import CoreAudio

    address = CoreAudio.AudioObjectPropertyAddress(
        CoreAudio.kAudioHardwarePropertyTranslatePIDToProcessObject,
        CoreAudio.kAudioObjectPropertyScopeGlobal,
        CoreAudio.kAudioObjectPropertyElementMain)
    qualifier = struct.pack("i", int(pid))
    try:
        err, _size, data = CoreAudio.AudioObjectGetPropertyData(
            CoreAudio.kAudioObjectSystemObject, address,
            len(qualifier), qualifier, 4, bytearray(4))
    except Exception:
        log.warning("翻译 pid %s 失败", pid, exc_info=True)
        return None
    if err != 0 or not data:
        return None
    obj = struct.unpack("I", bytes(data)[:4])[0]
    return obj or None


def pids_for(app_name: str) -> list[int]:
    """Process ids whose command line mentions `app_name`."""
    import subprocess

    try:
        out = subprocess.run(["pgrep", "-f", app_name], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(line) for line in out.split() if line.isdigit()]


def _lowpass(cutoff_hz: float, rate: int, taps: int = TAPS) -> np.ndarray:
    """A windowed-sinc low-pass, normalised to unity gain at DC."""
    fc = min(cutoff_hz / rate, 0.49)
    n = np.arange(taps) - (taps - 1) / 2
    h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(taps)
    return (h / h.sum()).astype(np.float32)


class _Resampler:
    """Down to 16 kHz mono, keeping filter state across chunks.

    Stateful on purpose: filtering each chunk independently puts a click at
    every boundary, forty times a second, which a VAD hears as speech.
    """

    def __init__(self, rate: int):
        self.rate = rate
        self._h = _lowpass(CUTOFF_HZ, rate)
        self._tail = np.zeros(len(self._h) - 1, dtype=np.float32)
        self._position = 0.0

    def __call__(self, block: np.ndarray) -> np.ndarray:
        mono = block.mean(axis=1) if block.ndim > 1 else block
        mono = np.asarray(mono, dtype=np.float32)

        padded = np.concatenate([self._tail, mono])
        filtered = np.convolve(padded, self._h, mode="valid")
        self._tail = padded[-(len(self._h) - 1):]

        step = self.rate / SAMPLE_RATE
        count = int((len(filtered) - self._position) / step)
        if count <= 0:
            return np.zeros(0, dtype=np.float32)
        idx = self._position + np.arange(count) * step
        out = np.interp(idx, np.arange(len(filtered)), filtered)
        # Carry the fractional remainder, so 48000/16000 stays exact and an
        # awkward rate like 44100 does not drift over an hour.
        self._position = idx[-1] + step - len(filtered)
        return out.astype(np.float32)


class SystemAudioSource:
    """What the computer is playing, shaped like `MicSource`.

    Same surface as the microphone — `start`, `stop`, `chunks`, `device_name`,
    `stopped_reason` — so the pipeline does not know which one it is holding.
    That is the whole point of the AudioSource slot in design §4.
    """

    def __init__(self, chunk_samples: int = 1600, max_chunks: int = 400,
                 pids: list[int] | None = None):
        #: Which applications to listen to. None taps everything the machine
        #: plays, which is measurably the wrong default: a global tap during a
        #: test picked up an unrelated video playing in another window and
        #: transcribed it into the middle of the meeting. Naming the meeting
        #: application keeps the transcript clean and keeps everything else the
        #: user happens to be playing out of the recording.
        self.pids = list(pids or [])
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_chunks)
        self._chunk_samples = chunk_samples
        self._tap = None
        self._aggregate = None
        self._stream = None
        self._resampler = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

        self.device_name = AGGREGATE_NAME
        self.dropped = 0
        self.overflows = 0
        self.last_chunk = None
        self.stopped_reason: str | None = None
        self.first_chunk_at: float | None = None

    # -- Core Audio plumbing ----------------------------------------------

    @staticmethod
    def available() -> tuple[bool, str]:
        try:
            import CoreAudio
        except ImportError:
            return False, "pyobjc 的 CoreAudio 模块没装"
        if not hasattr(CoreAudio, "AudioHardwareCreateProcessTap"):
            return False, "这台机器的 macOS 太旧，没有 Core Audio Taps（需要 14.4+）"
        return True, ""

    def _create_tap(self):
        import CoreAudio

        key = lambda k: k.decode() if isinstance(k, bytes) else k  # noqa: E731

        objects = [obj for obj in (pid_to_audio_object(p) for p in self.pids) if obj]
        if self.pids and not objects:
            raise SystemAudioError(
                "指定的程序没有在放声音，或者已经退出了")
        if objects:
            description = CoreAudio.CATapDescription.alloc(
                ).initMonoMixdownOfProcessesIDs_(objects)
        else:
            description = CoreAudio.CATapDescription.alloc(
                ).initStereoGlobalTapButExcludeProcesses_([])
        description.setName_("Utter 听记")
        description.setPrivate_(True)
        # Unmuted: tapping must not silence what the user is listening to.
        description.setMuteBehavior_(CoreAudio.CATapUnmuted)
        tap_uid = str(description.UUID().UUIDString())

        err, self._tap = CoreAudio.AudioHardwareCreateProcessTap(description, None)
        if err != 0:
            raise SystemAudioError(
                f"建不了系统音频通道（错误 {err}）。"
                "多半是「系统设置 → 隐私与安全性 → 系统录音」里没给 Utter 权限。")

        # Every key .decode()d: these are CFDictionary keys and must be
        # CFStrings. Passing pyobjc's raw bytes constants segfaults outright.
        layout = {
            key(CoreAudio.kAudioAggregateDeviceNameKey): AGGREGATE_NAME,
            key(CoreAudio.kAudioAggregateDeviceUIDKey): AGGREGATE_UID,
            key(CoreAudio.kAudioAggregateDeviceIsPrivateKey): 1,
            key(CoreAudio.kAudioAggregateDeviceIsStackedKey): 0,
            key(CoreAudio.kAudioAggregateDeviceTapAutoStartKey): 1,
            key(CoreAudio.kAudioAggregateDeviceSubDeviceListKey): [],
            key(CoreAudio.kAudioAggregateDeviceTapListKey): [{
                key(CoreAudio.kAudioSubTapUIDKey): tap_uid,
                key(CoreAudio.kAudioSubTapDriftCompensationKey): 1,
            }],
        }
        err, self._aggregate = CoreAudio.AudioHardwareCreateAggregateDevice(layout, None)
        if err != 0:
            raise SystemAudioError(f"建不了聚合音频设备（错误 {err}）")

    def _destroy(self) -> None:
        import CoreAudio

        if self._aggregate:
            CoreAudio.AudioHardwareDestroyAggregateDevice(self._aggregate)
            self._aggregate = None
        if self._tap:
            CoreAudio.AudioHardwareDestroyProcessTap(self._tap)
            self._tap = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "SystemAudioSource":
        if self._stream is not None:
            return self

        import sounddevice as sd

        from backend.audio_source import refresh_devices

        ok, why = self.available()
        if not ok:
            raise SystemAudioError(why)

        self._create_tap()
        try:
            # PortAudio enumerated its devices at init and the aggregate did
            # not exist then. Without this it is simply not in the list.
            refresh_devices()
            index = self._find_device(sd)
            info = sd.query_devices(index)
            rate = int(info["default_samplerate"])
            channels = int(info["max_input_channels"])
            self._resampler = _Resampler(rate)

            # The device's own rate and channel count, never a requested
            # conversion — asking PortAudio for 16 kHz mono here returns exact
            # zeros while reporting success.
            self._stream = sd.InputStream(
                device=index, channels=channels, samplerate=rate,
                dtype="float32", callback=self._on_audio)
            self._stream.start()
            log.info("系统音频已打开：设备 %d，%dHz %d声道 → 16kHz 单声道",
                     index, rate, channels)
        except Exception:
            self._destroy()
            raise
        return self

    def _find_device(self, sd) -> int:
        for index, device in enumerate(sd.query_devices()):
            if AGGREGATE_NAME in device["name"] and device["max_input_channels"] > 0:
                return index
        raise SystemAudioError("聚合设备建好了，但音频系统看不到它")

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            self.overflows += 1
        if self.first_chunk_at is None:
            self.first_chunk_at = time.perf_counter()

        converted = self._resampler(indata)
        if not len(converted):
            return
        with self._lock:
            self._pending = np.concatenate([self._pending, converted])
            while len(self._pending) >= self._chunk_samples:
                block = self._pending[:self._chunk_samples]
                self._pending = self._pending[self._chunk_samples:]
                self.last_chunk = block
                try:
                    self._queue.put_nowait(block)
                except queue.Full:
                    self.dropped += 1

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        self._destroy()

    def chunks(self):
        """Yield buffered audio, then stop. Never blocks — same contract as
        `MicSource`, so the caller owns the loop."""
        while True:
            try:
                yield self._queue.get_nowait()
            except queue.Empty:
                return

    @property
    def running(self) -> bool:
        return self._stream is not None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_exc):
        self.stop()
