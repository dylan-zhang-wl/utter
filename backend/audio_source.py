"""The microphone, as a stream of 16 kHz mono chunks.

Not a revival of v1's `audio_capture.py`. That one is built on pyaudio, which P1
did not adopt, and it wraps the import in a try/except that leaves the class
constructible but silently inert — the failure mode this project spends most of
its rules avoiding.

One design note worth stating: the queue between the audio callback and the
consumer is **bounded, and drops the oldest chunk when full**. PortAudio's
callback runs on a realtime thread and cannot block, so something has to give
when the consumer stalls. An unbounded queue would trade a visible stall for
unbounded memory and a lag that grows silently — the user would hear themselves
finish a sentence and watch text arrive from thirty seconds ago. Dropping the
oldest keeps latency honest and makes the loss countable, and the count goes
into the timing report.
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100 ms
MAX_CHUNKS = 100  # 10 seconds of slack before anything is dropped


class AudioSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    channels: int
    is_default: bool


def list_devices() -> list[Device]:
    """Input-capable devices only, with the system default marked.

    `utter doctor` prints this. On the author's machine the default is a
    Bluetooth speaker and no built-in microphone is listed at all, which is
    worth seeing rather than guessing at.
    """
    try:
        default_index = sd.default.device[0]
    except Exception:  # pragma: no cover - defensive
        default_index = None

    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] < 1:
            continue
        devices.append(
            Device(
                index=index,
                name=info["name"],
                channels=info["max_input_channels"],
                is_default=(index == default_index),
            )
        )
    return devices


class MicSource:
    """Microphone capture. Use as a context manager."""

    def __init__(
        self,
        device_index: int | None = None,
        chunk_samples: int = CHUNK_SAMPLES,
        max_chunks: int = MAX_CHUNKS,
    ):
        self.device_index = device_index
        self.chunk_samples = chunk_samples
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_chunks)
        self._stream = None
        self.dropped = 0
        self.overflows = 0
        self.stopped_reason: str | None = None

    # -- lifecycle --

    def start(self) -> "MicSource":
        if self._stream is not None:
            return self

        try:
            sd.check_input_settings(
                device=self.device_index,
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype="float32",
            )
        except Exception as exc:
            name = self.device_index if self.device_index is not None else "the default device"
            raise AudioSourceError(f"cannot open audio input {name}: {exc}") from exc

        self._stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=SAMPLE_RATE,
            dtype="float32",
            blocksize=self.chunk_samples,
            callback=self._on_audio,
        )
        self._stream.start()
        return self

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()

    def fail(self, reason: str) -> None:
        """Stop because something went wrong, keeping the reason readable.

        Bluetooth microphones disconnect mid-sentence. The daemon needs to say
        so, not surface a traceback from inside PortAudio.
        """
        self.stopped_reason = reason
        log.warning("audio input stopped: %s", reason)
        self.stop()

    def __enter__(self) -> "MicSource":
        return self.start()

    def __exit__(self, *_exc_info) -> None:
        self.stop()

    # -- the realtime callback --

    def _on_audio(self, frames, _count, _time, status) -> None:
        if status:
            # An overflow means we were too slow for one buffer. Worth counting,
            # never worth killing a dictation in progress over.
            self.overflows += 1

        audio = np.asarray(frames, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        try:
            self._queue.put_nowait(audio)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self.dropped += 1
                self._queue.put_nowait(audio)
            except queue.Empty:  # pragma: no cover - the consumer just drained it
                pass

    # -- consumption --

    def chunks(self) -> Iterator[np.ndarray]:
        """Yield buffered audio, then stop. Never blocks waiting for more.

        The caller owns the loop: a dictation ends when the hotkey comes up, not
        when the microphone runs out of sound, so this must not block on an
        empty queue.
        """
        while True:
            try:
                yield self._queue.get_nowait()
            except queue.Empty:
                return

    @property
    def running(self) -> bool:
        return self._stream is not None
