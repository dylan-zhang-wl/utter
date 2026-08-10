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
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100 ms
# Big enough to hold an entire utterance, not merely to smooth over a stall.
#
# This was 100 chunks — ten seconds — which was correct for a consumer that
# drains continuously and wrong for the one we actually have. In push-to-talk
# nothing drains the queue until the key comes up, so the queue *is* the
# recording. Past ten seconds it began discarding the oldest audio, and the
# author got back only the tail of what they had said, with no indication that
# anything was missing. Reported 2026-08-10.
#
# Five minutes. The first attempt at this fix used 35 seconds — design §5.4's
# VAD force-cut ceiling plus margin — which was the wrong number copied from the
# wrong mode. That ceiling exists so LISTEN mode cannot buffer a speaker who
# never pauses. In push-to-talk the length is decided by the author's finger,
# and they hit 35 seconds on their second real attempt.
#
# There was never a reason to be frugal: a chunk is 1600 float32 samples, so
# five minutes of audio is 19 MB. Memory was not the constraint; an unexamined
# constant was.
MAX_CHUNKS = 3000


class AudioSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    channels: int
    is_default: bool


def shared_with_output(device_index: int | None = None) -> str | None:
    """Name of the input device if it is also the default output, else None.

    A headset used as both is almost always Bluetooth, and on macOS opening its
    microphone drags the whole device from A2DP to the hands-free profile:
    mono, low bitrate, extra latency. The author noticed it as video going out
    of sync and the system volume changing — neither of which looks like it has
    anything to do with a dictation tool.

    No code can prevent that; it is what CoreAudio does when an app asks a
    Bluetooth headset for input. What code can do is not trigger it when nobody
    asked to dictate, and say out loud that it is happening.
    """
    try:
        default_in, default_out = sd.default.device
        index = device_index if device_index is not None else default_in
        if index is None or default_out is None:
            return None
        name = sd.query_devices(index)["name"]
        return name if name == sd.query_devices(default_out)["name"] else None
    except Exception:  # pragma: no cover - defensive
        return None


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
        #: Newest chunk, for the level meter. A *copy* of what went into the
        #: queue — reading the queue to drive a display would consume audio the
        #: transcription needs.
        self.last_chunk = None
        self.stopped_reason: str | None = None
        #: `perf_counter` when the first chunk arrived. Against the moment the
        #: hotkey went down, this is exactly how much of the author's first word
        #: was never recorded — the number they have been describing as "the
        #: first second or two goes missing".
        self.first_chunk_at: float | None = None

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

        if self.first_chunk_at is None:
            self.first_chunk_at = time.perf_counter()

        audio = np.asarray(frames, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        self.last_chunk = audio

        try:
            self._queue.put_nowait(audio)
        except queue.Full:
            # Something has to give — the callback runs on a realtime thread and
            # cannot block. Dropping the oldest keeps the most recent speech,
            # but it is still lost audio, so it is counted and the count is
            # surfaced all the way to the user's timing report. Silent loss is
            # the one thing this project does not do.
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
