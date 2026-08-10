"""Voice activity detection: where does one utterance end and the next begin?

Two layers, kept apart so the interesting half needs no neural network:

  VadSegmenter  a pure state machine over speech probabilities. Owns every
                decision that matters — pause length, the hard ceiling, how much
                audio to hand back — and is fully testable with a two-line fake.
  SileroVad     512-sample frames to probabilities, on onnxruntime.

铁律 5: the `silero-vad` pip package pulls 328M of torch to run a 1.2M model.
This module loads the ONNX weights directly and imports onnxruntime only.

There is no partial/interim event, and there must never be one. Design §3(b)
measured v2's "re-transcribe the growing utterance every second" at 172-221%
duty cycle — permanently behind the speaker. An interim event is the hook such a
design would hang itself from.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # silero's window at 16 kHz: 32 ms
CONTEXT_SAMPLES = 64


@dataclass(frozen=True)
class SpeechStart:
    at_sample: int


@dataclass(frozen=True)
class SpeechEnd:
    start_sample: int
    end_sample: int
    audio: np.ndarray
    forced: bool = False
    """True when the cut was imposed rather than heard.

    A forced cut can land mid-word, so downstream code may want to treat the
    text as continuing rather than as a finished sentence.
    """


Event = SpeechStart | SpeechEnd


# --- the ONNX layer ----------------------------------------------------------


def find_silero_onnx() -> Path | None:
    """Locate silero VAD weights already on this machine.

    faster-whisper ships them (it needs the same model for `vad_filter`), and
    faster-whisper is an unconditional dependency here, so on a working install
    this always hits and nothing is downloaded.

    This reads a data file off disk; it does not import faster_whisper or call
    any of its inference code, so 铁律 6 is not in play. The filename has
    changed across releases (v4 -> v5 -> v6), so several are tried and the
    newest wins.
    """
    spec = importlib.util.find_spec("faster_whisper")
    if spec is None or not spec.origin:
        return None

    assets = Path(spec.origin).parent / "assets"
    candidates = sorted(assets.glob("silero_vad*.onnx"), reverse=True)
    return candidates[0] if candidates else None


class SileroVad:
    """Speech probability per 512-sample frame.

    Silero's ONNX signature changed between releases — v4 wants (input, sr, h, c),
    v5 wants (input, state, sr), v6 wants (input, h, c). Rather than pinning a
    version we cannot control, the input names are read off the graph at load
    time and the call is built to match.
    """

    def __init__(self, model_path: Path | None = None):
        import onnxruntime

        path = model_path or find_silero_onnx()
        if path is None:
            raise FileNotFoundError(
                "no silero VAD weights found; expected them to ship with "
                "faster-whisper (pip install faster-whisper)"
            )

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"], sess_options=options
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def speech_prob(self, frame: np.ndarray) -> float:
        if len(frame) != FRAME_SAMPLES:
            raise ValueError(f"expected {FRAME_SAMPLES} samples, got {len(frame)}")

        windowed = np.concatenate([self._context, frame])[None, :].astype(np.float32)
        self._context = frame[-CONTEXT_SAMPLES:].copy()

        feed: dict[str, np.ndarray] = {"input": windowed}
        if "sr" in self._inputs:
            feed["sr"] = np.array(SAMPLE_RATE, dtype=np.int64)
        if "state" in self._inputs:
            feed["state"] = self._state
        if "h" in self._inputs:
            feed["h"] = self._h
            feed["c"] = self._c

        outputs = self._session.run(None, feed)

        if "state" in self._inputs:
            self._state = outputs[1]
        elif "h" in self._inputs:
            self._h, self._c = outputs[1], outputs[2]

        return float(np.ravel(outputs[0])[0])


# --- the state machine -------------------------------------------------------


@dataclass
class VadSegmenter:
    """Turn a stream of audio into utterance boundaries.

    Feed it whatever chunk sizes the audio source produces; it buffers into
    512-sample frames internally so segmentation does not depend on them.
    """

    vad_silence_ms: int = 600
    vad_sensitivity: float = 0.5
    max_utterance_sec: int = 30
    pre_roll_ms: int = 200
    speech_prob: Callable[[np.ndarray], float] | None = None

    _tail: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _history: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _history_origin: int = 0
    _cursor: int = 0
    _in_speech: bool = False
    _speech_start: int = 0
    _silence_frames: int = 0
    _last_voiced: int = 0

    def __post_init__(self):
        if self.speech_prob is None:
            self.speech_prob = SileroVad().speech_prob

    # -- configuration in frames --

    @property
    def _silence_limit(self) -> int:
        return max(1, round(self.vad_silence_ms * SAMPLE_RATE / 1000 / FRAME_SAMPLES))

    @property
    def _max_samples(self) -> int:
        return self.max_utterance_sec * SAMPLE_RATE

    @property
    def _pre_roll(self) -> int:
        return int(self.pre_roll_ms * SAMPLE_RATE / 1000)

    # -- public API --

    def reset(self) -> None:
        self._tail = np.zeros(0, dtype=np.float32)
        self._history = np.zeros(0, dtype=np.float32)
        self._history_origin = self._cursor
        self._in_speech = False
        self._silence_frames = 0

    def feed(self, chunk: np.ndarray) -> list[Event]:
        """Consume audio, return whatever boundaries it revealed."""
        chunk = np.asarray(chunk, dtype=np.float32)
        self._remember(chunk)

        buffer = np.concatenate([self._tail, chunk]) if len(self._tail) else chunk
        usable = len(buffer) // FRAME_SAMPLES * FRAME_SAMPLES
        self._tail = buffer[usable:].copy()

        events: list[Event] = []
        for offset in range(0, usable, FRAME_SAMPLES):
            events.extend(self._step(buffer[offset : offset + FRAME_SAMPLES]))
        return events

    def flush(self) -> list[Event]:
        """Close anything still open — end of file, or the hotkey coming up."""
        if not self._in_speech:
            return []
        return [self._close(self._cursor, forced=True)]

    # -- internals --

    def _remember(self, chunk: np.ndarray) -> None:
        """Keep enough recent audio to satisfy pre-roll and the current utterance."""
        self._history = np.concatenate([self._history, chunk])

        keep_from = (
            self._speech_start - self._pre_roll
            if self._in_speech
            else self._cursor + len(chunk) - self._pre_roll - FRAME_SAMPLES * 4
        )
        keep_from = max(self._history_origin, keep_from, 0)

        drop = keep_from - self._history_origin
        if drop > 0:
            self._history = self._history[drop:]
            self._history_origin = keep_from

    def _slice(self, start: int, end: int) -> np.ndarray:
        lo = max(0, start - self._history_origin)
        hi = max(lo, end - self._history_origin)
        return self._history[lo:hi].copy()

    def _step(self, frame: np.ndarray) -> list[Event]:
        events: list[Event] = []
        voiced = self.speech_prob(frame) >= self.vad_sensitivity
        frame_start = self._cursor
        self._cursor += FRAME_SAMPLES

        if not self._in_speech:
            if voiced:
                self._in_speech = True
                self._speech_start = frame_start
                self._silence_frames = 0
                self._last_voiced = self._cursor
                events.append(SpeechStart(at_sample=frame_start))
            return events

        if voiced:
            self._silence_frames = 0
            self._last_voiced = self._cursor
        else:
            self._silence_frames += 1
            if self._silence_frames >= self._silence_limit:
                # Cut at the last voiced frame, not here — the trailing silence
                # is the detector's evidence, not part of what was said.
                events.append(self._close(self._last_voiced, forced=False))
                return events

        if self._cursor - self._speech_start >= self._max_samples:
            # Design §5: a speaker who never pauses, or a VAD false negative
            # that never hears one, must not buffer without limit. Cut and
            # reopen immediately — they are still talking.
            events.append(self._close(self._cursor, forced=True))
            self._in_speech = True
            self._speech_start = self._cursor
            self._silence_frames = 0
            self._last_voiced = self._cursor
            events.append(SpeechStart(at_sample=self._cursor))

        return events

    def _close(self, end_sample: int, *, forced: bool) -> SpeechEnd:
        start = max(0, self._speech_start - self._pre_roll)
        audio = self._slice(start, end_sample)

        self._in_speech = False
        self._silence_frames = 0

        return SpeechEnd(
            start_sample=start, end_sample=end_sample, audio=audio, forced=forced
        )


def has_speech(
    audio: np.ndarray,
    speech_prob: Callable[[np.ndarray], float] | None = None,
    sensitivity: float = 0.5,
    min_voiced_frames: int = 3,
) -> bool:
    """Is there actually anyone talking in here?

    Push-to-talk has no VAD in its segmentation path — the finger decides the
    boundaries — which leaves one hole: a hotkey pressed and released without
    speaking hands Whisper a buffer of room tone. Whisper does not return
    nothing for that. It hallucinates, confidently and briefly: measured on this
    machine, two seconds of an empty room produced "you" and "Good job."

    Injecting a fabricated sentence into the author's document is worse than any
    latency problem in this project, so the buffer is checked before the model
    ever sees it. The check costs about 0.4% of real time (design §5.4
    measurement), which is free next to the second it saves when the answer is
    no.
    """
    if audio is None or len(audio) < FRAME_SAMPLES:
        return False

    if speech_prob is None:
        speech_prob = SileroVad().speech_prob

    voiced = 0
    usable = len(audio) // FRAME_SAMPLES * FRAME_SAMPLES
    for offset in range(0, usable, FRAME_SAMPLES):
        if speech_prob(audio[offset : offset + FRAME_SAMPLES]) >= sensitivity:
            voiced += 1
            if voiced >= min_voiced_frames:
                return True
    return False
