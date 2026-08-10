"""The resident dictation daemon — where P2a becomes something you can use.

Resident rather than launched per use. P1 measured the first utterance at 2.76s
against 1.20s for every one after: model load plus the first Metal kernel
compile. Launching per use would charge that to every first sentence, which is
exactly the sentence a user judges the tool by.

Threading. The hotkey callback arrives on pynput's listener thread, and
transcription takes over a second. Doing that work on the callback thread would
stall key delivery for the whole system, so utterances go onto a queue and a
single worker drains it. Single, not a pool: 铁律 9 means text is delivered once
and never revised, so it has to be delivered in order.

Push-to-talk sends the whole held buffer as one utterance — the finger is the
segmenter, and no VAD is involved. Toggle mode works here too but still only
transcribes when you stop; streaming a long dictation clause by clause is P2b
(design §4.1a), and needs the paragraph batching and overlay that go with it.
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from backend.audio_source import MicSource
from backend.config import AppConfig
from backend.hotkey import HotkeyError, HotkeyEvent, HotkeyListener
from backend.injection import Injector
from backend.pipeline import Utterance
from backend.scratchpad import Scratchpad, SessionArchive
from backend.timing import Stopwatch
from backend.vad import SAMPLE_RATE, SileroVad, has_speech

log = logging.getLogger(__name__)

__all__ = ["DictationDaemon", "HotkeyError"]

WARM_UP_SECONDS = 1.0
TARGET_MS_WITHOUT_POLISH = 1500
TARGET_MS_WITH_POLISH = 3000


@dataclass
class _Job:
    index: int
    audio: np.ndarray
    started_at: float
    """`perf_counter` at the moment the hotkey came up — the user's t=0."""


@dataclass
class DictationDaemon:
    config: AppConfig
    stt: object
    injector: object = field(default_factory=Injector)
    make_mic: Callable[[], object] | None = None
    hotkey_factory: Callable[[Callable[[HotkeyEvent], None]], object] | None = None
    polish: Callable[..., str] | None = None
    on_text: Callable[[Utterance], None] | None = None
    speech_check: Callable[[np.ndarray], bool] | None = None
    """Overridable so tests can drive the wiring without a real VAD session."""

    scratchpad: Scratchpad = field(init=False)
    hotkey: object = field(init=False, default=None)
    last_timing: Stopwatch | None = field(init=False, default=None)
    running: bool = field(init=False, default=False)

    _mic: object | None = field(init=False, default=None)
    _queue: "queue.Queue[_Job]" = field(init=False, default_factory=queue.Queue)
    _worker: threading.Thread | None = field(init=False, default=None)
    _index: int = field(init=False, default=0)
    _previous_text: str | None = field(init=False, default=None)
    _stop_flag: threading.Event = field(init=False, default_factory=threading.Event)
    _idle: threading.Event = field(init=False, default_factory=threading.Event)
    _vad: object | None = field(init=False, default=None)

    def __post_init__(self):
        self.scratchpad = Scratchpad(archive=SessionArchive())
        self._idle.set()
        if self.make_mic is None:
            self.make_mic = lambda: MicSource(device_index=self.config.input_device)
        if self.hotkey_factory is None:
            self.hotkey_factory = lambda on_event: HotkeyListener(
                on_event=on_event,
                combination=self.config.hotkey,
                mode=self.config.hotkey_mode,
            )

    # -- lifecycle --

    def start(self) -> "DictationDaemon":
        self.hotkey = self.hotkey_factory(self._on_hotkey)
        self.hotkey.start()  # raises HotkeyError without Accessibility

        self._stop_flag.clear()
        self._worker = threading.Thread(target=self._drain, daemon=True, name="utter-dictation")
        self._worker.start()

        self._warm_up()
        self.running = True
        return self

    def stop(self) -> None:
        if not self.running and self.hotkey is None:
            return

        self._stop_flag.set()
        if self._mic is not None:
            self._mic.stop()
            self._mic = None
        if self.hotkey is not None:
            self.hotkey.close()
            self.hotkey = None
        if self._worker is not None:
            self._worker.join(timeout=5)
            self._worker = None

        self.injector.release()
        self.running = False

    def run_forever(self) -> None:  # pragma: no cover - interactive
        """Block until interrupted. Assumes start() has already run."""
        stopping = threading.Event()
        try:
            signal.signal(signal.SIGINT, lambda *_: stopping.set())
            signal.signal(signal.SIGTERM, lambda *_: stopping.set())
        except ValueError:
            # Not on the main thread — the caller owns interruption.
            pass
        while not stopping.wait(0.2):
            pass

    def _warm_up(self) -> None:
        """Pay the model load now, so the author's first sentence does not.

        A failure here is not fatal: a cold model is slow, not broken.
        """
        try:
            silence = np.zeros(int(SAMPLE_RATE * WARM_UP_SECONDS), dtype=np.float32)
            started = time.perf_counter()
            self.stt.transcribe(silence, language=self.config.dictate_language)
            log.info("model warmed in %.2fs", time.perf_counter() - started)
        except Exception:
            log.warning("model warm-up failed; the first utterance will be slow", exc_info=True)

    # -- hotkey --

    def _on_hotkey(self, event: HotkeyEvent) -> None:
        if event.kind == "start":
            self.begin_utterance()
        else:
            self.end_utterance(at=event.at)

    def begin_utterance(self) -> None:
        if self._mic is not None:
            return  # key repeat, or a second press before the first was released

        # Lock the target now rather than at the end: by then the user may have
        # switched windows, and the text belongs where they started talking.
        self.injector.lock_target()

        try:
            self._mic = self.make_mic().start()
        except Exception:
            log.warning("could not open the microphone", exc_info=True)
            self._mic = None

    def end_utterance(self, at: float | None = None) -> None:
        mic, self._mic = self._mic, None
        if mic is None:
            return

        # Drain and queue BEFORE closing the stream. Measured on this machine:
        # draining takes 0.1ms, concatenating 0.1ms, and PortAudio's close takes
        # 130ms. Closing first put all of that on the user's critical path for
        # no reason — the audio is already in hand by then.
        chunks = list(mic.chunks())
        if chunks:
            job = _Job(
                index=self._index,
                audio=np.concatenate(chunks),
                started_at=at if at is not None else time.perf_counter(),
            )
            self._index += 1
            self._idle.clear()
            self._queue.put(job)

        threading.Thread(target=mic.stop, daemon=True, name="utter-mic-close").start()

    # -- the worker --

    def _drain(self) -> None:
        while not self._stop_flag.is_set():
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                self._idle.set()
                continue
            try:
                self._process(job)
            except Exception:  # pragma: no cover - the worker must outlive a bad job
                log.warning("dictation %d failed", job.index, exc_info=True)
            finally:
                self._queue.task_done()
                if self._queue.empty():
                    self._idle.set()

    def _process(self, job: _Job) -> None:
        watch = Stopwatch(
            target_ms=TARGET_MS_WITH_POLISH if self._polishing else TARGET_MS_WITHOUT_POLISH
        )
        watch.mark("hotkey → buffer closed", (time.perf_counter() - job.started_at) * 1000)

        # The silence gate. Push-to-talk has no VAD in its segmentation path, so
        # a hotkey pressed without speaking hands Whisper a buffer of room tone
        # — and Whisper does not answer nothing. Measured here: two seconds of
        # an empty room produced "you" and then "Good job.". Injecting a
        # fabricated sentence into the author's document is worse than any
        # latency problem in this project.
        with watch.span("speech check"):
            speaking = (self.speech_check or self._has_speech)(job.audio)
        if not speaking:
            log.info("utterance %d contained no speech, discarded", job.index)
            watch.skip("transcription", "no speech")
            self.last_timing = watch
            return

        try:
            with watch.span("transcription"):
                raw = self.stt.transcribe(
                    job.audio,
                    language=self.config.dictate_language,
                    initial_prompt=self._vocabulary_prompt(),
                )
        except Exception:
            # One utterance lost, the session continues. Nothing has been shown
            # to the user yet, so there is nothing inconsistent to clean up.
            log.warning("transcription failed for utterance %d", job.index, exc_info=True)
            self.last_timing = watch
            return

        raw = (raw or "").strip()
        if not raw:
            self.last_timing = watch
            return

        text, polished = raw, False
        if self._polishing:
            with watch.span("polish"):
                text, polished = self._safe_polish(raw)
        else:
            watch.skip("polish", "off")

        utterance = Utterance(
            index=job.index,
            mode="dictate",
            text=text,
            raw_text=raw,
            start_sec=0.0,
            end_sec=len(job.audio) / SAMPLE_RATE,
            polished=polished,
        )

        # Archive first, always. Everything after this can fail without costing
        # the words (铁律 8).
        self.scratchpad.add(utterance)

        if self.config.dictate_target == "cursor":
            with watch.span("clipboard + paste"):
                self.injector.inject(job.index, text)

        if self.on_text is not None:
            try:
                self.on_text(utterance)
            except Exception:  # pragma: no cover
                log.warning("on_text callback failed", exc_info=True)

        self.last_timing = watch
        if watch.over_target:
            log.warning("dictation %d took %.0f ms", job.index, watch.total_ms)

    # -- helpers --

    @property
    def _polishing(self) -> bool:
        return bool(self.config.polish_enabled and self.polish is not None)

    def _has_speech(self, audio: np.ndarray) -> bool:
        """Reuses one silero session for the life of the daemon — constructing
        an onnxruntime session per utterance would cost more than the check."""
        try:
            if self._vad is None:
                self._vad = SileroVad()
            self._vad.reset()
            return has_speech(
                audio,
                speech_prob=self._vad.speech_prob,
                sensitivity=self.config.vad_sensitivity,
            )
        except Exception:
            # If the gate itself is broken, let the audio through. A hallucinated
            # sentence is bad; refusing to transcribe anything is worse.
            log.warning("speech check failed, transcribing anyway", exc_info=True)
            return True

    def _vocabulary_prompt(self) -> str | None:
        return ", ".join(self.config.vocabulary) if self.config.vocabulary else None

    def _safe_polish(self, raw: str) -> tuple[str, bool]:
        try:
            result = self.polish(raw, context=self._previous_text)
        except Exception:
            log.warning("polish failed, keeping the raw transcript", exc_info=True)
            return raw, False
        finally:
            self._previous_text = raw
        return (result.strip(), True) if result and result.strip() else (raw, False)

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Block until the queue is drained. For tests and for clean shutdown."""
        return self._idle.wait(timeout)
