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

from backend.audio_source import MicSource, shared_with_output
from backend.config import AppConfig
from backend.hotkey import HotkeyError, HotkeyEvent, HotkeyListener
from backend.injection import Injector
from backend.pipeline import Utterance
from backend.punctuation import close_sentence, collapse_repetition
from backend.punctuation import normalise as normalise_punctuation
from backend.scratchpad import Scratchpad, SessionArchive
from backend.timing import Stopwatch
from backend.vad import SAMPLE_RATE, SileroVad, SpeechEnd, VadSegmenter, has_speech

log = logging.getLogger(__name__)

__all__ = ["DictationDaemon", "HotkeyError"]

WARM_UP_SECONDS = 1.0

# Past this, a single hold is split at its own pauses before transcription.
#
# Whisper punctuates what it can see the shape of. Handed 65 seconds of
# unbroken speech — measured on the author's own dictation — it returns 65
# seconds of unbroken text with a single full stop at the end, because nothing
# in that block tells it where one thought finished. Cut at the pauses the
# speaker already made and each piece comes back punctuated.
#
# 20s rather than lower: below that Whisper's own windowing copes, and each
# extra cut costs another ~1s model pass.
# Splitting a long hold into clauses is DISABLED. Set a number of seconds to
# re-enable it, but read this first.
#
# The idea was that Whisper punctuates what it can see the shape of, so cutting
# a long hold at its pauses would give each clause its own punctuation. Measured
# against the author's real dictation, it does not:
#
#   16.4s, not split  -> 4 punctuation marks
#   24.2s, split      -> 2 punctuation marks
#
# And a 3.4-second Chinese question — exactly what one of those pieces looks
# like — comes back as 「所以这个标点的问题到底应该怎么解决了。」 with no question
# mark and no internal comma. Whisper's Chinese punctuation is sparse at every
# length, so there was nothing for the split to unlock.
#
# It also cost real time: five model passes for a 30-second hold where one would
# do, which is the slowness the author noticed. Deciding where a comma belongs
# in a run-on Chinese sentence is a language task, not an audio one — it belongs
# to polish, which 铁律 10 explicitly permits to add punctuation.
SEGMENT_ABOVE_SECONDS = None

# Silence threshold used ONLY when carving up a long hold. Deliberately far
# below config.vad_silence_ms (600ms), which answers a different question.
#
# 600ms means "that utterance is over" — right for listen mode, wrong here. The
# author dictates fluently, breathing for two or three hundred milliseconds
# between clauses, so at 600ms a 30-second hold yielded exactly one piece and
# went to Whisper whole: 104 characters, two punctuation marks. What we want
# here is the clause boundary, and that is what a breath is.
CLAUSE_SILENCE_MS = 320
TARGET_MS_WITHOUT_POLISH = 1500
TARGET_MS_WITH_POLISH = 3000


@dataclass
class _Job:
    index: int
    audio: np.ndarray
    started_at: float
    dropped: int = 0
    """`perf_counter` at the moment the hotkey came up — the user's t=0."""


@dataclass
class DictationDaemon:
    config: AppConfig
    stt: object
    injector: object | None = None
    make_mic: Callable[[], object] | None = None
    hotkey_factory: Callable[[Callable[[HotkeyEvent], None]], object] | None = None
    polish: Callable[..., str] | None = None
    on_text: Callable[[Utterance], None] | None = None
    speech_check: Callable[[np.ndarray], bool] | None = None
    """Overridable so tests can drive the wiring without a real VAD session."""
    overlay: object | None = None
    """The floating indicator, or None to run headless. Optional on purpose:
    the daemon must keep working when there is no window server at all."""

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
    _locked_at_press: str | None = field(init=False, default=None)
    _level_stop: threading.Event = field(init=False, default_factory=threading.Event)

    def __post_init__(self):
        if self.injector is None:
            self.injector = Injector(
                settle_seconds=self.config.paste_settle_ms / 1000,
                type_out_fallback=self.config.type_out_fallback,
            )
        self.scratchpad = Scratchpad(archive=SessionArchive())
        self._idle.set()
        if self.make_mic is None:
            self.make_mic = lambda: MicSource(
                device_index=self.config.input_device,
                # Deliberately NOT derived from max_utterance_sec: that ceiling
                # governs the VAD's force-cut in listen mode, and using it here
                # capped push-to-talk at 35 seconds. The author's finger decides
                # how long an utterance is; 19 MB buys five minutes of it.
            )
        if self.hotkey_factory is None:
            self.hotkey_factory = self._default_hotkeys

    def _default_hotkeys(self, on_event):
        """Both gestures at once, each on its own key.

        Returns something with start/close, so the daemon does not care whether
        one listener is running or two.
        """
        bindings = []
        if self.config.hotkey_push:
            bindings.append((self.config.hotkey_push, "push"))
        if self.config.hotkey_toggle:
            bindings.append((
                self.config.hotkey_toggle,
                "double_toggle" if self.config.hotkey_toggle_double_tap else "toggle",
            ))
        if not bindings:
            raise HotkeyError(
                "没有配置任何热键。请在 ~/Utter/config.json 里设 hotkey_push 或 "
                "hotkey_toggle；用 `utter keys` 找一个没被别的 app 占用的键。"
            )
        # One listener, both bindings. See HotkeyListener's docstring: a second
        # listener plus the injector's Controller aborts the process.
        return HotkeyListener(on_event=on_event, bindings=bindings)

    # -- lifecycle --

    def start(self) -> "DictationDaemon":
        # Build the keyboard Controller before any listener thread exists.
        # Both touch macOS's non-reentrant keycode_context, and doing it in this
        # order means they never contend.
        warm = getattr(getattr(self.injector, "keyboard", None), "warm_up", None)
        if warm is not None:
            try:
                warm()
            except Exception:  # pragma: no cover - defensive
                log.warning("could not pre-build the keyboard controller", exc_info=True)

        self.hotkey = self.hotkey_factory(self._on_hotkey)
        self.hotkey.start()  # raises HotkeyError without Accessibility

        self._stop_flag.clear()
        self._worker = threading.Thread(target=self._drain, daemon=True, name="utter-dictation")
        self._worker.start()

        self._warm_up()
        self._warm_up_audio()
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

        # Anything still held has to go somewhere. flush() falls back to the
        # clipboard with a notification when the target is gone, so quitting
        # never silently discards words the author said.
        try:
            if getattr(self.injector, "pending", 0):
                self.injector.flush()
        except Exception:  # pragma: no cover
            log.warning("could not flush on shutdown", exc_info=True)

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

    def switch_provider(self, provider_id: str) -> tuple[bool, str]:
        """Change engine without restarting. Returns (worked, message).

        Comparing two models means running the same sentences through both, and
        anything that makes switching cost a restart makes that comparison not
        happen. The daemon holds one reference; replacing it is enough because
        nothing else in the process knows which engine is in use.
        """
        from backend.providers.stt import get_stt_provider

        try:
            provider = get_stt_provider(preferred=provider_id)
        except Exception as exc:
            return False, str(exc)

        if provider.id != provider_id:
            return False, f"{provider_id} 不可用，仍在用 {provider.id}"

        self.stt = provider
        self.config.stt_provider = provider_id
        self._save_config()
        self._warm_up()
        log.info("switched to %s", provider.display_name)
        return True, provider.display_name

    def _save_config(self) -> None:
        """Persist whatever the menu just changed.

        The first version of the menu mutated the in-memory config and nothing
        else, so every setting silently reverted on restart — the same illusion
        v1's Settings page created, rebuilt by hand.
        """
        try:
            from backend.config import save

            save(self.config)
        except Exception:
            log.warning("could not save the config", exc_info=True)

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

    def _warm_up_audio(self) -> None:
        """Open and close a stream once, so the first real one is not slow.

        Measured 2026-08-10: the first MicSource of a process takes 713ms from
        start() to the first chunk of audio; every one after takes ~220ms. That
        difference is CoreAudio initialising, and it is charged to whatever the
        author says first — the opening of their first sentence, silently
        missing.

        ~220ms of clipping remains on every dictation. Fixing that needs the
        stream held permanently open with a rolling pre-roll buffer, which
        lights the microphone indicator for as long as the daemon runs. That is
        a privacy trade the author should make deliberately, not one to slip in.
        """
        shared = shared_with_output(self.config.input_device)
        if shared:
            # Opening this device would switch a Bluetooth headset into call
            # mode — degrading whatever the author is listening to, before they
            # have dictated a single word. Half a second off the first sentence
            # is not worth interrupting their music to buy.
            log.info("skipping audio warm-up: %s is also the output device", shared)
            return

        try:
            started = time.perf_counter()
            mic = self.make_mic()
            mic.start()
            time.sleep(0.15)
            list(mic.chunks())
            mic.stop()
            log.info("audio warmed in %.2fs", time.perf_counter() - started)
        except Exception:
            log.warning("audio warm-up failed", exc_info=True)

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
        locked = self.injector.lock_target()
        self._locked_at_press = getattr(locked, "name", None)
        log.info("press: locked target = %s", self._locked_at_press)

        if self.overlay is not None:
            self.overlay.show("recording")
            self._level_stop = threading.Event()
            threading.Thread(target=self._pump_levels, daemon=True,
                             name="utter-levels").start()

        try:
            self._mic = self.make_mic().start()
        except Exception as exc:
            # PortAudio prints its own wall of text to stderr before we ever see
            # this. Say the one thing the author can act on.
            log.warning("could not open the microphone: %s", exc)
            print(
                f"\n⚠ 打不开麦克风：{exc}\n"
                "  跑 `utter mics` 看哪个设备真的能录到声音。\n",
                flush=True,
            )
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
                dropped=getattr(mic, "dropped", 0),
            )
            self._index += 1
            self._idle.clear()
            self._queue.put(job)

        if self.overlay is not None:
            self._level_stop.set()
            self.overlay.set_state("working", "转录中")

        threading.Thread(target=mic.stop, daemon=True, name="utter-mic-close").start()

    def _pump_levels(self) -> None:
        """Drive the overlay at ~25fps while the key is held.

        Reads the level off a copy of the newest chunk rather than consuming the
        queue — the queue IS the recording, and taking chunks out of it here
        would silently shorten what gets transcribed.
        """
        from backend.overlay import rms_to_level

        started = time.perf_counter()
        while not self._level_stop.is_set():
            mic = self._mic
            chunk = getattr(mic, "last_chunk", None) if mic is not None else None
            self.overlay.feed(
                rms_to_level(chunk), f"{time.perf_counter() - started:.1f}s"
            )
            time.sleep(0.04)

    # -- the worker --

    def _drain(self) -> None:
        while not self._stop_flag.is_set():
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                self._idle.set()
                # Design §4.1e promised text buffered during a focus change
                # would land when the author came back. Injector.flush() existed
                # and was tested; nothing ever called it, so buffered text sat
                # there forever. This is that call.
                self._flush_if_back()
                continue
            try:
                self._process(job)
            except Exception:  # pragma: no cover - the worker must outlive a bad job
                log.warning("dictation %d failed", job.index, exc_info=True)
            finally:
                self._queue.task_done()
                if self._queue.empty():
                    self._idle.set()

    def _flush_if_back(self) -> None:
        """Deliver anything held, once the target window is frontmost again."""
        if getattr(self.injector, "pending", 0) < 1:
            return
        target = getattr(self.injector, "target", None)
        if target is None:
            return
        from backend.injection import _frontmost

        try:
            front = _frontmost()
            if front is not None and front.pid == target.pid:
                result = self.injector.flush()
                if result.injected:
                    log.info("flushed buffered dictation into %s", target.name)
        except Exception:  # pragma: no cover - the worker must survive this
            log.warning("could not flush buffered dictation", exc_info=True)

    def _process(self, job: _Job) -> None:
        watch = Stopwatch(
            target_ms=TARGET_MS_WITH_POLISH if self._polishing else TARGET_MS_WITHOUT_POLISH,
            dropped_chunks=job.dropped,
            audio_seconds=len(job.audio) / SAMPLE_RATE,
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
            if self.overlay is not None:
                self.overlay.set_state("error", "没听到")
                threading.Timer(1.2, self.overlay.hide).start()
            self.last_timing = watch
            return

        try:
            with watch.span("transcription"):
                raw = self._transcribe(job.audio, watch)
        except Exception:
            # One utterance lost, the session continues. Nothing has been shown
            # to the user yet, so there is nothing inconsistent to clean up.
            log.warning("transcription failed for utterance %d", job.index, exc_info=True)
            self.last_timing = watch
            return

        # Mechanical, and applied before anything else sees the text: only the
        # width of punctuation changes, never a word. Whisper does not hold one
        # punctuation style across a code-switch, so a bilingual sentence comes
        # back with Chinese commas after English clauses.
        raw, repeats = collapse_repetition(normalise_punctuation((raw or "").strip()))
        if self.config.close_sentences:
            raw = close_sentence(raw)
        if repeats:
            # The bounded temperature ladder is supposed to escape these, and on
            # 2026-08-10 it did not: 13.4s of speech came back as 「英文是，」
            # forty times. Injecting that into the author's document is worse
            # than any latency problem, so there is a deterministic net under
            # the model. Loud, never silent — the author must know the model
            # degenerated rather than believe they said this.
            log.warning("collapsed %d repetitions in utterance %d", repeats, job.index)
            watch.note_repetition(repeats)
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
            from backend.injection import _frontmost

            before = _frontmost()
            with watch.span("注入（写剪贴板→⌘V→还原）"):
                result = self.injector.inject(job.index, text)
            after = _frontmost()
            # Everything needed to tell where this went and why, in one line.
            # Reasoning about it from the outside has cost several rounds; the
            # daemon knows all four facts and was reporting none of them.
            watch.note_route(
                pressed=self._locked_at_press,
                target=getattr(getattr(self.injector, "target", None), "name", None),
                before=getattr(before, "name", None),
                after=getattr(after, "name", None),
                result=result,
            )
            # Whether the text landed, and where, was invisible until now — the
            # report showed a duration for an injection that may never have
            # happened. 铁律 8 is about not losing text silently; not saying
            # where it went is the same failure one step later.


        if self.on_text is not None:
            try:
                self.on_text(utterance)
            except Exception:  # pragma: no cover
                log.warning("on_text callback failed", exc_info=True)

        if self.overlay is not None:
            self.overlay.hide()

        self.last_timing = watch
        if watch.over_target:
            log.warning("dictation %d took %.0f ms", job.index, watch.total_ms)

    # -- helpers --

    @property
    def _polishing(self) -> bool:
        return bool(self.config.polish_enabled and self.polish is not None)

    def _transcribe(self, audio: np.ndarray, watch) -> str:
        """One pass for a short hold; clause by clause for a long one."""
        prompt = self._vocabulary_prompt()
        language = self.config.dictate_language

        if (
            SEGMENT_ABOVE_SECONDS is None
            or len(audio) / SAMPLE_RATE <= SEGMENT_ABOVE_SECONDS
        ):
            return self.stt.transcribe(audio, language=language, initial_prompt=prompt)

        pieces = self._split_at_pauses(audio)
        if len(pieces) < 2:
            # No breath long enough to cut at. Rather than hand Whisper a
            # minute of unbroken speech again, cut on a fixed interval — an
            # arbitrary boundary that produces punctuated clauses beats a
            # natural one that produces none.
            pieces = self._split_evenly(audio)
        if len(pieces) < 2:
            return self.stt.transcribe(audio, language=language, initial_prompt=prompt)

        watch.note_segments(len(pieces))
        out = []
        for piece in pieces:
            text = self.stt.transcribe(piece, language=language, initial_prompt=prompt)
            if text and text.strip():
                out.append(text.strip())
        from backend.scratchpad import join_text

        return join_text(out)

    def _split_evenly(self, audio: np.ndarray, seconds: float = 14.0) -> list[np.ndarray]:
        """Last resort for a speaker who never pauses long enough to cut at."""
        step = int(seconds * SAMPLE_RATE)
        if len(audio) <= step:
            return []
        return [audio[i : i + step] for i in range(0, len(audio), step)]

    def _split_at_pauses(self, audio: np.ndarray) -> list[np.ndarray]:
        """Cut a long hold where the speaker paused. Falls back to one piece."""
        try:
            if self._vad is None:
                self._vad = SileroVad()
            self._vad.reset()
            segmenter = VadSegmenter(
                vad_silence_ms=CLAUSE_SILENCE_MS,
                vad_sensitivity=self.config.vad_sensitivity,
                max_utterance_sec=self.config.max_utterance_sec,
                speech_prob=self._vad.speech_prob,
            )
            pieces = []
            step = 1600
            for i in range(0, len(audio), step):
                for event in segmenter.feed(audio[i : i + step]):
                    if isinstance(event, SpeechEnd) and len(event.audio):
                        pieces.append(event.audio)
            for event in segmenter.flush():
                if isinstance(event, SpeechEnd) and len(event.audio):
                    pieces.append(event.audio)
            return pieces
        except Exception:
            log.warning("could not split the utterance, transcribing whole", exc_info=True)
            return []

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

    #: Primes the decoder toward Simplified Chinese with ordinary punctuation.
    #: Whisper's Chinese output is unstable in two ways the author hit in real
    #: use — it sometimes emits Traditional characters, and it sometimes drops
    #: punctuation entirely. Both are decoding habits, and both respond to being
    #: shown an example. Measured 2026-08-10: adding this line recovered the
    #: full stop and question mark that were missing without it.
    ZH_PRIMER = "以下是简体中文的学术口述内容。"

    def _vocabulary_prompt(self) -> str | None:
        """The initial_prompt: design §4.1g layer 1, plus script priming.

        This is the cheapest correction there is — it costs no latency and it
        works with polish switched off. It is also the *right* layer for
        terminology: the author's "signs" came back as "science", and no
        downstream model could recover that, because nothing in "science" points
        back to the word actually spoken.
        """
        parts = []
        if (self.config.dictate_language or "").startswith("zh"):
            parts.append(self.ZH_PRIMER)
        if self.config.vocabulary:
            parts.append(", ".join(self.config.vocabulary))
        return " ".join(parts) if parts else None

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
