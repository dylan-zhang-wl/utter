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

Two gestures, two shapes. Push-to-talk sends the whole held buffer as one
utterance — the finger is the segmenter, and no VAD is involved. The toggle
streams: the microphone stays open and silero cuts it at the author's own
pauses, so each clause is transcribed and injected while the next is still
being spoken (P2b, design §4.1a).

Streaming is bounded by 铁律 9 — injected text is never revised — so there are
no partial hypotheses to correct. A clause is transcribed only after its pause
has arrived, and what lands in the document is final. Both shapes share the one
queue and the one worker, which is what keeps 铁律 11 true without any extra
machinery.
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
from backend.providers.llm import strip_fillers
from backend.punctuation import (
    close_sentence,
    collapse_repetition,
    is_hallucination,
    punctuate_pause,
)
from backend.punctuation import normalise as normalise_punctuation
from backend.scratchpad import Scratchpad, SessionArchive
from backend.timing import Stopwatch
from backend.vad import (
    SAMPLE_RATE,
    SileroVad,
    SpeechEnd,
    VadSegmenter,
    has_speech,
    speech_duration,
)

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

# The gate's threshold, expressed in seconds rather than frames. Three 32ms
# frames — `has_speech`'s own min_voiced_frames — so switching to a counting
# pass did not quietly change what counts as silence.
MIN_SPEECH_SECONDS = 3 * 512 / SAMPLE_RATE

# The fixed part of the budget: everything that does not depend on how long
# the author spoke — the speech check, the polish round trip, the paste.
TARGET_MS_WITHOUT_POLISH = 1500
TARGET_MS_WITH_POLISH = 3000

# And the part that does. Whisper turbo runs at 90–146ms per second of audio on
# this machine, so a fixed ceiling meant every hold over about fifteen seconds
# tripped the warning: 3857ms to transcribe 28.2 seconds is entirely normal and
# was being flagged as slow. A warning that fires on healthy runs is one the
# author learns to ignore, which is worse than not having it.
TARGET_MS_PER_AUDIO_SECOND = 150


@dataclass
class _Job:
    index: int
    audio: np.ndarray
    started_at: float
    """`perf_counter` at the moment the hotkey came up — the user's t=0."""
    dropped: int = 0
    overflows: int = 0
    held: float | None = None
    """How long the key was actually down. Compared against the length of the
    audio, it is the difference between "the model dropped what I said" and
    "what I said was never recorded"."""
    mic_open: float | None = None
    """Milliseconds from the key going down to the first frame of audio. This
    is the opening of the author's first word, and it is simply gone."""
    streamed: bool = False
    """One clause of a live session rather than a whole held utterance. Both
    the full stop and the polish call are wrong for a clause, for different
    reasons — see _process_inner."""
    gap_ms: float | None = None
    """Silence before this clause. The speaker's own answer to "did that
    sentence end", which is the question Whisper cannot answer about a
    fragment."""
    final: bool = False
    """The last clause of a streaming session, which always closes."""


@dataclass
class DictationDaemon:
    config: AppConfig
    stt: object
    injector: object | None = None
    make_mic: Callable[[], object] | None = None
    hotkey_factory: Callable[[Callable[[HotkeyEvent], None]], object] | None = None
    polish: Callable[..., str] | None = None
    polish_factory: Callable[[AppConfig], Callable[..., str] | None] | None = None
    """Rebuilds `polish` from config when the menu changes a setting.

    Without it the menu's 润色 toggle flips a boolean and nothing else: the
    callable was decided at startup, so switching polish on mid-session did
    exactly nothing. That is v1's Save button again, and it is worth a field to
    make impossible."""
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
    _pressed_at: float | None = field(init=False, default=None)
    _last_device: str | None = field(init=False, default=None)
    _stream_stop: threading.Event | None = field(init=False, default=None)
    _stream_vad_session: object | None = field(init=False, default=None)
    _streaming: bool = field(init=False, default=False)

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

        overgrown = self._check_vocabulary_fits()
        if overgrown:
            log.warning("vocabulary prompt exceeds Whisper's channel")
            print(f"\n{overgrown}\n", flush=True)

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

    def set_polish(self, enabled: bool, level: str | None = None) -> tuple[bool, str]:
        """Turn polish on or off, or change its level, without a restart.

        Returns (is it actually on now, what to tell the user). "Actually" is
        the load-bearing word: asking for polish with no API key stored leaves
        it off, and the menu must show off rather than a tick that lies.
        """
        self.config.polish_enabled = enabled
        if level is not None:
            self.config.polish_level = level
        self._save_config()

        if self.polish_factory is None:
            # Headless or under test. Honour the flag; there is nothing to build.
            return enabled, ""

        self.polish = self.polish_factory(self.config)
        if enabled and self.polish is None:
            self.config.polish_enabled = False
            self._save_config()
            return False, "没有可用的 LLM —— 先存 API key（utter doctor 里有说明）"
        return bool(self.polish), ""

    #: The only settings the menu can change, and therefore the only ones the
    #: daemon may write back. Everything else in config.json belongs to whoever
    #: edited it last.
    MENU_OWNED = (
        "stt_provider",
        "dictate_language",
        "dictate_target",
        "polish_enabled",
        "polish_level",
        "stream_while_speaking",
    )

    def _save_config(self) -> None:
        """Persist what the menu changed, without clobbering the rest.

        Two bugs, one line. The first: the menu mutated the in-memory config
        and nothing else, so every setting reverted on restart — v1's Settings
        page rebuilt by hand.

        The second, found 2026-08-10 while the daemon was running: saving wrote
        the *whole* in-memory config back, so an edit made to config.json in an
        editor was silently reverted the next time anyone touched the menu. Our
        own error messages tell the author to edit that file ("在
        ~/Utter/config.json 里填上项目 ID"), and then a background process undid
        it. Reload, apply only what the menu owns, write.
        """
        try:
            from backend.config import load_or_none, save

            on_disk = load_or_none()
            if on_disk is None:
                # The file is there and unreadable. Writing now would replace
                # it with defaults and take the vocabulary with it, which is
                # exactly how 30 terms were lost. A menu toggle is worth less
                # than the file.
                log.warning("配置文件读不了，这次不保存菜单的改动")
                return
            for field in self.MENU_OWNED:
                setattr(on_disk, field, getattr(self.config, field))
            # Keep memory and disk in step, so anything edited externally while
            # we ran is picked up rather than sitting there waiting to be lost.
            for field, value in on_disk.model_dump().items():
                if field not in self.MENU_OWNED:
                    setattr(self.config, field, value)
            save(on_disk)
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
            # Streaming belongs to the toggle gesture only. Push-to-talk means
            # "this phrase, now" and the finger is already the segmenter; the
            # toggle is the one used for a paragraph, which is where waiting
            # until the end is the thing that hurts.
            self.begin_utterance(at=event.at, stream=event.mode != "push")
        else:
            self.end_utterance(at=event.at)

    def begin_utterance(self, at: float | None = None, stream: bool | None = None) -> None:
        if self._mic is not None:
            return  # key repeat, or a second press before the first was released

        self._pressed_at = at if at is not None else time.perf_counter()
        # `stream` says whether the gesture supports it; the config says
        # whether the author wants it. Both have to agree.
        self._streaming = bool(stream) and bool(self.config.stream_while_speaking)

        # The microphone goes first, ahead of everything else this method does.
        #
        # It used to go last, after locking the target and raising the overlay.
        # Measured on this machine: asking AppKit which app is frontmost costs
        # 82ms and the panel costs more, and every one of those milliseconds is
        # speech the author had already started saying. CoreAudio's own ~240ms
        # is unavoidable without holding the stream open permanently; this part
        # was ours and was free to give back.
        #
        # Nothing below depends on the microphone, and the target lock is still
        # taken at press time — a quarter of a second later is still press time.
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

        # Lock the target now rather than at the end: by then the user may have
        # switched windows, and the text belongs where they started talking.
        locked = self.injector.lock_target()
        self._locked_at_press = getattr(locked, "name", None)
        log.info("press: locked target = %s", self._locked_at_press)

        if self.overlay is not None:
            self.overlay.show("recording")
            # Stop whatever pump is still running before starting another.
            # The event was being *replaced* rather than set, so the previous
            # thread went on to poll the new one — which was unset — and never
            # exited. One leaked thread per dictation, all drawing to the same
            # panel.
            self._level_stop.set()
            stop = self._level_stop = threading.Event()
            threading.Thread(target=self._pump_levels, args=(stop,), daemon=True,
                             name="utter-levels").start()

        if self._streaming and self._mic is not None:
            self._stream_stop = threading.Event()
            threading.Thread(
                target=self._stream_utterances, args=(self._stream_stop,),
                daemon=True, name="utter-stream",
            ).start()

    def end_utterance(self, at: float | None = None) -> None:
        mic, self._mic = self._mic, None
        if mic is None:
            return

        # Drain and queue BEFORE closing the stream. Measured on this machine:
        # draining takes 0.1ms, concatenating 0.1ms, and PortAudio's close takes
        # 130ms. Closing first put all of that on the user's critical path for
        # no reason — the audio is already in hand by then.
        released = at if at is not None else time.perf_counter()
        self._last_device = getattr(mic, "device_name", None)

        if self._streaming:
            # The streaming reader has been draining the microphone all along
            # and will flush the last clause on its way out. Draining here too
            # would race it and lose whatever it took.
            if self._stream_stop is not None:
                self._stream_stop.set()
            if self.overlay is not None:
                self._level_stop.set()
                self.overlay.set_state("working", "收尾中")
            # `_streaming` stays true until the reader has flushed and the
            # queue has drained, so _process does not start hiding the panel
            # out from under the clauses still in flight.
            def finish():
                self.wait_idle(timeout=30)
                self._streaming = False
                mic.stop()

            threading.Thread(target=finish, daemon=True, name="utter-mic-close").start()
            return

        chunks = list(mic.chunks())
        if chunks:
            self._enqueue(
                np.concatenate(chunks),
                started_at=released,
                held=(released - self._pressed_at) if self._pressed_at is not None else None,
                dropped=getattr(mic, "dropped", 0),
                overflows=getattr(mic, "overflows", 0),
                mic_open=(
                    (mic.first_chunk_at - self._pressed_at) * 1000
                    if getattr(mic, "first_chunk_at", None) and self._pressed_at is not None
                    else None
                ),
            )

        if self.overlay is not None:
            self._level_stop.set()
            self.overlay.set_state("working", "转录中")

        threading.Thread(target=mic.stop, daemon=True, name="utter-mic-close").start()

    # -- streaming (P2b) --

    def _stream_utterances(self, stop: threading.Event) -> None:
        """Cut the live microphone at pauses and send each clause on as it ends.

        This is 边说边出字, and the shape is forced by 铁律 9: text that has been
        injected is never revised. So there is no draft to correct later — a
        clause is transcribed only once its pause has arrived, and what lands in
        the document is final. What the author gets is not a live caption; it is
        their sentences appearing one behind the other, a beat late.

        The alternative shape — stream partial hypotheses and rewrite them —
        was ruled out on day one. It would mean the tool editing a document the
        author is also editing.

        Same queue and same single worker as push-to-talk, so 铁律 11 holds
        without extra machinery: clauses are transcribed, polished and injected
        strictly in the order they were spoken.

        The segmenter is P1's, unchanged. faster-whisper-dictation arrived at
        the same architecture independently — silero for boundaries, one whole
        utterance per model call — which is some comfort that it is the obvious
        answer rather than a clever one.
        """
        segmenter = VadSegmenter(
            # Deliberately NOT vad_silence_ms / max_utterance_sec: those answer
            # "is the utterance over", and measured on 47 seconds of real
            # dictation they produced two segments, both of them the 30-second
            # force-cut. See config.stream_silence_ms for the sweep.
            vad_silence_ms=self.config.stream_silence_ms,
            vad_sensitivity=self.config.vad_sensitivity,
            max_utterance_sec=self.config.stream_max_seconds,
            speech_prob=self._stream_vad().speech_prob,
        )
        started = time.perf_counter()

        previous_end = {"sample": None}

        def emit(event, final: bool = False) -> None:
            if not isinstance(event, SpeechEnd) or not len(event.audio):
                return
            gap = None
            if previous_end["sample"] is not None:
                gap = (event.start_sample - previous_end["sample"]) / SAMPLE_RATE * 1000
            previous_end["sample"] = event.end_sample
            self._enqueue(event.audio, started_at=time.perf_counter(), held=None,
                          streamed=True, gap_ms=gap, final=final)
            if self.overlay is not None:
                # A clause just left for the model; say so, then go back to
                # listening, because the microphone is still open.
                self.overlay.set_state("working", "出字中")

        while not stop.is_set():
            mic = self._mic
            if mic is None:
                break
            for chunk in mic.chunks():
                for event in segmenter.feed(chunk):
                    emit(event)
            time.sleep(0.05)

        for event in segmenter.flush():
            emit(event, final=True)

        # The reader owns the panel for the length of the session, so it is the
        # one that puts it away — after the worker has drained, or the last
        # clause would be injected into an empty screen.
        if self.overlay is not None:
            self.wait_idle(timeout=30)
            self.overlay.hide()
        log.info("streaming session ended after %.1fs", time.perf_counter() - started)

    def _stream_vad(self):
        """A silero session of its own.

        Not `self._vad`: that one is reset by the speech gate on the worker
        thread for every utterance, and resetting a VAD mid-sentence while it
        is segmenting a live stream loses the boundary it was in the middle of
        finding.
        """
        if self._stream_vad_session is None:
            self._stream_vad_session = SileroVad()
        self._stream_vad_session.reset()
        return self._stream_vad_session

    def _enqueue(self, audio: np.ndarray, *, started_at: float, held: float | None,
                 dropped: int = 0, overflows: int = 0, mic_open: float | None = None,
                 streamed: bool = False, gap_ms: float | None = None,
                 final: bool = False) -> None:
        job = _Job(
            index=self._index,
            audio=audio,
            started_at=started_at,
            dropped=dropped,
            overflows=overflows,
            held=held,
            mic_open=mic_open,
            streamed=streamed,
            gap_ms=gap_ms,
            final=final,
        )
        self._index += 1
        self._idle.clear()
        self._queue.put(job)

    def _pump_levels(self, stop: threading.Event) -> None:
        """Drive the overlay at ~25fps while the key is held.

        Reads the level off a copy of the newest chunk rather than consuming the
        queue — the queue IS the recording, and taking chunks out of it here
        would silently shorten what gets transcribed.
        """
        from backend.overlay import rms_to_level

        started = time.perf_counter()
        # The event is passed in, not read off self — see begin_utterance.
        while not stop.is_set():
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
        try:
            self._process_inner(job)
        finally:
            # Every exit path, not just the happy one. Transcription failure and
            # an empty transcript both returned early, leaving the panel stuck
            # on 「转录中」 with the bars frozen — which is worse than no overlay
            # at all, because it reports work that is not happening.
            #
            # Except while streaming, where this runs once per clause and the
            # microphone is still open: hiding here made the panel blink out
            # mid-sentence and come back, which reads as "it stopped listening".
            if self.overlay is not None and not self._streaming:
                self.overlay.hide()

    def _process_inner(self, job: _Job) -> None:
        audio_seconds = len(job.audio) / SAMPLE_RATE
        watch = Stopwatch(
            target_ms=(
                (TARGET_MS_WITH_POLISH if self._polishing else TARGET_MS_WITHOUT_POLISH)
                + TARGET_MS_PER_AUDIO_SECOND * audio_seconds
            ),
            dropped_chunks=job.dropped,
            overflows=job.overflows,
            audio_seconds=audio_seconds,
            held_seconds=job.held,
            mic_open_ms=job.mic_open,
        )
        # Published before the work, not after. `on_text` prints this report, and
        # `on_text` is called from inside this method — so assigning at the end
        # meant every report the author read belonged to the *previous*
        # sentence. Four rounds of analysis tonight were done on numbers paired
        # with the wrong words. The Stopwatch is mutable and fills in as it goes,
        # so an early reference is the whole fix.
        self.last_timing = watch
        watch.mark("hotkey → buffer closed", (time.perf_counter() - job.started_at) * 1000)

        # The silence gate. Push-to-talk has no VAD in its segmentation path, so
        # a hotkey pressed without speaking hands Whisper a buffer of room tone
        # — and Whisper does not answer nothing. Measured here: two seconds of
        # an empty room produced "you" and then "Good job.". Injecting a
        # fabricated sentence into the author's document is worse than any
        # latency problem in this project.
        with watch.span("speech check"):
            if self.speech_check is not None:
                speaking = self.speech_check(job.audio)
            else:
                # One pass, two answers. The gate needs a boolean and the report
                # needs the duration, and scanning twice for that would be silly.
                watch.speech_seconds = self._speech_seconds(job.audio)
                speaking = watch.speech_seconds >= MIN_SPEECH_SECONDS
        if not speaking:
            # Say what the audio actually looked like. "No speech" on its own
            # has now sent two separate investigations chasing the model when
            # the microphone was the problem: which device, how loud, how long.
            import numpy as _np

            peak = float(_np.abs(job.audio).max()) if len(job.audio) else 0.0
            log.info(
                "utterance %d contained no speech, discarded "
                "(设备=%s 时长=%.1fs 峰值=%.4f%s)",
                job.index,
                self._last_device or "?",
                len(job.audio) / SAMPLE_RATE,
                peak,
                "  ← 纯静音，麦克风没收到东西" if peak < 0.001 else "",
            )
            watch.skip("transcription", "no speech")
            if self.overlay is not None:
                self.overlay.set_state("error", "没听到")
                time.sleep(0.9)  # let the author see it before the finally hides it
            return

        try:
            with watch.span("transcription"):
                raw = self._transcribe(job.audio, watch)
        except Exception:
            # One utterance lost, the session continues. Nothing has been shown
            # to the user yet, so there is nothing inconsistent to clean up.
            log.warning("transcription failed for utterance %d", job.index, exc_info=True)
            return

        # Mechanical, and applied before anything else sees the text: only the
        # width of punctuation changes, never a word. Whisper does not hold one
        # punctuation style across a code-switch, so a bilingual sentence comes
        # back with Chinese commas after English clauses.
        raw, repeats = collapse_repetition(normalise_punctuation((raw or "").strip()))
        if job.streamed:
            # The speaker's pause decides, not the model. See punctuate_pause.
            raw = punctuate_pause(raw, job.gap_ms, final=job.final)
        elif self.config.close_sentences:
            # Not on a streamed clause. close_sentence exists because Whisper
            # leaves a held utterance open mid-breath, and one hold is one
            # thought — but in a live session each clause is its own utterance,
            # so 「而且」 came back as 「而且。」 and the author's paragraph read
            #
            #     但是这个延迟。好像。还是比较多的。
            #
            # The clauses are meant to join into a sentence, and Whisper's own
            # punctuation already knows where the sentence ends.
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
        if is_hallucination(raw):
            # Whisper filling a silence with a subtitle credit. Found in the
            # author's archive: 「字幕志愿者 李宗盛。」 injected into a document
            # after a near-silent recording. Loud, never quiet — they need to
            # know the model invented something rather than wonder where their
            # sentence went.
            log.warning("discarded a Whisper hallucination: %r", raw)
            watch.note_hallucination(raw)
            if self.overlay is not None:
                self.overlay.set_state("error", "没听清")
                time.sleep(0.9)
            return

        if not raw:
            return

        text, polished = raw, False
        if self._polishing and job.streamed:
            # Streaming and polish pull against each other, and streaming wins
            # while the microphone is open.
            #
            # Measured on the author's session: clauses of 0.7 to 2.8 seconds
            # were taking 3.4-4.1 seconds end to end, and most of that was one
            # LLM round trip per clause. Worse, a 0.7-second 「而且」 gives the
            # model nothing to work with — punctuation is a judgement about a
            # sentence, and a clause is not one.
            #
            # Filler removal still happens: that is ours, mechanical, and free.
            watch.skip("polish", "流式不润色")
            text = strip_fillers(raw) if self.config.polish_level != "light" else raw
        elif self._polishing:
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

    def _speech_seconds(self, audio: np.ndarray) -> float:
        """Seconds of talking in the buffer — the gate and the report in one pass.

        On failure it returns the whole buffer length, which reads as "all
        speech" and so lets the audio through. Same reasoning as `_has_speech`:
        a broken gate must not become a mute button.
        """
        try:
            if self._vad is None:
                self._vad = SileroVad()
            self._vad.reset()
            return speech_duration(
                audio,
                speech_prob=self._vad.speech_prob,
                sensitivity=self.config.vad_sensitivity,
            )
        except Exception:
            log.warning("speech check failed, transcribing anyway", exc_info=True)
            return len(audio) / SAMPLE_RATE

    #: Primes the decoder toward Simplified Chinese with ordinary punctuation.
    #: Whisper's Chinese output is unstable in two ways the author hit in real
    #: use — it sometimes emits Traditional characters, and it sometimes drops
    #: punctuation entirely. Both are decoding habits, and both respond to being
    #: shown an example. Measured 2026-08-10: adding this line recovered the
    #: full stop and question mark that were missing without it.
    ZH_PRIMER = "以下是简体中文的学术口述内容。"

    # Whisper's initial_prompt is capped at `n_text_ctx // 2 - 1` = 223 tokens,
    # and anything longer is **silently truncated from the front** — the oldest
    # terms in the list vanish and nothing anywhere says so.
    #
    # Measured on the author's own 30-term list: 340 characters = 133 tokens,
    # so about 2.5 characters per token for this Chinese/English mix. 500
    # characters leaves headroom under 223 tokens while still catching a list
    # that has quietly outgrown the channel. The author intends to keep adding
    # terminology, so the ceiling will be reached, and reaching it must not be
    # the kind of thing you find out from a transcript.
    VOCAB_PROMPT_LIMIT_CHARS = 500

    def _check_vocabulary_fits(self) -> str | None:
        """Warn if the terminology list has outgrown Whisper's prompt channel."""
        prompt = self._vocabulary_prompt()
        if not prompt or len(prompt) <= self.VOCAB_PROMPT_LIMIT_CHARS:
            return None
        return (
            f"⚠ 术语表太长了（{len(self.config.vocabulary)} 条、{len(prompt)} 字）。\n"
            f"  Whisper 的提示通道上限约 223 个 token（约 {self.VOCAB_PROMPT_LIMIT_CHARS} 字），"
            "超出的部分会被**从头截掉**，而且不会有任何提示。\n"
            "  在 ~/Utter/config.json 的 vocabulary 里删掉一些不常说的词，"
            "把名额留给最容易被听错的那几个。"
        )

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
