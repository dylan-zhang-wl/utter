"""听记控制器 — what `utter listen` does, driven by a window instead of a terminal.

P3 task 4b. The pipeline, the session and the window all existed and worked
separately; this is the part that runs them together and owns the lifetime.

The audio pump is a background thread. Everything it touches — the session, the
translation queue — is off the main thread, and the only things it hands to the
window are two calls that marshal themselves. That is what keeps the interface
responsive while a meeting runs: the UI thread never waits for Whisper.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

TICK_SECONDS = 0.02


class ListenController:
    """One meeting, from 开始 to 结束."""

    def __init__(self, config, *, stt=None, translate=None, complete=None,
                 window=None, on_finished=None):
        self.config = config
        self._stt = stt
        self._translate = translate
        self._complete = complete
        self.window = window
        self.on_finished = on_finished

        self.session = None
        self.recording = None
        self._queue = None
        self._source = None
        self._thread = None
        self._stop = threading.Event()
        self._started_at = None
        self._activity = None
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def uses_microphone(self) -> bool:
        """Whether dictation would be fighting us for the input device."""
        from backend.system_audio import SystemAudioSource

        return self.running and not isinstance(self._source, SystemAudioSource)

    # -- lifecycle ---------------------------------------------------------

    def start(self, *, source: str = "mic", pids=None, title: str = "") -> bool:
        if self.running:
            return True
        self.error = None
        try:
            self._build(source=source, pids=pids, title=title)
        except Exception as exc:
            self.error = str(exc)
            log.warning("听记起不来：%s", exc, exc_info=True)
            self._teardown()
            return False

        self._stop.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="utter-listen")
        self._thread.start()
        if self.window is not None:
            show = getattr(self.window, "show_listen", None) or self.window.show
            show()
        return True

    def _build(self, *, source, pids, title) -> None:
        from backend.cli import build_complete, build_translate
        from backend.listen import ListenSession
        from backend.pipeline import listen_pipeline
        from backend.providers.stt import get_stt_provider
        from backend.session_audio import SessionRecording, sweep
        from backend.translator import TranslationQueue
        from backend.vad import VadSegmenter

        if self._stt is None:
            self._stt = get_stt_provider(preferred=self.config.stt_provider)
        if self._translate is None:
            self._translate = build_translate(self.config)
        if self._complete is None:
            self._complete = build_complete(self.config)

        sweep()
        self.session = ListenSession.create(title=title)
        self.session.start()
        self.recording = SessionRecording(self.session.directory)
        self.recording.start()

        self._queue = TranslationQueue(
            self._translate or (lambda text, context=None: None),
            self._on_translation)
        if self._translate is not None:
            self._queue.start()

        self._pipeline = listen_pipeline(
            stt=self._stt, sink=self._on_utterance,
            language=self.config.dictate_language or "en",
            vocabulary=list(self.config.vocabulary or []),
            translate=None)          # the queue does it, in batches
        self._segmenter = VadSegmenter(
            vad_silence_ms=self.config.vad_silence_ms,
            max_utterance_sec=self.config.max_utterance_sec)
        self._source = self._open_source(source, pids)
        self._source.start()
        self._hold_awake()

    def _open_source(self, source: str, pids):
        from backend.audio_source import MicSource

        if source == "system":
            from backend.system_audio import SystemAudioSource

            return SystemAudioSource(pids=list(pids or []))
        return MicSource(device_index=self.config.input_device)

    def _hold_awake(self) -> None:
        """Keep the machine working; let the screen go dark.

        `NSActivityUserInitiated` implies IdleSystemSleepDisabled and does not
        imply IdleDisplaySleepDisabled, which is exactly the pair the author
        asked for. It also blocks App Nap, which would otherwise throttle us
        the moment the window is not frontmost — which is the normal case
        during a meeting.
        """
        try:
            import Foundation

            self._activity = Foundation.NSProcessInfo.processInfo(
                ).beginActivityWithOptions_reason_(0x00FFFFFF | (1 << 20), "Utter 听记")
        except Exception:  # pragma: no cover - defensive
            log.warning("取不到活动断言，长会议可能被系统降频", exc_info=True)

    def _release_awake(self) -> None:
        if self._activity is None:
            return
        try:
            import Foundation

            Foundation.NSProcessInfo.processInfo().endActivity_(self._activity)
        except Exception:  # pragma: no cover
            pass
        self._activity = None

    # -- the pump ----------------------------------------------------------

    def _run(self) -> None:
        from backend.vad import SpeechEnd

        try:
            while not self._stop.is_set():
                got = False
                for chunk in self._source.chunks():
                    got = True
                    self.recording.write(chunk)
                    for event in self._segmenter.feed(chunk):
                        if isinstance(event, SpeechEnd):
                            self._pipeline.handle(event)
                if not got:
                    time.sleep(TICK_SECONDS)
                if getattr(self._source, "stopped_reason", None):
                    log.warning("音源停了：%s", self._source.stopped_reason)
                    break
                self._tick_clock()
        except Exception:
            log.warning("听记循环出错", exc_info=True)
        finally:
            for event in self._segmenter.flush():
                if isinstance(event, SpeechEnd):
                    self._pipeline.handle(event)

    def _tick_clock(self) -> None:
        if self.window is None or self._started_at is None:
            return
        seconds = int(time.monotonic() - self._started_at)
        if seconds != getattr(self, "_last_second", None):
            self._last_second = seconds
            self.window.set_clock(f"{seconds // 60:02d}:{seconds % 60:02d}")

    def _on_utterance(self, utterance) -> None:
        from backend.listen import Entry
        from backend.providers.llm import strip_fillers
        from backend.punctuation import is_hallucination

        if is_hallucination(utterance.raw_text):
            log.info("丢掉一条静音幻觉：%r", utterance.raw_text[:40])
            return
        entry = Entry(index=utterance.index, started_at=utterance.start_sec,
                      source=utterance.raw_text,
                      display=strip_fillers(utterance.raw_text),
                      forced=utterance.forced, confidence=utterance.confidence)
        self.session.add(entry)
        if self.window is not None:
            self.window.append(entry.index, entry.display)
        if self._translate is not None:
            self._queue.submit(entry.index, entry.source)

    def _on_translation(self, index: int, text: str) -> None:
        self.session.set_translation(index, text)
        if self.window is not None:
            self.window.translated(index, text)

    # -- ending ------------------------------------------------------------

    def pause(self) -> None:
        """Write both records without ending the meeting."""
        if self._queue is not None:
            self._queue.flush()
        if self.session is not None:
            self.session.pause()

    def stop(self, *, summarise: bool = True) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=30)
        self._teardown(summarise=summarise)
        if self.on_finished:
            try:
                self.on_finished(self.session)
            except Exception:  # pragma: no cover
                log.warning("结束回调出错", exc_info=True)

    def _teardown(self, *, summarise: bool = False) -> None:
        if self._source is not None:
            try:
                self._source.stop()
            except Exception:
                log.warning("关音源出错", exc_info=True)
            self._source = None
        if self._queue is not None:
            self._queue.stop(flush=True)
            self._queue = None
        if self.session is not None:
            self.session.stop()
        # The recording goes before the summary: whatever else happens, other
        # people's voices do not outlive the meeting (§3.2).
        if self.recording is not None:
            if not self.recording.discard():
                log.warning("会话录音没删掉：%s", self.recording.path)
            self.recording = None
        self._release_awake()

        if summarise and self.session is not None and self.session.entries:
            self._write_summary()

    def _write_summary(self) -> None:
        from backend.summary import summarise as make_summary

        if self._complete is None:
            log.info("没有可用的模型，跳过纪要")
            return
        try:
            make_summary(self.session, self._complete)
        except Exception:  # pragma: no cover - summarise already guards itself
            log.warning("纪要生成失败，记录不受影响", exc_info=True)
