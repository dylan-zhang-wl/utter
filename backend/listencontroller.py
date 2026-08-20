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
        self._sentences = None
        self._next_index = 0
        #: Audio since the last boundary, for the preview to re-read. Held
        #: separately from the segmenter's own history so that nothing the
        #: preview does can reach dictation, which shares that class.
        self._heard = None
        self._heard_lock = threading.Lock()
        self._preview = None
        self.error: str | None = None
        #: Whether 结束 also writes 纪要.md. The author asked whether it was
        #: forced; it was, and it should not be — a summary is several model
        #: round trips and not every meeting wants one.
        self.summarise_at_end = True

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def uses_microphone(self) -> bool:
        """Whether dictation would be fighting us for the input device."""
        from backend.system_audio import SystemAudioSource

        return self.running and not isinstance(self._source, SystemAudioSource)

    # -- lifecycle ---------------------------------------------------------

    def start(self, *, source: str = "mic", pids=None, device=None,
              title: str = "") -> bool:
        if self.running:
            return True
        self.error = None
        try:
            self._build(source=source, pids=pids, device=device, title=title)
        except Exception as exc:
            self.error = str(exc)
            log.warning("听记起不来：%s", exc, exc_info=True)
            self._teardown()
            self._discard_empty_session()
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

    def _build(self, *, source, pids, device, title) -> None:
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
            self._on_translation,
            batch_size=max(1, int(self.config.translate_batch)),
            max_wait=float(self.config.translate_wait_seconds))
        if self._translate is not None:
            self._queue.start()

        self._pipeline = listen_pipeline(
            stt=self._stt, sink=self._on_utterance,
            language=self.config.listen_language or "en",
            vocabulary=list(self.config.vocabulary or []),
            translate=None)          # the queue does it, in batches
        self._segmenter = VadSegmenter(
            vad_silence_ms=self.config.listen_silence_ms,
            max_utterance_sec=self.config.listen_max_seconds,
            min_utterance_sec=self.config.listen_min_seconds)
        from backend.sentences import SentenceAssembler

        # Sentences are cut from the text, not from the silence. The VAD finds
        # *a* boundary cheaply; Whisper's punctuation says where the sentence
        # actually ended, and a chunk that stops mid-sentence hands its tail to
        # the next one.
        self._sentences = SentenceAssembler()
        self._next_index = 0
        self._reset_heard()
        self._preview = self._build_preview()
        self._source = self._open_source(source, pids, device)
        self._source.start()
        if self._preview is not None:
            self._preview.start()
        self._hold_awake()

    #: Audio kept for the preview to re-read. A hard ceiling rather than a
    #: guess: a VAD that never hears a pause must not grow this without bound.
    HEARD_CEILING_SECONDS = 30

    #: How much audio the preview keeps across a ceiling cut. A cut imposed at
    #: the ceiling lands mid-word, and a preview buffer that starts there reads
    #: the fragment as a word of its own — 「unalienable rights」 came back as
    #: 「I'm not a highly in-n-able rights」. A natural pause needs none of
    #: this: the speaker stopped, so the next buffer starts on a word.
    CARRY_SECONDS = 2.0

    def _reset_heard(self, *, carry: float = 0.0) -> None:
        import numpy as np

        with self._heard_lock:
            keep = int(carry * 16000)
            if keep and self._heard is not None and len(self._heard) > keep:
                self._heard = self._heard[-keep:].copy()
            else:
                self._heard = np.zeros(0, dtype="float32")

    def _remember_heard(self, chunk) -> None:
        import numpy as np

        with self._heard_lock:
            if self._heard is None:
                return
            self._heard = np.concatenate([self._heard, np.asarray(chunk, dtype="float32")])
            ceiling = self.HEARD_CEILING_SECONDS * 16000
            if len(self._heard) > ceiling:
                self._heard = self._heard[-ceiling:]

    def _heard_so_far(self):
        with self._heard_lock:
            return None if self._heard is None else self._heard.copy()

    def _build_preview(self):
        """The grey tail, or None if it is switched off or cannot be had.

        Failing to build it is not a reason to fail the meeting: the committed
        transcript does not depend on it in any way.
        """
        if not getattr(self.config, "listen_preview", False):
            return None
        try:
            from backend.preview import PreviewStream
            from backend.providers.stt import draft_provider

            draft = draft_provider(self.config.listen_preview_tier,
                                   preferred=self.config.stt_provider)
            if draft is None:
                return None
            return PreviewStream(
                draft.transcribe, self._heard_so_far, self._show_preview,
                tick=self.config.listen_preview_tick,
                language=self.config.listen_language or "en")
        except Exception:
            log.warning("预览层起不来，只是没有灰色的字，记录不受影响", exc_info=True)
            return None

    def _show_preview(self, text: str) -> None:
        if self.window is not None and hasattr(self.window, "preview"):
            self.window.preview(text)

    def _discard_empty_session(self) -> None:
        """A meeting that never started should not leave a folder behind.

        The session directory is created before the audio device is opened, so
        every failed 开始听记 left an empty dated folder in ~/Utter/listen.
        Four of them appeared in thirteen seconds on 2026-08-20 while the
        aggregate device was being retried by hand.
        """
        session, self.session = self.session, None
        directory = getattr(session, "directory", None)
        if directory is None:
            return
        try:
            if any(directory.iterdir()):
                return          # something was written; keep it
            directory.rmdir()
            log.info("没起来的听记，删掉空目录 %s", directory.name)
        except Exception:
            log.debug("删空目录失败", exc_info=True)

    def _open_source(self, source: str, pids, device=None):
        from backend.audio_source import MicSource

        if source == "system":
            from backend.system_audio import SystemAudioSource

            return SystemAudioSource(pids=list(pids or []))
        index = device if device is not None else self.config.input_device
        return MicSource(device_index=index)

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

    #: How long to wait before deciding that silence is a permission problem
    #: rather than a quiet room.
    SILENCE_CHECK_SECONDS = 6.0

    def _run(self) -> None:
        from backend.vad import SpeechEnd

        checked_silence = False
        try:
            while not self._stop.is_set():
                got = False
                for chunk in self._source.chunks():
                    got = True
                    if getattr(self, "_paused", False):
                        # Drained, not fed: the audio keeps flowing so the
                        # device stays open, but a coffee break does not become
                        # a paragraph of room noise.
                        continue
                    self.recording.write(chunk)
                    self._remember_heard(chunk)
                    for event in self._segmenter.feed(chunk):
                        if isinstance(event, SpeechEnd):
                            # The preview was speculating about this utterance;
                            # the committed sentences are about to say what it
                            # actually was, so drop the guess first.
                            self._reset_heard(
                                carry=self.CARRY_SECONDS if event.forced else 0.0)
                            if self._preview is not None:
                                self._preview.clear()
                            self._pipeline.handle(event)
                if not got:
                    time.sleep(TICK_SECONDS)
                if getattr(self._source, "stopped_reason", None):
                    log.warning("音源停了：%s", self._source.stopped_reason)
                    break
                self._tick_clock()
                if (not checked_silence and self._started_at is not None
                        and time.monotonic() - self._started_at > self.SILENCE_CHECK_SECONDS):
                    checked_silence = True
                    self._warn_if_silent()
        except Exception:
            log.warning("听记循环出错", exc_info=True)
        finally:
            for event in self._segmenter.flush():
                if isinstance(event, SpeechEnd):
                    self._pipeline.handle(event)
            self._flush_sentences()

    def _flush_sentences(self) -> None:
        """铁律 8: a speaker stopping mid-thought must not cost the thought."""
        if self._sentences is None or self._last_utterance is None:
            return
        for source in self._sentences.flush():
            self._emit(source, self._last_utterance)

    def _warn_if_silent(self) -> None:
        """Say it out loud rather than producing an empty transcript.

        A tap that was granted and then fed zeros looks exactly like a meeting
        nobody has started talking in. The difference matters: one resolves
        itself, the other never will, and the user finds out at the end of the
        lecture either way.
        """
        check = getattr(self._source, "silence_warning", None)
        if check is None:
            return
        why = check()
        if not why:
            return
        log.warning("听记收不到声音：%s", why.replace("\n", " "))
        self.error = why
        if self.window is not None and hasattr(self.window, "say"):
            self.window.say(
                "听记没有收到声音", why,
                settings_url="x-apple.systempreferences:"
                             "com.apple.preference.security?Privacy_AudioCapture")

    def _tick_clock(self) -> None:
        if self.window is None or self._started_at is None:
            return
        seconds = int(time.monotonic() - self._started_at)
        if seconds != getattr(self, "_last_second", None):
            self._last_second = seconds
            self.window.set_clock(f"{seconds // 60:02d}:{seconds % 60:02d}")

    _last_utterance = None

    def _on_utterance(self, utterance) -> None:
        self._last_utterance = utterance
        from backend.listen import Entry
        from backend.providers.llm import strip_fillers
        from backend.punctuation import collapse_repetition, is_hallucination

        if is_hallucination(utterance.raw_text):
            log.info("丢掉一条静音幻觉：%r", utterance.raw_text[:40])
            if self._preview is not None:
                self._preview.resume()
            return

        # 铁律 1's failure mode, and it reached a real transcript: one entry
        # carried 「既的」 repeated for 136 characters, and the translator
        # faithfully rendered the whole loop into Chinese. Dictation has folded
        # these since P2a; the listen path simply never called it.
        #
        # Confidence does not catch this — that entry scored -0.147, right in
        # the middle of the healthy range — so the guard has to be structural.
        text, removed = collapse_repetition(utterance.raw_text)
        if removed:
            log.warning("折叠了 %d 次复读：%d 字 → %d 字",
                        removed, len(utterance.raw_text), len(text))

        for source in self._sentences.feed(text, cut=utterance.forced):
            self._emit(source, utterance)
        # This chunk has had its say; the tail can start guessing again.
        if self._preview is not None:
            self._preview.resume()

    def _emit(self, source: str, utterance) -> None:
        """One finished sentence into the record and onto the screen."""
        from backend.listen import Entry
        from backend.providers.llm import strip_fillers
        entry = Entry(index=self._next_index, started_at=utterance.start_sec,
                      source=source, display=strip_fillers(source),
                      forced=utterance.forced, confidence=utterance.confidence)
        self._next_index += 1
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

    def pause(self) -> bool:
        """Pause or resume. Returns whether it is now paused.

        Pausing writes both records — the author asked for that specifically —
        and stops feeding the segmenter, so a break in the meeting does not
        become thirty seconds of room noise in the transcript.
        """
        self._paused = not getattr(self, "_paused", False)
        if self._preview is not None:
            # Audio is drained but not fed while paused, so there is nothing
            # for the preview to read — and a coffee break should not keep a
            # model warm on the GPU.
            self._preview.clear() if self._paused else None
        if self._paused:
            self._reset_heard()
            if self._queue is not None:
                self._queue.flush()
            if self.session is not None:
                self.session.pause()
        elif self.session is not None:
            self.session.resume()
        return self._paused

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
        if self._preview is not None:
            self._preview.stop()
            if self._preview.rounds:
                log.info("预览：%d 轮，模型共 %.1fs",
                         self._preview.rounds, self._preview.model_seconds)
            self._preview = None
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

        if (summarise and self.summarise_at_end
                and self.session is not None and self.session.entries):
            self._write_summary()

    def _write_summary(self) -> None:
        from backend.summary import summarise as make_summary

        if self._complete is None:
            log.info("没有可用的模型，跳过纪要")
            self._tell("会议记录已保存", "没有可用的模型，所以没有生成纪要。")
            return
        self._tell_status("正在生成纪要…")
        try:
            written = make_summary(self.session, self._complete)
        except Exception:  # pragma: no cover - summarise already guards itself
            log.warning("纪要生成失败，记录不受影响", exc_info=True)
            written = None
        self._tell_status(None)
        self._tell(
            "会议记录已保存" if written else "会议记录已保存（纪要没生成）",
            f"{len(self.session.entries)} 条\n\n"
            f"{self.session.directory}\n\n"
            "原文.md 一字不少，对照.md 是中英对照"
            + ("，纪要.md 是会后整理。" if written else "。"))

    def _tell(self, title: str, body: str) -> None:
        if self.window is not None and hasattr(self.window, "say"):
            self.window.say(title, body)

    def _tell_status(self, text) -> None:
        if self.window is not None and hasattr(self.window, "set_status_line"):
            self.window.set_status_line(text)
