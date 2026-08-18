"""The grey tail: words on screen while the sentence is still being spoken.

Listen mode's committed transcript waits for the speaker to pause, because that
pause is the only boundary that can be trusted — it is where an utterance is
complete, where Whisper punctuates correctly, and where the audio buffer can be
cleared without losing a word. That is right, and it is also why the screen sat
blank for eight to twenty-two seconds at a time.

So this adds a second layer that is *not* trusted. A small model re-reads the
utterance in progress every second and the result is shown after the committed
text, greyed, with an ellipsis. It is never archived, never translated, never
counted as a sentence. When the speaker pauses, the big model transcribes the
whole utterance as it always did, the real sentences replace the preview, and
nothing about the record has changed.

The division of labour is the point. Two earlier attempts put the big model on
a fast tick and both failed: text from truncated audio got committed and the
model invented words at the cut (「Chrissy was weak」 for 「when democracy was
weak」), or a buffer got trimmed at a boundary nothing knew the audio offset of
and whole sentences vanished. Neither can happen here, because the fast layer
cannot write anything down.

Measured on this machine, 2026-08-18 (whisper-base-q4 vs large-v3-turbo):

    音频       小模型     大模型
     2s         83 ms     934 ms
     4s        108 ms     980 ms
    10s        166 ms    1109 ms
    20s        275 ms    1346 ms

At a one-second tick over a typical utterance that is 11% duty, against the 14%
the committed path already costs. A tick of the big model would have been 70%,
which is what made the earlier design unaffordable as well as wrong.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

#: Below this there is nothing worth reading, and Whisper on a fragment this
#: short mostly invents.
MIN_SECONDS = 0.6

#: Whisper's window. Past it the model chunks internally and the preview would
#: start disagreeing with itself; the committed path has its own ceiling well
#: below this anyway.
MAX_SECONDS = 28.0


#: Sentence-final punctuation. Text ending here was not cut off mid-word.
_ENDERS = ".!?。！？…\"'”’)）】」』"


def trim_partial_tail(text: str) -> str:
    """Drop the last word when the audio stopped in the middle of it.

    Measured against the small model on truncated buffers: the errors cluster
    at the very end, because that is where the audio runs out. 「just under a
    mile」 came back as 「just under a month」 one round before the audio
    reached 「mile」. The ellipsis already says that the end is unknown, so
    showing a guessed word there is worse than showing nothing — it reads as
    a mistranscription rather than as incompleteness.

    Chinese and Japanese are left alone: they are written without spaces, so
    there is no last *word* to drop — trimming by whitespace there would throw
    away the whole line.
    """
    text = (text or "").strip()
    if not text or text[-1] in _ENDERS or _has_cjk(text):
        return text
    words = text.split()
    return " ".join(words[:-1])


def _has_cjk(text: str) -> bool:
    return any("\u3040" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff"
               for ch in text)


class PreviewStream:
    """A speculative transcription of the utterance in progress.

    `audio()` is asked for the audio so far and may return None when nobody is
    speaking. `on_text` receives the preview; it is called with "" to clear.
    """

    def __init__(self, transcribe, audio, on_text, *, tick: float = 1.0,
                 language: str = "en"):
        self._transcribe = transcribe
        self._audio = audio
        self._on_text = on_text
        self._tick = tick
        self._language = language
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last = ""
        self._lock = threading.Lock()
        #: Bumped whenever a boundary lands. A round that started before the
        #: boundary must not repaint after it — it would show, as a guess about
        #: what comes next, text that was just committed above.
        self._generation = 0
        #: Set from the boundary until the committed sentences for it arrive.
        #: In that second or so the only audio available is the fragment after
        #: the cut, which reads as nonsense on its own — 「Reliable R」 — and
        #: guessing at it while the real text is seconds away is noise for
        #: nothing.
        self._held = False
        #: Wall-clock spent in the model, for the timing report.
        self.model_seconds = 0.0
        self.rounds = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="utter-preview")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        self._emit("")

    def _run(self) -> None:
        while not self._stop.wait(self._tick):
            try:
                self.once()
            except Exception:
                # A preview is a luxury. It must never be able to stop a
                # meeting from being recorded.
                log.debug("预览这一轮失败", exc_info=True)

    def hold(self) -> None:
        """Stop guessing: the committed layer is speaking for this stretch."""
        with self._lock:
            self._held = True

    def resume(self) -> None:
        """The committed sentences landed; the tail is ours again."""
        with self._lock:
            self._held = False

    def once(self) -> str:
        """One round. Returns what is now on screen ("" if nothing)."""
        with self._lock:
            if self._held:
                return ""
        clip = self._audio()
        if clip is None or len(clip) < MIN_SECONDS * SAMPLE_RATE:
            return self._emit("")
        if len(clip) > MAX_SECONDS * SAMPLE_RATE:
            clip = clip[-int(MAX_SECONDS * SAMPLE_RATE):]

        with self._lock:
            generation = self._generation

        started = time.monotonic()
        text = (self._transcribe(np.asarray(clip, dtype=np.float32),
                                 language=self._language) or "").strip()
        self.model_seconds += time.monotonic() - started
        self.rounds += 1

        with self._lock:
            if generation != self._generation:
                return ""      # a boundary landed while this was in the model
        return self._emit(trim_partial_tail(text))

    def clear(self) -> None:
        """The utterance closed; the committed sentences take over."""
        with self._lock:
            self._generation += 1
            self._held = True
        self._emit("")

    def _emit(self, text: str) -> str:
        if text == self._last:
            return text
        self._last = text
        try:
            self._on_text(text)
        except Exception:  # pragma: no cover - a UI callback misbehaving
            log.debug("预览回调出错", exc_info=True)
        return text
