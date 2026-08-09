"""The five-slot pipeline. Both modes, one shape.

    AudioSource -> Segmenter -> SttProvider -> PostProcessor -> Sink

Design §4. Listening and dictating differ in what fills each slot, never in the
slots themselves — that is what makes Utter one platform rather than two
programs sharing a repository.

This module knows about no concrete backend. Not mlx_whisper, not
faster_whisper, not any HTTP client. Everything arrives as a callable or an
object with the right shape, which is what keeps 铁律 6 true at the layer most
tempted to break it.

Error handling follows 铁律 8 throughout: a failure anywhere downstream of
transcription costs at most the enrichment, never the text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

import numpy as np

from backend.vad import SAMPLE_RATE, SpeechEnd

log = logging.getLogger(__name__)

Mode = str  # "listen" | "dictate"
MODES = ("listen", "dictate")


class Stt(Protocol):
    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class Utterance:
    """One finished unit of speech, with everything known about it."""

    index: int
    mode: Mode
    text: str
    raw_text: str
    """What the model actually heard, before any polish.

    Always kept. 铁律 10 — this is the only way the author can later notice that
    a polish quietly changed an argument.
    """
    start_sec: float
    end_sec: float
    translation: str | None = None
    polished: bool = False
    forced: bool = False


@dataclass
class Pipeline:
    stt: Stt
    sink: Callable[[Utterance], None]
    mode: Mode = "listen"
    language: str | None = None
    vocabulary: list[str] = field(default_factory=list)
    translate: Callable[..., str] | None = None
    polish: Callable[..., str] | None = None

    _index: int = 0
    _previous_text: str | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of {MODES}")

    # -- the vocabulary prompt (design §4.1g layer 1) --

    def _initial_prompt(self) -> str | None:
        return ", ".join(self.vocabulary) if self.vocabulary else None

    # -- one utterance through the whole chain --

    def handle(self, segment: SpeechEnd) -> Utterance | None:
        if segment.audio is None or len(segment.audio) == 0:
            return None

        try:
            raw = self.stt.transcribe(
                segment.audio,
                language=self.language,
                initial_prompt=self._initial_prompt(),
            )
        except Exception:
            # One bad segment must not end the session. The speaker is still
            # talking and the next utterance is a second away.
            log.warning("transcription failed, skipping this utterance", exc_info=True)
            return None

        raw = (raw or "").strip()
        if not raw:
            # Whisper returns empty or whitespace for a segment that turned out
            # to be a cough, a door, or a microphone bump.
            return None

        text, polished = raw, False
        translation = None

        if self.mode == "dictate" and self.polish is not None:
            text, polished = self._apply(self.polish, raw, default=raw)
        elif self.mode == "listen" and self.translate is not None:
            translation, _ = self._apply(self.translate, raw, default=None)

        utterance = Utterance(
            index=self._index,
            mode=self.mode,
            text=text,
            raw_text=raw,
            start_sec=segment.start_sample / SAMPLE_RATE,
            end_sec=segment.end_sample / SAMPLE_RATE,
            translation=translation,
            polished=polished,
            forced=segment.forced,
        )
        self._index += 1
        self._previous_text = raw

        self._emit(utterance)
        return utterance

    def _apply(self, fn, raw: str, default):
        """Run an enrichment step, or give up on it quietly.

        铁律 8: the transcript is already safe by this point, so a failure here
        costs the translation or the polish and nothing else.
        """
        try:
            result = fn(raw, context=self._previous_text)
        except Exception:
            log.warning("post-processing failed, keeping the raw transcript", exc_info=True)
            return default, False
        return (result, True) if result else (default, False)

    def _emit(self, utterance: Utterance) -> None:
        try:
            self.sink(utterance)
        except Exception:
            # A blocked injection target or a full disk must not stop
            # transcription — the next utterance may well succeed, and the
            # session archive is a separate sink.
            log.warning("sink failed for utterance %d", utterance.index, exc_info=True)

    # -- driving from an audio stream --

    def run_stream(self, chunks: Iterable[np.ndarray], segmenter) -> None:
        """Feed audio through a segmenter and handle every utterance it yields."""
        for chunk in chunks:
            for event in segmenter.feed(chunk):
                if isinstance(event, SpeechEnd):
                    self.handle(event)

        # Whatever is still open — end of file, or the hotkey coming up — is
        # still something the user said.
        for event in segmenter.flush():
            if isinstance(event, SpeechEnd):
                self.handle(event)


def listen_pipeline(**kwargs) -> Pipeline:
    """Lecture mode: transcribe, translate, show and archive."""
    kwargs.setdefault("language", "en")
    return Pipeline(mode="listen", **kwargs)


def dictate_pipeline(**kwargs) -> Pipeline:
    """Dictation mode: transcribe, optionally polish, inject at the cursor.

    Polish defaults to off, per design §8's P2a decision.
    """
    return Pipeline(mode="dictate", **kwargs)
