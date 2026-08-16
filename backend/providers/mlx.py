"""Whisper on Apple Silicon via MLX (Metal GPU).

The fastest path on the author's machine: design §3 measured ~1.0s per utterance
for large-v3-turbo here, which is the number the whole real-time design rests on.

`mlx_whisper` is imported inside methods, never at module scope. The registry has
to be able to load this module on a machine with no MLX in order to report it as
unavailable — an ImportError at import time would take the registry down with it.
"""

from __future__ import annotations

import numpy as np

from backend import catalog
from backend.hardware import detect as detect_hardware

# 铁律 1, amended 2026-08-10 after a repetition loop in real use.
#
# The full default ladder (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) was measured at 8.19s
# and 10.84s on a 3-second chunk, so P1 pinned a scalar 0.0. What that missed:
# the ladder's *purpose* is escaping repetition. Whisper checks
# compression_ratio after each pass and retries hotter when the output is
# degenerate — with a scalar there is nothing to retry with, so a loop runs to
# the end of the window. The author got "of the model sign" fifty times.
#
# Two steps, not six. The second only runs when the first produces something
# degenerate, so the normal case costs exactly what it did before and the worst
# case is ~2.2s rather than 8-11s. The rule's intent was always to bound the
# worst case, not to forbid retrying.
TEMPERATURE = (0.0, 0.2)

# Each window is conditioned on the previous window's text by default, which is
# how a repetition that starts in one window feeds itself into the next. Off.
CONDITION_ON_PREVIOUS = False


class MlxWhisperProvider:
    id = "mlx"
    display_name = "Whisper (MLX, Apple Silicon GPU)"

    def __init__(self, tier: str = "balanced", hardware=None):
        self._tier = tier
        self._hardware = hardware

    def _hw(self):
        return self._hardware or detect_hardware()

    def is_available(self) -> tuple[bool, str]:
        if self._hw().kind != "apple_silicon":
            return False, "requires Apple Silicon — MLX has no Metal backend elsewhere"
        try:
            import mlx_whisper  # noqa: F401
        except ImportError:
            return False, "mlx-whisper is not installed"
        return True, ""

    def _repo(self) -> str:
        return catalog.resolve(self._tier, self._hw()).repo

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> str:
        if audio is None or len(audio) == 0:
            # VAD can hand over an empty buffer at a segment boundary. Calling
            # the model costs a second and returns nothing.
            return ""

        import mlx_whisper

        options = {
            "path_or_hf_repo": self._repo(),
            "temperature": TEMPERATURE,
            "condition_on_previous_text": CONDITION_ON_PREVIOUS,
            "language": language,
        }
        # An empty prompt is not the same as no prompt: "" can bias the decoder
        # toward continuing a sentence that does not exist.
        if initial_prompt:
            options["initial_prompt"] = initial_prompt

        result = mlx_whisper.transcribe(audio, **options)
        return result["text"].strip()

    def transcribe_detailed(self, audio, language=None, initial_prompt=None):
        """The same call, keeping the numbers `transcribe()` discards.

        Whisper reports `avg_logprob` and `no_speech_prob` per segment and this
        provider was dropping both. Listen mode wants them: a meeting cannot be
        repeated, so knowing which lines the model was unsure about is what
        tells the reader where to spend their attention.
        """
        from backend.providers.stt import Detailed

        if audio is None or len(audio) == 0:
            return Detailed(text="")

        import mlx_whisper

        options = {
            "path_or_hf_repo": self._repo(),
            "temperature": TEMPERATURE,
            "condition_on_previous_text": CONDITION_ON_PREVIOUS,
            "language": language,
        }
        if initial_prompt:
            options["initial_prompt"] = initial_prompt

        result = mlx_whisper.transcribe(audio, **options)
        segments = result.get("segments") or []
        return Detailed(
            text=(result.get("text") or "").strip(),
            confidence=_weighted(segments, "avg_logprob"),
            no_speech=_weighted(segments, "no_speech_prob"),
        )


def _weighted(segments, key):
    """Duration-weighted mean over an utterance's segments.

    Weighted rather than plain: a two-second aside and a twenty-second
    sentence are not equally informative about how well the model heard the
    utterance as a whole.
    """
    total = 0.0
    weight = 0.0
    for segment in segments:
        value = segment.get(key)
        if value is None:
            continue
        span = max(float(segment.get("end", 0)) - float(segment.get("start", 0)), 0.01)
        total += float(value) * span
        weight += span
    return total / weight if weight else None
