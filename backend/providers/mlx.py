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

# 铁律 1. mlx_whisper defaults `temperature` to (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
# a retry ladder that design §3 measured at 8.19s and 10.84s on a 3-second chunk
# against 0.96s with it pinned to a scalar zero. In a real-time pipeline, "slow
# by eight seconds" is far worse than "occasionally wrong by one clause".
TEMPERATURE = 0.0


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
            "language": language,
        }
        # An empty prompt is not the same as no prompt: "" can bias the decoder
        # toward continuing a sentence that does not exist.
        if initial_prompt:
            options["initial_prompt"] = initial_prompt

        result = mlx_whisper.transcribe(audio, **options)
        return result["text"].strip()
