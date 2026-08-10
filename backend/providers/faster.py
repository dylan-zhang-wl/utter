"""Whisper via CTranslate2 — the everywhere-else backend.

CPU (int8) and NVIDIA (float16). This is what runs on the author's non-Apple
machine and on a colleague's Windows laptop, and it is also the fallback on
Apple Silicon when MLX is missing or broken.

Building it in P1 rather than "later" is deliberate: a provider abstraction with
one implementation is not an abstraction, it is a wrapper that happens to
compile (design §5.1).
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

# int8 on CPU is v1's hard-won setting; float32 there is unusably slow.
_COMPUTE = {"cuda": ("cuda", "float16"), "cpu": ("cpu", "int8")}


class FasterWhisperProvider:
    id = "faster"
    display_name = "Whisper (CTranslate2, CPU/CUDA)"

    def __init__(self, tier: str = "balanced", hardware=None):
        self._tier = tier
        self._hardware = hardware
        self._model = None

    def _hw(self):
        return self._hardware or detect_hardware()

    def is_available(self) -> tuple[bool, str]:
        """Cheap: an import check only.

        `utter doctor` probes every provider, so this must never construct a
        model — that would trigger a multi-gigabyte download from a command
        whose whole job is to tell you what is wrong.
        """
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False, "faster-whisper is not installed"
        return True, ""

    def _device(self) -> tuple[str, str]:
        # Apple Silicon has no CUDA, so it falls to the CPU settings — MLX is
        # what wins there, and this provider is only the fallback.
        return _COMPUTE.get(self._hw().kind, _COMPUTE["cpu"])

    def _load(self):
        """Load once and keep it. Reconstructing per utterance would re-read
        weights from disk every clause, in a pipeline budgeted at one second."""
        if self._model is None:
            from faster_whisper import WhisperModel

            device, compute_type = self._device()
            self._model = WhisperModel(
                catalog.resolve(self._tier, self._hw()).repo,
                device=device,
                compute_type=compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> str:
        if audio is None or len(audio) == 0:
            return ""

        segments, _info = self._load().transcribe(
            audio,
            language=language,
            temperature=TEMPERATURE,
            condition_on_previous_text=CONDITION_ON_PREVIOUS,
            initial_prompt=initial_prompt or None,
            # Utter segments upstream (design §5.4). Letting faster-whisper run
            # its own VAD too would re-cut clauses the segmenter already decided.
            vad_filter=False,
        )
        return "".join(segment.text for segment in segments).strip()
