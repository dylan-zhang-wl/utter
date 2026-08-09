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

# 铁律 1, identical to the MLX provider: faster_whisper defaults `temperature`
# to [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] and the retry ladder destroys the latency
# budget on exactly the hard chunks where real time matters most.
TEMPERATURE = 0.0

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
            initial_prompt=initial_prompt or None,
            # Utter segments upstream (design §5.4). Letting faster-whisper run
            # its own VAD too would re-cut clauses the segmenter already decided.
            vad_filter=False,
        )
        return "".join(segment.text for segment in segments).strip()
