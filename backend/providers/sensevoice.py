"""SenseVoice — the candidate for Chinese, and for Chinese with English in it.

Why this exists. Whisper carries one language token per 30-second window, so a
sentence that switches language mid-way gets one of its halves translated
rather than transcribed (openai/whisper#976, and measured here repeatedly). Its
Chinese punctuation is also sparse at every audio length, because its Chinese
training data is largely subtitles, which carry none. Neither is reachable by
configuration; both live in the weights.

SenseVoice was trained by a Chinese lab on Mandarin, Cantonese, English,
Japanese and Korean, emits punctuation of its own, and is non-autoregressive —
which is why published comparisons put it many times faster than Whisper-Large
despite being smaller.

## Not the official runtime

`funasr`, the reference implementation, declares 71 dependencies including
torch. 铁律 5 keeps torch out of this project — the whole VAD was built on raw
ONNX weights to avoid dragging in 476M for a 1.2M model, and importing it here
would undo that.

sherpa-onnx has one dependency and no torch, and the int8 ONNX weights are
228 MB against Whisper turbo's 1.6 GB. Seven times smaller, for the model that
is supposed to be *better* at this author's actual language.

## Nothing above this file changes

This is the return on P1's provider layer. The pipeline, the daemon, the
injector, the hotkey and the overlay never learn which engine ran — they were
written against `SttProvider`, and this is one.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from backend.hardware import detect as detect_hardware

log = logging.getLogger(__name__)

HF_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
WEIGHTS = "model.int8.onnx"
TOKENS = "tokens.txt"

# SenseVoice returns rich-transcription tags alongside the words — language,
# emotion, audio events, and the ITN marker. They are metadata, not speech, and
# the author dictates into documents.
_TAGS = re.compile(r"<\|[^|]*\|>")


class SenseVoiceProvider:
    id = "sensevoice"
    display_name = "SenseVoice (CJK, ONNX)"

    def __init__(self, hardware=None, use_itn: bool = True, threads: int = 4):
        self._hardware = hardware
        # Inverse text normalisation is where SenseVoice's punctuation comes
        # from, which is the entire reason this provider is a candidate.
        self._use_itn = use_itn
        self._threads = threads
        self._recogniser = None

    def is_available(self) -> tuple[bool, str]:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            return False, "sherpa-onnx is not installed"
        return True, ""

    def _paths(self) -> tuple[str, str]:
        from huggingface_hub import hf_hub_download

        return (
            hf_hub_download(HF_REPO, WEIGHTS),
            hf_hub_download(HF_REPO, TOKENS),
        )

    def _load(self):
        """Build the recogniser once. Reconstructing it per utterance would
        re-read 228 MB from disk inside a one-second budget."""
        if self._recogniser is None:
            import sherpa_onnx

            model, tokens = self._paths()
            self._recogniser = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model,
                tokens=tokens,
                num_threads=self._threads,
                use_itn=self._use_itn,
                debug=False,
            )
        return self._recogniser

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> str:
        if audio is None or len(audio) == 0:
            return ""

        # `language` and `initial_prompt` are accepted and ignored, deliberately.
        #
        # SenseVoice identifies the language itself and has no prompt channel,
        # so there is nowhere to put a vocabulary list — which means design
        # §4.1g layer 1, the free terminology fix that turned "Thorinization"
        # back into "foreignisation", does not exist here. That is a real cost
        # of switching and belongs in the comparison, not in a silent shrug.
        recogniser = self._load()
        stream = recogniser.create_stream()
        stream.accept_waveform(16000, np.asarray(audio, dtype=np.float32))
        recogniser.decode_stream(stream)

        return _TAGS.sub("", stream.result.text).strip()

    @property
    def supports_prompt(self) -> bool:
        """False. The registry and the comparison both need to know."""
        return False
