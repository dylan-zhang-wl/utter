import numpy as np
from faster_whisper import WhisperModel
from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    text: str
    start_time: float
    end_time: float
    is_partial: bool = False


class Transcriber:
    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.model_size = model_size
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8",
        )

    def transcribe(self, audio: np.ndarray) -> list[TranscriptSegment]:
        segments_iter, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,
        )
        results = []
        for seg in segments_iter:
            results.append(TranscriptSegment(
                text=seg.text.strip(),
                start_time=seg.start,
                end_time=seg.end,
                is_partial=False,
            ))
        return results
