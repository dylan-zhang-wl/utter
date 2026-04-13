import numpy as np
from unittest.mock import patch, MagicMock
from backend.transcriber import Transcriber, TranscriptSegment


def test_transcript_segment_model():
    seg = TranscriptSegment(
        text="Hello world",
        start_time=0.0,
        end_time=1.5,
        is_partial=False,
    )
    assert seg.text == "Hello world"
    assert seg.is_partial is False


def test_transcriber_init():
    with patch("backend.transcriber.WhisperModel") as mock:
        mock.return_value = MagicMock()
        t = Transcriber(model_size="base")
        assert t.model_size == "base"
        mock.assert_called_once()


def test_transcriber_process_audio():
    """Transcriber should accept audio and return segments."""
    with patch("backend.transcriber.WhisperModel") as mock_cls:
        mock_model = MagicMock()
        # Simulate whisper output
        mock_segment = MagicMock()
        mock_segment.text = " Hello there."
        mock_segment.start = 0.0
        mock_segment.end = 1.2
        mock_model.transcribe.return_value = ([mock_segment], MagicMock(language="en"))
        mock_cls.return_value = mock_model

        t = Transcriber(model_size="base")
        audio = np.random.randn(16000).astype(np.float32)  # 1 second
        segments = t.transcribe(audio)

        assert len(segments) == 1
        assert segments[0].text == "Hello there."
        assert segments[0].start_time == 0.0
