import json
import tempfile
from pathlib import Path
from backend.session import Session
from backend.transcriber import TranscriptSegment


def _make_segment(text, zh, start, end):
    return TranscriptSegment(text=text, start_time=start, end_time=end)


def test_session_creation():
    s = Session(audio_source="microphone")
    assert s.audio_source == "microphone"
    assert len(s.entries) == 0


def test_session_add_entry():
    s = Session(audio_source="microphone")
    s.add_entry("Hello world", "你好世界", 0.0, 1.5)
    assert len(s.entries) == 1
    assert s.entries[0]["text"] == "Hello world"
    assert s.entries[0]["translation"] == "你好世界"


def test_session_save_and_load(tmp_path):
    s = Session(audio_source="microphone", save_dir=str(tmp_path))
    s.add_entry("Hello", "你好", 0.0, 1.0)
    s.add_entry("World", "世界", 1.0, 2.0)
    filepath = s.save()

    assert Path(filepath).exists()
    with open(filepath) as f:
        data = json.load(f)
    assert len(data["entries"]) == 2


def test_export_txt(tmp_path):
    s = Session(audio_source="microphone", save_dir=str(tmp_path))
    s.add_entry("Hello", "你好", 0.0, 1.0)
    s.add_entry("World", "世界", 1.0, 2.0)
    content = s.export("txt")
    assert "Hello" in content
    assert "你好" in content


def test_export_srt(tmp_path):
    s = Session(audio_source="microphone", save_dir=str(tmp_path))
    s.add_entry("Hello", "你好", 0.0, 1.0)
    content = s.export("srt")
    assert "00:00:00,000" in content
    assert "Hello" in content


def test_export_markdown(tmp_path):
    s = Session(audio_source="microphone", save_dir=str(tmp_path))
    s.add_entry("Hello", "你好", 0.0, 1.0)
    content = s.export("markdown")
    assert "# " in content  # has a header
    assert "Hello" in content
