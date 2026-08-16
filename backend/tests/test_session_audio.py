"""铁律 3's only exception, and the tests that keep it an exception.

The recording exists so a garbled line in a lecture can be checked against
what was actually said. It holds other people's voices, so the tests that
matter most are the ones about it being gone afterwards — including when the
process died before it could tidy up.
"""

import numpy as np
import pytest

from backend.session_audio import AUDIO_FILE, SessionRecording, sweep


@pytest.fixture
def recording(tmp_path):
    return SessionRecording(tmp_path / "20260816-193000")


def _tone(seconds=1.0):
    t = np.linspace(0, seconds, int(16000 * seconds), dtype="float32")
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype("float32")


def test_audio_is_written_while_the_session_runs(recording):
    recording.start()
    recording.write(_tone(1.0))

    assert recording.path.exists()
    assert 0.9 < recording.seconds < 1.1


def test_the_file_is_a_readable_wav(recording):
    """A recording nothing can open is not a recording."""
    import soundfile as sf

    recording.start()
    recording.write(_tone(0.5))
    recording._close()

    data, rate = sf.read(recording.path, dtype="float32")
    assert rate == 16000
    assert 0.45 < len(data) / rate < 0.55


def test_stopping_deletes_it(recording):
    """The whole point. It holds other people talking; it does not outlive the
    meeting."""
    recording.start()
    recording.write(_tone(0.2))

    assert recording.discard() is True
    assert not recording.path.exists()


def test_discarding_twice_is_fine(recording):
    """The postcondition is 'no recording on disk', not 'a deletion happened'."""
    recording.start()
    recording.write(_tone(0.1))
    recording.discard()
    assert recording.discard() is True


def test_discarding_a_session_that_never_recorded_is_fine(recording):
    assert recording.discard() is True


def test_a_recording_left_by_a_crash_is_swept_on_the_next_launch(tmp_path):
    """A crash mid-meeting must not leave other people's conversation sitting
    in the home directory indefinitely — the same outcome the stop-time
    deletion prevents, arriving by another route."""
    listen_root = tmp_path / "listen"
    for name in ("20260815-100000", "20260816-193000"):
        rec = SessionRecording(listen_root / name)
        rec.start()
        rec.write(_tone(0.1))
        rec._close()

    assert sweep(listen_root) == 2
    assert not any(p.name == AUDIO_FILE for p in listen_root.rglob("*"))


def test_the_sweep_leaves_the_transcript_alone(tmp_path):
    """It deletes recordings. Only recordings."""
    listen_root = tmp_path / "listen"
    session = listen_root / "20260816-193000"
    session.mkdir(parents=True)
    (session / "entries.jsonl").write_text("{}", encoding="utf-8")
    (session / "原文.md").write_text("# 记录", encoding="utf-8")
    rec = SessionRecording(session)
    rec.start()
    rec.write(_tone(0.1))
    rec._close()

    sweep(listen_root)

    assert (session / "entries.jsonl").exists()
    assert (session / "原文.md").exists()
    assert not (session / AUDIO_FILE).exists()


def test_the_recording_is_private(recording):
    recording.start()
    recording.write(_tone(0.1))
    assert oct(recording.path.stat().st_mode)[-3:] == "600"


def test_a_forgotten_session_stops_growing(tmp_path):
    """Four hours is a session somebody forgot to stop, not a meeting."""
    rec = SessionRecording(tmp_path / "s", max_hours=0.0002)   # ~2.3 seconds
    rec.start()
    for _ in range(6):
        rec.write(_tone(1.0))

    assert rec.seconds < 4, "到了上限还在写"
    assert rec.path.exists(), "上限只停录音，不该删掉已录的"


def test_an_unwritable_directory_does_not_stop_the_meeting(tmp_path):
    """铁律 8: the transcript is written by a different path and must survive
    the recording failing entirely."""
    blocked = tmp_path / "nope"
    blocked.write_text("i am a file")
    rec = SessionRecording(blocked / "s")

    rec.start()          # must not raise
    rec.write(_tone(0.1))
    assert rec.discard() is True


def test_writing_before_start_is_ignored_rather_than_crashing(recording):
    recording.write(_tone(0.1))
    assert not recording.path.exists()
