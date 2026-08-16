"""P3 task 1 — the meeting record.

The theme running through these: a meeting cannot be repeated, so every path
that could lose one is tested, including the two nobody plans for — forgetting
to press stop, and the process dying mid-sentence.
"""

import json

import pytest

from backend.listen import Entry, ListenSession, list_sessions


def _entry(i, source="So the question of equivalence", translation=None):
    return Entry(index=i, started_at=float(i), source=source,
                 display=source, translation=translation)


@pytest.fixture
def session(tmp_path):
    s = ListenSession(directory=tmp_path / "20260816-193000", title="研讨会")
    s.start()
    return s


# --- the record exists as it goes, not at the end -------------------------------


def test_an_entry_is_on_disk_the_moment_it_arrives(session):
    """The most likely way to lose a meeting is to forget to press stop, so
    nothing may wait for stop."""
    session.add(_entry(0))

    lines = session.entries_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["source"] == "So the question of equivalence"


def test_a_crash_costs_one_sentence_not_the_meeting(session):
    """A half-written final line is the shape a crash leaves behind."""
    session.add(_entry(0))
    session.add(_entry(1, source="Venuti argues that"))
    with session.entries_path.open("a", encoding="utf-8") as fh:
        fh.write('{"index": 2, "source": "cut off mid')

    recovered = ListenSession.load(session.directory)

    assert [e.index for e in recovered.entries] == [0, 1]
    assert recovered.entries[1].source == "Venuti argues that"


def test_entries_survive_a_reload_with_their_translations(session):
    session.add(_entry(0))
    session.set_translation(0, "那么对等这个问题")

    recovered = ListenSession.load(session.directory)
    assert recovered.entries[0].translation == "那么对等这个问题"
    assert recovered.entries[0].source == "So the question of equivalence"


# --- the author's requirement: two records, and pause writes them ---------------


def test_pause_writes_both_documents(session):
    """Asked for explicitly. Pause is when somebody steps out of the room and
    wants to see what has been captured; a record that only appears after 结束
    is no record at all during a two-hour meeting."""
    session.add(_entry(0))
    session.pause()

    assert session.verbatim_path.exists(), "逐字记录应该在暂停时就落盘"
    assert session.bilingual_path.exists(), "对照记录应该在暂停时就落盘"


def test_stop_writes_both_documents(session):
    session.add(_entry(0))
    session.stop()
    assert session.verbatim_path.exists() and session.bilingual_path.exists()


def test_the_verbatim_record_keeps_every_filler(session):
    """The whole point of keeping two. Whatever the screen chooses to hide,
    this file is where the speaker's actual words live."""
    session.add(Entry(index=0, started_at=0.0,
                      source="So um you know the question of, I mean, equivalence",
                      display="So the question of equivalence"))
    session.stop()

    verbatim = session.verbatim_path.read_text(encoding="utf-8")
    assert "um" in verbatim and "I mean" in verbatim
    assert "未做任何清理" in verbatim


def test_the_bilingual_record_is_the_tidied_one(session):
    session.add(Entry(index=0, started_at=0.0,
                      source="So um the question of equivalence",
                      display="So the question of equivalence",
                      translation="那么对等这个问题"))
    session.stop()

    bilingual = session.bilingual_path.read_text(encoding="utf-8")
    assert "So the question of equivalence" in bilingual
    assert "那么对等这个问题" in bilingual
    assert " um " not in bilingual


def test_an_untranslated_entry_says_so_rather_than_vanishing(session):
    """铁律 8's shape here: the model being unreachable must never cost the
    source line."""
    session.add(_entry(0, translation=None))
    session.stop()

    bilingual = session.bilingual_path.read_text(encoding="utf-8")
    assert "So the question of equivalence" in bilingual
    assert "未翻译" in bilingual


def test_documents_can_be_rebuilt_after_being_deleted(session):
    """They are derived from the jsonl, which is the real archive."""
    session.add(_entry(0, translation="那么对等这个问题"))
    session.stop()
    session.verbatim_path.unlink()

    ListenSession.load(session.directory).write_documents()
    assert "So the question of equivalence" in session.verbatim_path.read_text(
        encoding="utf-8")


# --- lifecycle ------------------------------------------------------------------


def test_a_transcription_arriving_after_stop_is_refused(session):
    """The pipeline is asynchronous, so a sentence can finish transcribing
    after the user has pressed stop. It must not append itself to a document
    the user considers finished."""
    session.add(_entry(0))
    session.stop()

    assert session.add(_entry(1)) is False
    assert len(session.entries) == 1


def test_a_transcription_arriving_during_a_pause_is_still_kept(session):
    """Different from stop: the audio was captured before the pause, and the
    meeting is not over. Dropping it would be silent loss."""
    session.add(_entry(0))
    session.pause()

    assert session.add(_entry(1)) is True
    assert len(session.entries) == 2


def test_resume_goes_back_to_recording(session):
    session.pause()
    session.resume()
    assert session.is_recording


# --- housekeeping ---------------------------------------------------------------


def test_the_session_directory_is_private(session):
    """It holds a recording of other people talking."""
    session.add(_entry(0))
    assert oct(session.entries_path.stat().st_mode)[-3:] == "600"


def test_an_unwritable_directory_does_not_stop_the_meeting(tmp_path, caplog):
    """铁律 8. The text is in memory; losing the file is worse than nothing but
    much better than aborting a lecture that is still being captured."""
    blocked = tmp_path / "nope"
    blocked.write_text("i am a file, not a directory")
    s = ListenSession(directory=blocked / "sess")
    s.start()

    assert s.add(_entry(0)) is True
    assert s.entries[0].source == "So the question of equivalence"
    s.stop()   # must not raise


def test_sessions_are_listed_newest_first(tmp_path):
    for stamp in ("20260814-090000", "20260816-193000", "20260815-140000"):
        (tmp_path / stamp).mkdir()
    assert [p.name for p in list_sessions(tmp_path)] == [
        "20260816-193000", "20260815-140000", "20260814-090000"]


def test_an_empty_session_writes_no_documents(session):
    """A meeting nobody spoke in should not leave a file that looks like a
    record of one."""
    session.stop()
    assert not session.verbatim_path.exists()


def test_create_puts_the_session_under_the_sandbox_not_the_real_home(tmp_path):
    """Lesson 39: DEFAULT_DIR is read at call time, never imported by value."""
    s = ListenSession.create()
    assert "pytest" in str(s.directory), s.directory
