"""P2a Task 4 — scratchpad mode and the session archive.

Design §4.1f: dictation bound to no window. Text accumulates, the author decides
where it goes. Cheapest useful thing in P2a — it needs no injection at all — and
plausibly the one that gets used most, since dictating notes while reading a PDF
and pasting them somewhere later is a real workflow.

The archive tests carry 铁律 8. Entries are written as they arrive, so a crash
nine minutes into a ten-minute dictation costs the last sentence and nothing
else.
"""

import json
import stat

import pytest

from backend.pipeline import Utterance
from backend.scratchpad import Scratchpad, SessionArchive, join_text


def utt(index, text, raw=None, mode="dictate"):
    return Utterance(
        index=index,
        mode=mode,
        text=text,
        raw_text=raw if raw is not None else text,
        start_sec=index * 2.0,
        end_sec=index * 2.0 + 1.5,
    )


# --- joining -----------------------------------------------------------------


def test_english_sentences_join_with_a_space():
    assert join_text(["Translation is rewriting.", "Every choice leaves a trace."]) == (
        "Translation is rewriting. Every choice leaves a trace."
    )


def test_chinese_sentences_join_without_a_space():
    """A space between Chinese sentences is wrong typography and the author
    writes in both languages."""
    assert join_text(["翻译不是复制。", "而是一种重写。"]) == "翻译不是复制。而是一种重写。"


def test_mixed_script_joins_sensibly():
    assert join_text(["他引用了 Venuti。", "异化是他的主张。"]) == (
        "他引用了 Venuti。异化是他的主张。"
    )


def test_join_of_one_is_itself():
    assert join_text(["only this"]) == "only this"


def test_join_of_nothing_is_empty():
    assert join_text([]) == ""


def test_join_skips_blanks():
    assert join_text(["first", "", "  ", "second"]) == "first second"


# --- the scratchpad ----------------------------------------------------------


def test_entries_accumulate_in_order():
    pad = Scratchpad()
    pad.add(utt(0, "first"))
    pad.add(utt(1, "second"))

    assert [e.text for e in pad.entries] == ["first", "second"]


def test_peek_returns_the_joined_text():
    pad = Scratchpad()
    pad.add(utt(0, "Translation is rewriting."))
    pad.add(utt(1, "Every choice leaves a trace."))

    assert pad.peek() == "Translation is rewriting. Every choice leaves a trace."


def test_peek_does_not_empty():
    pad = Scratchpad()
    pad.add(utt(0, "text"))
    pad.peek()

    assert len(pad) == 1


def test_take_returns_and_empties():
    pad = Scratchpad()
    pad.add(utt(0, "text"))

    assert pad.take() == "text"
    assert len(pad) == 0


def test_take_of_an_empty_pad_is_empty_string():
    assert Scratchpad().take() == ""


def test_raw_text_is_retained_beside_the_processed_text():
    """铁律 10's audit trail survives into the scratchpad."""
    pad = Scratchpad()
    pad.add(utt(0, "polished version", raw="raw version"))

    assert pad.entries[0].raw_text == "raw version"
    assert pad.entries[0].text == "polished version"


def test_take_uses_the_processed_text():
    pad = Scratchpad()
    pad.add(utt(0, "polished version", raw="raw version"))

    assert pad.take() == "polished version"


def test_raw_text_is_separately_retrievable():
    """For a diff against the polished version when something reads wrong."""
    pad = Scratchpad()
    pad.add(utt(0, "polished", raw="raw"))

    assert pad.peek(raw=True) == "raw"


def test_clear_empties_without_returning():
    pad = Scratchpad()
    pad.add(utt(0, "text"))
    pad.clear()

    assert len(pad) == 0


# --- the archive -------------------------------------------------------------


def test_archive_writes_under_the_sessions_directory(tmp_path):
    archive = SessionArchive(base_dir=tmp_path, session_id="20260810_010203")
    archive.append(utt(0, "text"))

    assert archive.path.parent == tmp_path / "sessions"
    assert archive.path.exists()


def test_archive_file_is_0600(tmp_path):
    """It holds everything the author dictated, which may be an unpublished
    draft or a confidential translation."""
    archive = SessionArchive(base_dir=tmp_path, session_id="s")
    archive.append(utt(0, "text"))

    assert stat.S_IMODE(archive.path.stat().st_mode) == 0o600


def test_archive_appends_as_entries_arrive(tmp_path):
    """铁律 8. A crash nine minutes into a ten-minute dictation must cost the
    last sentence, not the whole session."""
    archive = SessionArchive(base_dir=tmp_path, session_id="s")
    archive.append(utt(0, "first"))

    assert "first" in archive.path.read_text(), "must be on disk before the next one"

    archive.append(utt(1, "second"))
    assert len(archive.path.read_text().strip().splitlines()) == 2


def test_archive_is_line_delimited(tmp_path):
    """One JSON object per line rather than a rewritten array: appending is
    O(1) and a crash mid-write costs one truncated line, not the file."""
    archive = SessionArchive(base_dir=tmp_path, session_id="s")
    archive.append(utt(0, "first"))
    archive.append(utt(1, "second"))

    lines = archive.path.read_text().strip().splitlines()
    assert [json.loads(line)["text"] for line in lines] == ["first", "second"]


def test_archive_keeps_raw_and_processed(tmp_path):
    archive = SessionArchive(base_dir=tmp_path, session_id="s")
    archive.append(utt(0, "polished", raw="raw"))

    record = json.loads(archive.path.read_text().strip())
    assert record["text"] == "polished"
    assert record["raw_text"] == "raw"


def test_archive_records_timing_and_mode(tmp_path):
    archive = SessionArchive(base_dir=tmp_path, session_id="s")
    archive.append(utt(3, "text"))

    record = json.loads(archive.path.read_text().strip())
    assert record["index"] == 3
    assert record["mode"] == "dictate"
    assert record["start_sec"] == pytest.approx(6.0)


def test_archive_survives_unicode(tmp_path):
    archive = SessionArchive(base_dir=tmp_path, session_id="s")
    archive.append(utt(0, "翻译不是复制，而是一种重写。"))

    record = json.loads(archive.path.read_text().strip())
    assert record["text"] == "翻译不是复制，而是一种重写。"


def test_archive_failure_does_not_raise(tmp_path):
    """铁律 8 again, from the other side: a full disk must not abort the
    dictation that is still usable in memory."""
    archive = SessionArchive(base_dir=tmp_path / "file-not-dir", session_id="s")
    (tmp_path / "file-not-dir").write_text("I am a file, not a directory")

    archive.append(utt(0, "text"))  # must not raise


def test_scratchpad_writes_through_to_the_archive(tmp_path):
    archive = SessionArchive(base_dir=tmp_path, session_id="s")
    pad = Scratchpad(archive=archive)
    pad.add(utt(0, "text"))

    assert "text" in archive.path.read_text()


def test_taking_does_not_erase_the_archive(tmp_path):
    """The scratchpad empties; the record does not. That is the point of it."""
    archive = SessionArchive(base_dir=tmp_path, session_id="s")
    pad = Scratchpad(archive=archive)
    pad.add(utt(0, "text"))
    pad.take()

    assert "text" in archive.path.read_text()


def test_session_id_defaults_to_something_sortable(tmp_path):
    archive = SessionArchive(base_dir=tmp_path)
    assert archive.path.name.startswith("session_")
    assert archive.path.suffix == ".jsonl"
