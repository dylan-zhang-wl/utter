"""P3 task 5 — the meeting summary.

A summary is the one place in this project where a model is allowed to write
freely, because compressing and rephrasing is the request. So the tests are not
about wording. They are about the two things that make that safe: every claim
is traceable to a stretch of transcript, and nothing the summary does can touch
the transcript itself.
"""

import pytest

from backend.listen import Entry, ListenSession
from backend.summary import CHUNK_CHARS, plan_chunks, summarise


def _entry(i, source="讲者说了一段话，内容大致是这样的。"):
    return Entry(index=i, started_at=float(i * 30), source=source, display=source)


@pytest.fixture
def session(tmp_path):
    s = ListenSession(directory=tmp_path / "20260816-193000", title="组会")
    s.start()
    for i in range(4):
        s.add(_entry(i))
    s.stop()
    return s


# --- chunking -------------------------------------------------------------------


def test_a_short_meeting_is_one_chunk():
    assert len(plan_chunks([_entry(i) for i in range(3)])) == 1


def test_a_long_meeting_is_split():
    entries = [_entry(i, "x" * 1000) for i in range(20)]
    chunks = plan_chunks(entries, max_chars=6000)

    assert len(chunks) > 1
    assert sum(len(c) for c in chunks) == 20, "分块不能丢条目"


def test_an_entry_is_never_split_across_chunks():
    """Cutting somebody's sentence in half to fit a budget is how a summary
    ends up reporting something they did not say."""
    entries = [_entry(0, "x" * 9000), _entry(1, "y" * 100)]
    chunks = plan_chunks(entries, max_chars=6000)

    assert all(len(c) >= 1 for c in chunks)
    assert [e.index for c in chunks for e in c] == [0, 1]
    assert len(chunks[0]) == 1 and chunks[0][0].index == 0


def test_chunks_keep_the_meeting_in_order():
    entries = [_entry(i, "x" * 2000) for i in range(10)]
    flat = [e.index for c in plan_chunks(entries, max_chars=5000) for e in c]
    assert flat == list(range(10))


# --- the summary itself ---------------------------------------------------------


def test_a_summary_is_written_and_traceable(session):
    def complete(system, user):
        return "- 讨论了对等问题\n- 决定下周核对语料"

    text = summarise(session, complete)

    assert text is not None
    assert "讨论了对等问题" in text
    assert "条目 0–3" in text, "每一段都要能追回原文"
    assert (session.directory / "纪要.md").exists()


def test_the_summary_says_it_is_a_summary(session):
    """The reader has to know these are the model's words, not the speaker's."""
    text = summarise(session, lambda s, u: "- 要点")
    assert "模型的概括" in text
    assert "原文.md" in text


def test_the_summary_never_touches_the_transcript(session):
    """The transcript is the record; the summary is a convenience. If the model
    invents an agenda item, the transcript beside it still says otherwise."""
    before = session.verbatim_path.read_text(encoding="utf-8")

    summarise(session, lambda s, u: "- 模型编的东西")

    assert session.verbatim_path.read_text(encoding="utf-8") == before


def test_a_long_meeting_gets_a_second_pass(session, tmp_path):
    """Map then reduce: per-stretch points, then one document."""
    s = ListenSession(directory=tmp_path / "long", title="长会")
    s.start()
    for i in range(20):
        s.add(_entry(i, "x" * 1000))
    s.stop()

    calls = []

    def complete(system, user):
        calls.append(system)
        return "- 要点"

    summarise(s, complete, max_chars=6000)

    assert len(calls) > 2, "长会议应该先分段再合并"
    assert any("合成一份完整纪要" in c for c in calls), "少了合并那一步"


def test_a_single_chunk_skips_the_merge(session):
    """One stretch needs no second pass; asking for one is a wasted request and
    a second chance to drift."""
    calls = []

    def complete(system, user):
        calls.append(system)
        return "- 要点"

    summarise(session, complete)
    assert len(calls) == 1


# --- 铁律 8: the meeting is already safe by the time this runs -------------------


def test_a_model_failure_costs_the_summary_and_nothing_else(session):
    def boom(system, user):
        raise RuntimeError("no network")

    assert summarise(session, boom) is None
    assert session.verbatim_path.exists(), "记录必须完好"


def test_a_failed_merge_falls_back_to_the_per_stretch_points(tmp_path):
    """Half a summary beats none: the per-stretch points are already usable."""
    s = ListenSession(directory=tmp_path / "long", title="长会")
    s.start()
    for i in range(20):
        s.add(_entry(i, "x" * 1000))
    s.stop()

    def complete(system, user):
        if "合成一份完整纪要" in system:
            raise RuntimeError("merge failed")
        return "- 这一段的要点"

    text = summarise(s, complete, max_chars=6000)
    assert text is not None and "这一段的要点" in text


def test_an_empty_meeting_produces_no_summary(tmp_path):
    s = ListenSession(directory=tmp_path / "empty", title="空会")
    s.start()
    s.stop()
    assert summarise(s, lambda a, b: "- 要点") is None


def test_a_stretch_with_nothing_in_it_is_left_out(session):
    """Silence between agenda items should not become a bullet point."""
    assert summarise(session, lambda s, u: "（无实质内容）") is None


def test_the_prompts_forbid_inventing_agenda_items():
    """The failure mode for a meeting summary is plausible business filler —
    'agreed to follow up next week' when nobody said anything of the kind."""
    from backend.summary import _MAP_PROMPT, _REDUCE_PROMPT

    assert "只写记录里确实说过的" in _MAP_PROMPT
    assert "不要引入任何各段要点里没有的内容" in _REDUCE_PROMPT
    assert "不要把没有结论的讨论写成有结论" in _REDUCE_PROMPT
