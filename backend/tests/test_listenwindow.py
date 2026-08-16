"""P3 task 4 — the listen window.

Appearance is checked by eye. What is tested here is the one piece of real
logic: a translation arrives after its English was already drawn, so it has to
be spliced into the middle of the text and every entry after it shifts. Get
that wrong and Chinese lands under the wrong English — silently, because both
lines still look like sentences.
"""

import pytest

AppKit = pytest.importorskip("AppKit")

from backend.listenwindow import ListenWindow  # noqa: E402


@pytest.fixture
def window():
    AppKit.NSApplication.sharedApplication()
    w = ListenWindow()
    assert w._build(), "窗口没建起来"
    return w


def body(window) -> str:
    return str(window._text.textStorage().string())


def test_an_entry_shows_english_and_a_placeholder(window):
    window.append(0, "So the question of equivalence")

    assert "So the question of equivalence" in body(window)
    assert "…" in body(window), "还没译出来时要有占位"


def test_a_translation_replaces_its_own_placeholder(window):
    window.append(0, "So the question of equivalence")
    window.translated(0, "那么对等这个问题")

    text = body(window)
    assert "那么对等这个问题" in text
    assert "…" not in text


def test_a_late_translation_lands_under_its_own_english(window):
    """The failure this whole mechanism exists to prevent."""
    window.append(0, "First sentence")
    window.append(1, "Second sentence")
    window.append(2, "Third sentence")

    window.translated(1, "第二句的译文")

    text = body(window)
    after_second = text.index("Second sentence")
    third = text.index("Third sentence")
    assert after_second < text.index("第二句的译文") < third, "译文挂到别的句子下面了"


def test_translations_arriving_out_of_order_still_line_up(window):
    """Batches return whenever they return; the queue does not promise order
    across batches."""
    for i, english in enumerate(["Alpha one", "Beta two", "Gamma three"]):
        window.append(i, english)

    window.translated(2, "丙")
    window.translated(0, "甲")
    window.translated(1, "乙")

    text = body(window)
    assert text.index("Alpha one") < text.index("甲") < text.index("Beta two")
    assert text.index("Beta two") < text.index("乙") < text.index("Gamma three")
    assert text.index("Gamma three") < text.index("丙")


def test_a_long_translation_does_not_corrupt_later_entries(window):
    """The replacement is longer than the placeholder, so everything after it
    shifts. If the bookkeeping is wrong this is where it shows."""
    for i in range(4):
        window.append(i, f"Sentence number {i}")

    window.translated(0, "一段很长的译文" * 12)
    window.translated(3, "最后一句")

    text = body(window)
    assert text.index("Sentence number 3") < text.index("最后一句")
    for i in range(4):
        assert f"Sentence number {i}" in text


def test_a_translation_for_an_unknown_entry_is_ignored(window):
    """A stray callback after the session moved on must not write anywhere."""
    window.append(0, "Only entry")
    before = body(window)

    window.translated(99, "野译文")

    assert body(window) == before


def test_an_entry_that_already_has_a_translation_shows_it_immediately(window):
    window.append(0, "English here", "中文在此")

    text = body(window)
    assert "中文在此" in text and "…" not in text


def test_the_transcript_is_selectable_but_not_editable(window):
    """The author asked to copy lines mid-meeting; nobody should be able to
    type into the record."""
    assert window._text.isSelectable()
    assert not window._text.isEditable()


def test_long_lines_wrap_instead_of_running_off_the_edge(window):
    """Measured, because the first build did not: the text view came out twice
    the width of its clip view and every long sentence ran off the right."""
    window.append(0, "Venuti argues that fluency is itself an ideology, "
                     "one that makes the translator invisible entirely.")
    window._window.contentView().layoutSubtreeIfNeeded()

    # usedRect is lazy: without forcing layout it reports 0x0 and the test
    # passes for the wrong reason.
    manager = window._text.layoutManager()
    container = window._text.textContainer()
    manager.ensureLayoutForTextContainer_(container)
    used = manager.usedRectForTextContainer_(container)
    clip = window._text.enclosingScrollView().contentView().frame().size.width

    assert used.size.width <= clip, f"文字宽 {used.size.width} 超过了可视宽 {clip}"
    assert used.size.height > 30, "没换行的话高度会很小"
