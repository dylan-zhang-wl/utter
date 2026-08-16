"""P3 task 4 — the listen window.

Appearance is checked by eye. What is tested here is the one piece of real
logic: a translation arrives after its English was already drawn, so it has to
be spliced into the middle of the text and every entry after it shifts. Get
that wrong and Chinese lands under the wrong English — silently, because both
lines still look like sentences.
"""

import pytest

AppKit = pytest.importorskip("AppKit")

from backend.listenwindow import TranscriptPane  # noqa: E402


@pytest.fixture
def window():
    """The 听记 page as a view. It lives inside the main window now — the
    application does two things, and starting the second from a right-click on
    the menu bar was not where anyone would look for it."""
    AppKit.NSApplication.sharedApplication()
    pane = TranscriptPane()
    assert pane.view() is not None, "页面没建起来"
    return pane


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


def test_long_lines_wrap_instead_of_running_off_the_edge():
    """Measured inside a real window, because that is the only place the pane
    has a width. The first build laid out against a 1080pt container inside a
    560pt window and every long sentence ran off the right edge."""
    import types

    from backend.config import AppConfig
    from backend.window import MainWindow

    AppKit.NSApplication.sharedApplication()
    daemon = types.SimpleNamespace(
        config=AppConfig(), stt=None, running=True, polish=None,
        scratchpad=types.SimpleNamespace(
            archive=types.SimpleNamespace(path="/tmp/x")))
    daemon._save_config = lambda: None

    main = MainWindow(daemon)
    assert main._build()
    main.append(0, "Venuti argues that fluency is itself an ideology, "
                   "one that makes the translator invisible entirely.")
    main._window.contentView().layoutSubtreeIfNeeded()

    text = main.listen_pane._text
    manager, container = text.layoutManager(), text.textContainer()
    # usedRect is lazy: without forcing layout it reports 0x0 and the test
    # passes for the wrong reason.
    manager.ensureLayoutForTextContainer_(container)
    used = manager.usedRectForTextContainer_(container)
    clip = text.enclosingScrollView().contentView().frame().size.width

    assert clip > 100, "窗口里应该有真实宽度"
    assert used.size.width <= clip, f"文字宽 {used.size.width} 超过可视宽 {clip}"
    assert used.size.height > 30, "没换行的话高度会很小"


def test_the_window_has_both_pages_and_starts_on_listening():
    """The application does two things; the window should say so."""
    import types

    from backend.config import AppConfig
    from backend.window import MainWindow

    AppKit.NSApplication.sharedApplication()
    daemon = types.SimpleNamespace(
        config=AppConfig(), stt=None, running=True, polish=None,
        scratchpad=types.SimpleNamespace(
            archive=types.SimpleNamespace(path="/tmp/x")))
    daemon._save_config = lambda: None

    main = MainWindow(daemon)
    assert main._build()

    assert set(main._pages) == {"listen", "settings"}
    assert not main._pages["listen"].isHidden(), "默认应该停在听记页"
    assert main._pages["settings"].isHidden()

    main._select_page("settings")
    assert main._pages["listen"].isHidden()
    assert not main._pages["settings"].isHidden()
