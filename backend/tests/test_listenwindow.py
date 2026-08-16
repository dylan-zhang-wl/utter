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


# --- the start button, which did nothing at all in the first build ---------------


def _main_window():
    import types

    from backend.config import AppConfig
    from backend.window import MainWindow

    AppKit.NSApplication.sharedApplication()
    daemon = types.SimpleNamespace(
        config=AppConfig(), stt=None, running=True, polish=None,
        scratchpad=types.SimpleNamespace(
            archive=types.SimpleNamespace(path="/tmp/x")))
    daemon._save_config = lambda: None
    return MainWindow(daemon)


def test_the_start_button_actually_fires():
    """It did not. The action was registered as "toggle_" — the Python method
    name — where Objective-C wants the selector "toggle:", so the button was
    wired to something that did not exist and clicking it raised
    "unrecognized selector" into the void. Everything else in this codebase
    writes the colon; this one place did not.
    """
    window = _main_window()
    fired = []
    window.on_listen_toggle = lambda: fired.append(1)
    assert window._build()

    window.listen_pane._button.performClick_(None)

    assert fired == [1], "开始按钮没有触发任何东西"


def test_the_button_says_it_is_working_rather_than_looking_dead():
    """Loading the model and opening a device takes seconds. A button that
    still reads 开始听记 through all of it looks like a button that ignored
    the click — which is how the author described it."""
    window = _main_window()
    assert window._build()
    pane = window.listen_pane

    pane.set_preparing(True)
    assert "准备" in str(pane._button.title())
    assert not pane._button.isEnabled()

    pane.set_preparing(False)
    pane.set_running(True)
    assert "结束" in str(pane._button.title())
    assert pane._button.isEnabled()


def test_the_source_list_is_real_devices_not_invented_categories():
    """The author's point: 「线下会议」/「线上会议」 is a category they have to
    translate into a device. List what macOS actually reports, and nothing
    that is not there."""
    window = _main_window()
    assert window._build()
    pane = window.listen_pane

    titles = [str(pane._source.itemTitleAtIndex_(i))
              for i in range(pane._source.numberOfItems())]

    assert titles, "一个音源都没列出来"
    assert not any("线下会议" in t or "线上会议" in t for t in titles)
    kind, _device = pane.source
    assert kind in ("mic", "system")


def test_every_listed_source_maps_to_something_real():
    """No entry may exist that cannot be started."""
    window = _main_window()
    assert window._build()
    pane = window.listen_pane

    assert len(pane._choices) == pane._source.numberOfItems() or not pane._choices
    for kind, device in pane._choices:
        assert kind in ("mic", "system")
        assert kind == "system" or isinstance(device, int)


def test_the_window_forwards_everything_the_controller_calls():
    """The button fired, then died on window.set_preparing — added to the pane
    and never forwarded from the window — and AppKit swallowed the
    AttributeError, so the control looked inert with an empty log. The
    forwarding surface is small enough to just assert."""
    window = _main_window()
    assert window._build()

    for name in ("append", "translated", "set_clock", "set_listening",
                 "set_preparing", "show_listen"):
        assert callable(getattr(window, name, None)), f"窗口少了 {name}"

    # and they must not raise when called, which is what the controller does
    window.set_preparing(True)
    window.set_preparing(False)
    window.set_listening(True)
    window.set_clock("01:23")
    window.append(0, "hello", "你好")
    window.translated(0, "你好啊")


def test_recording_shrinks_the_window_to_a_bookmark_and_restores_it():
    """The author watches the transcript beside something else, so the
    full-width settings shape is wrong for the only moment it is read."""
    window = _main_window()
    assert window._build()
    before = window._window.frame()

    window.set_listening(True)
    small = window._window.frame()
    assert small.size.width < before.size.width, "开始后应该收窄成书签"
    assert window._tabs.isHidden(), "录制时只有一页，不该还显示切换器"

    window.set_listening(False)
    after = window._window.frame()
    assert abs(after.size.width - before.size.width) < 1, "结束后应该复原"
    assert not window._tabs.isHidden()


def test_pause_is_offered_only_while_recording():
    window = _main_window()
    assert window._build()
    pane = window.listen_pane

    pane._set_running(False)
    assert pane._pause.isHidden(), "没在录的时候不该有暂停"

    pane._set_running(True)
    assert not pane._pause.isHidden()
    assert str(pane._pause.title()) == "暂停"


def test_pause_toggles_and_reports_state():
    """Pausing writes both records and stops feeding the segmenter, so a coffee
    break does not become a paragraph of room noise."""
    from backend.config import AppConfig
    from backend.listencontroller import ListenController

    controller = ListenController(AppConfig())
    assert controller.pause() is True
    assert controller.pause() is False


# --- the things that made it feel unfinished ------------------------------------


def test_the_transcript_lays_out_only_what_is_visible():
    """TextKit re-flows the whole document on a width change by default, and a
    meeting's transcript only grows. Measured before this: 11-21ms per resize
    at 150 entries, rising with length; after: 5-10ms and flat to 400."""
    window = _main_window()
    assert window._build()
    assert window.listen_pane._text.layoutManager().allowsNonContiguousLayout()


def test_resizing_stays_cheap_as_the_meeting_grows():
    import time

    window = _main_window()
    assert window._build()
    pane = window.listen_pane
    for i in range(300):
        pane.append(i, "So the question of equivalence is not about words at all.",
                    "因此，等值问题其实根本不是一个关于词语的问题。")

    worst = 0.0
    for width in (420, 380, 460, 400):
        frame = window._window.frame()
        frame.size.width = width
        start = time.perf_counter()
        window._window.setFrame_display_(frame, True)
        window._window.contentView().layoutSubtreeIfNeeded()
        worst = max(worst, (time.perf_counter() - start) * 1000)

    assert worst < 40, f"300 条时改窗口大小要 {worst:.0f}ms，会看出卡顿"


def test_the_status_line_says_what_is_happening():
    """Ending a meeting runs several model round trips. Saying nothing through
    them reads as the application having ignored the click."""
    window = _main_window()
    assert window._build()

    window.set_status_line("正在生成纪要…")
    assert "纪要" in str(window.listen_pane._status.stringValue())

    window.set_status_line(None)
    assert str(window.listen_pane._status.stringValue()) == ""


def test_translation_waits_for_at_most_one_more_sentence():
    """Two constraints pulling opposite ways, and the setting has to satisfy
    both. A batch of three was 15-36 seconds of nothing once entries became
    whole sentences; a batch of one arrived fast and read like unrelated
    fragments, because the model never saw two sentences together. Two is the
    pair that coheres at the cost of one sentence of lag."""
    from backend.config import AppConfig

    config = AppConfig()
    assert config.translate_batch == 2
    assert config.translate_wait_seconds <= 3.0
