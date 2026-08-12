"""The floating indicator's arithmetic. No AppKit — that half is smoke-tested.

The overlay exists to answer one question at a glance: is the microphone
hearing me? Every assertion below defends that, not the appearance.
"""

import numpy as np

from backend.overlay import BarModel, rms_to_level


def settle(model, level, frames=30):
    for _ in range(frames):
        model.feed(level)
    return model.heights()


def test_silence_reads_as_a_flat_line():
    """The whole point. If the author speaks and this stays flat, the microphone
    is not working — and they learn it now rather than from an empty result."""
    heights = settle(BarModel(), 0.0)
    assert max(heights) < 0.1


def test_speech_lifts_the_row():
    assert max(settle(BarModel(), 0.8)) > 0.6


def test_louder_is_taller():
    quiet = max(settle(BarModel(), 0.3))
    loud = max(settle(BarModel(), 0.9))
    assert loud > quiet * 1.5


def test_the_centre_is_taller_than_the_ends():
    """One shape rising and falling, not twenty-two independent meters."""
    heights = settle(BarModel(), 0.9)
    middle = heights[len(heights) // 2]
    assert middle > heights[0] and middle > heights[-1]


def test_bars_never_vanish_entirely():
    assert min(settle(BarModel(), 0.0)) > 0


def test_rises_faster_than_it_falls():
    """A meter that tracks the signal exactly flickers; one that falls slowly
    stays readable."""
    rising = BarModel()
    rising.feed(1.0)
    after_one_rise = rising.level

    falling = BarModel()
    settle(falling, 1.0)
    peak = falling.level
    falling.feed(0.0)
    dropped = peak - falling.level

    assert after_one_rise > dropped


def test_reset_returns_to_silence():
    model = BarModel()
    settle(model, 1.0)
    model.reset()
    assert max(model.heights()) < 0.1


def test_heights_stay_in_range():
    model = BarModel()
    for level in (0.0, 0.5, 1.0, 2.0, -1.0):
        for value in settle(model, level, frames=5):
            assert 0.0 <= value <= 1.0


def test_silent_audio_gives_zero_level():
    assert rms_to_level(np.zeros(1600, dtype=np.float32)) == 0.0


def test_empty_audio_is_safe():
    assert rms_to_level(np.zeros(0, dtype=np.float32)) == 0.0
    assert rms_to_level(None) == 0.0


def test_speech_lands_in_the_useful_part_of_the_range():
    """Speech sits far below full scale; a linear RMS would leave the bars
    almost flat at normal volume, which would make the display useless."""
    speech = (np.random.RandomState(0).randn(1600) * 0.05).astype(np.float32)
    assert 0.3 < rms_to_level(speech) < 0.95


def test_the_panel_never_becomes_key():
    """Load-bearing. A panel that takes focus becomes the injection target, and
    the author's dictation would be pasted into this overlay instead of their
    document."""
    import ast

    from backend import overlay

    source = open(overlay.__file__).read()
    assert "NSWindowStyleMaskNonactivatingPanel" in source
    assert "setBecomesKeyOnlyIfNeeded_(True)" in source
    tree = ast.parse(source)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "makeKeyAndOrderFront_" not in names, "orderFrontRegardless, never makeKey"


def test_importing_needs_no_appkit():
    """The module must import on a machine with no AppKit so the daemon can run
    headless."""
    import ast

    from backend import overlay

    tree = ast.parse(open(overlay.__file__).read())
    top = {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    top |= {n.module for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
    assert "AppKit" not in top and "Foundation" not in top


# --- the menu bar's warnings ---------------------------------------------------


class _FakeDaemon:
    def __init__(self, config, polish=None):
        self.config = config
        self.polish = polish


def test_a_shared_microphone_is_warned_about_in_the_menu(monkeypatch):
    """It degrades every transcription and looks like a bad model. `utter
    doctor` said so and nothing the author saw during normal use did."""
    from backend.config import AppConfig
    from backend.menubar import MenuBar

    monkeypatch.setattr("backend.audio_source.shared_with_output", lambda _d: "AirPods Pro")
    warnings = MenuBar(_FakeDaemon(AppConfig()))._warnings(AppConfig())

    assert any("AirPods Pro" in note.short for note in warnings)


def test_polish_that_is_on_but_broken_is_warned_about(monkeypatch):
    from backend.config import AppConfig
    from backend.menubar import MenuBar

    monkeypatch.setattr("backend.audio_source.shared_with_output", lambda _d: None)
    config = AppConfig(polish_enabled=True)
    warnings = MenuBar(_FakeDaemon(config, polish=None))._warnings(config)

    assert any("润色" in note.short for note in warnings)


def test_a_healthy_setup_shows_no_warnings(monkeypatch):
    """A menu with a permanent warning in it is a menu nobody reads."""
    from backend.config import AppConfig
    from backend.menubar import MenuBar

    monkeypatch.setattr("backend.audio_source.shared_with_output", lambda _d: None)
    config = AppConfig(polish_enabled=True)
    daemon = _FakeDaemon(config, polish=lambda text, context=None: text)

    assert MenuBar(daemon)._warnings(config) == []


def test_every_menu_bar_appkit_call_goes_through_the_main_thread():
    """The status item existed in the accessibility tree and was invisible on
    screen, because the startup status line rebuilt the NSMenu from the
    background thread that warms the model. AppKit is main-thread only, and it
    does not raise — it just quietly does not draw.

    Asserted structurally rather than by mocking AppKit: the methods that touch
    it must hand off, and the ones that do the touching must be the `_now`
    variants nothing else calls directly.
    """
    import inspect

    from backend import menubar

    for name in ("_rebuild", "_set_symbol"):
        source = inspect.getsource(getattr(menubar.MenuBar, name))
        assert "_on_main" in source, f"{name} touches AppKit without hopping threads"
        assert "import AppKit" not in source, f"{name} should delegate to {name}_now"
