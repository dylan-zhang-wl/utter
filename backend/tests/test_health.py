"""The one answer to "is everything all right", shared by the menu bar and the
settings window so they cannot disagree."""

import types

import pytest

from backend import health
from backend.config import AppConfig


@pytest.fixture(autouse=True)
def no_shared_device(monkeypatch):
    """Hardware answers vary by machine; a test that reads the real one is not
    a test. Each case says what it wants the hardware to be."""
    import backend.audio_source as audio

    monkeypatch.setattr(audio, "shared_with_output", lambda *a, **k: None)


def _daemon(polish=object()):
    return types.SimpleNamespace(polish=polish)


def test_a_healthy_setup_says_nothing():
    assert health.warnings_for(_daemon(), AppConfig()) == []


def test_a_microphone_that_is_also_the_speakers_is_reported(monkeypatch):
    """The one that has actually bitten: it degrades every transcription and
    reads as a bad model rather than as a bad input."""
    import backend.audio_source as audio

    monkeypatch.setattr(audio, "shared_with_output", lambda *a, **k: "AirPods Pro")
    note, = health.warnings_for(_daemon(), AppConfig())
    assert note.short == "⚠ AirPods Pro 同时是麦克风和扬声器"


def test_polish_switched_on_but_unusable_is_reported():
    note, = health.warnings_for(_daemon(polish=None), AppConfig(polish_enabled=True))
    assert note.short == "⚠ 润色用不了"


def test_polish_off_is_not_a_problem_even_with_no_polisher():
    """Nobody asked for polish, so its absence is not news."""
    assert health.warnings_for(_daemon(polish=None), AppConfig()) == []


def test_probing_the_hardware_must_never_throw(monkeypatch):
    """A window that will not open because CoreAudio hiccupped is worse than a
    window with a missing warning."""
    import backend.audio_source as audio

    def boom(*a, **k):
        raise OSError("CoreAudio is having a day")

    monkeypatch.setattr(audio, "shared_with_output", boom)
    assert health.warnings_for(_daemon(), AppConfig()) == []


def test_every_note_explains_itself(monkeypatch):
    """The triangle in the window is one glyph. Whatever is behind it has to
    say enough that a user who has never heard of Bluetooth audio profiles
    knows what to do — a tooltip repeating the warning is not an explanation.
    """
    import backend.audio_source as audio

    monkeypatch.setattr(audio, "shared_with_output", lambda *a, **k: "AirPods Pro")
    notes = health.warnings_for(_daemon(polish=None), AppConfig(polish_enabled=True))

    assert len(notes) == 2
    for note in notes:
        assert note.why != note.short, f"{note.short} just repeats itself"
        assert len(note.why) > len(note.short), f"{note.short} explains nothing"
        # Two paragraphs: what is happening, then what to do about it. Short is
        # good — the author asked for shorter — but a note that only describes
        # the problem leaves the reader exactly where they started.
        assert "\n\n" in note.why, f"{note.short} says what, never what to do"
