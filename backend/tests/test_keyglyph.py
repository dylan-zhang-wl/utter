"""The hotkey spec is stored as pynput writes it and shown as the keyboard
prints it. Written after the settings window shipped reading `按住 <alt_l>`.
"""

import pytest

from backend.keyglyph import glyph


@pytest.mark.parametrize("spec, shown", [
    ("<alt_l>", "左⌥"),
    ("<alt_r>", "右⌥"),
    ("<alt>", "⌥"),
    ("<ctrl_l>", "左⌃"),
    ("<cmd>", "⌘"),
    ("<shift_r>", "右⇧"),
    ("<caps_lock>", "⇪"),
    ("<space>", "空格"),
    ("<fn>", "fn"),
])
def test_the_glyph_is_what_is_printed_on_the_key(spec, shown):
    assert glyph(spec) == shown


def test_function_keys_keep_their_number():
    assert glyph("<f13>") == "F13"
    assert glyph("<f5>") == "F5"


def test_a_letter_is_shown_as_the_capital():
    assert glyph("j") == "J"


def test_no_hotkey_is_a_state_worth_printing():
    """An empty hotkey is a thing the user did, not a bug. The window still
    needs something to put in the sentence."""
    assert glyph(None) == "—"
    assert glyph("") == "—"


def test_an_unknown_key_comes_back_readable_not_raw():
    """Better a bare word than the angle brackets it was stored in — pynput
    grows key names and this must not print `<media_play_pause>`."""
    assert glyph("<media_play_pause>") == "media_play_pause"


def test_a_side_is_only_stripped_off_a_key_that_has_one():
    """`_r` is a suffix on ctrl_r and also the last two characters of plenty of
    words. Only a real sided key may lose it."""
    assert glyph("<something_r>") == "something_r"
