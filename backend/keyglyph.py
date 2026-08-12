"""Config key names as the glyphs printed on the keyboard.

The hotkey lives in `config.json` as a pynput spec — `<alt_l>` — because that
is what pynput reports and what the user has to type back if they change it.
It is the right thing to store and the wrong thing to show: a settings window
that says "按住 <alt_l>" is showing its internals.

Sided keys keep their side. macOS writes ⌥ with no side because either one
works for a shortcut; here the left Option and the right Option are genuinely
different hotkeys, and the right one is WeChat's, so 「左 ⌥」 is the honest
label and 「⌥」 would be a lie.
"""

from __future__ import annotations

#: Glyphs first, words only where macOS itself uses a word.
_GLYPHS = {
    "alt": "⌥",
    "ctrl": "⌃",
    "cmd": "⌘",
    "shift": "⇧",
    "caps_lock": "⇪",
    "tab": "⇥",
    "esc": "⎋",
    "enter": "↩",
    "backspace": "⌫",
    "delete": "⌦",
    "space": "空格",
    "fn": "fn",
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
}

_SIDES = {"_l": "左", "_r": "右"}


def glyph(spec: str | None) -> str:
    """`<alt_l>` → `左 ⌥`. Anything unrecognised comes back readable, not raw.

    Never returns the empty string: an empty hotkey is a real state (the user
    cleared it) and the caller wants something to print.
    """
    if not spec:
        return "—"
    name = spec.strip()
    if name.startswith("<") and name.endswith(">"):
        name = name[1:-1]

    side = ""
    for suffix, word in _SIDES.items():
        if name.endswith(suffix) and name[: -len(suffix)] in _GLYPHS:
            side, name = word, name[: -len(suffix)]
            break

    if name in _GLYPHS:
        return f"{side}{_GLYPHS[name]}"
    if len(name) == 2 and name[0] == "f" and name[1].isdigit():
        return name.upper()
    if len(name) == 3 and name[0] == "f" and name[1:].isdigit():
        return name.upper()
    if len(name) == 1:
        return name.upper()
    return name
