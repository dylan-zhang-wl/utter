"""Anything wrong, said where the user is looking.

This lived inside `MenuBar` until the settings window needed the same answer.
Two copies of "is everything all right" is one copy too many: the day they
disagree, the menu says fine and the window says broken and neither is
trustworthy.

Each note carries its own explanation. A warning that says only
「AirPods Pro 同时是麦克风和扬声器」 tells a user who has never heard of
Bluetooth audio profiles precisely nothing — and this one is worth
understanding, because the damage it does looks exactly like a bad model.
"""

from __future__ import annotations

from typing import NamedTuple


class Note(NamedTuple):
    """`short` goes in the menu and the tooltip; `why` is the paragraph behind
    it, shown when the user asks."""

    short: str
    why: str


def warnings_for(daemon, config) -> list[Note]:
    """Problems worth interrupting the user about, most concrete first.

    The microphone being shared with the speakers is the one that has actually
    bitten: it degrades every transcription and looks like a bad model. It was
    reported by `utter doctor` and nowhere the author would see it during
    normal use.
    """
    notes = []
    try:
        from backend.audio_source import shared_with_output

        shared = shared_with_output(config.input_device)
        if shared:
            notes.append(Note(
                f"⚠ {shared} 同时是麦克风和扬声器",
                "蓝牙耳机同时收音和放音时，macOS 会把它切到通话模式：单声道、低码率。"
                "转录会变差——像是模型不行，其实声音进来时就差了。\n\n"
                "想避开：「麦克风」里改选内置麦克风。"))
    except Exception:  # pragma: no cover - probing hardware must never throw
        pass
    if config.polish_enabled and getattr(daemon, "polish", None) is None:
        notes.append(Note(
            "⚠ 润色用不了",
            "钥匙串里没有可用的 key，或者服务连不上。\n\n"
            "听写不受影响，会照原样注入转录。"))
    return notes
