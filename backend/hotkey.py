"""Global hotkey: the only way to start dictating without leaving your document.

Two layers, as with the VAD. `HotkeyMatcher` is a pure state machine over key
press and release and holds all the behaviour; `HotkeyListener` is the pynput
wiring around it.

`pynput.keyboard.GlobalHotKeys` is not enough. It fires on activation only, and
push-to-talk needs the release as well — that release *is* the end of the
utterance.

## The permission

macOS gates global key capture behind Accessibility. Denied, pynput's listener
receives nothing and raises nothing: the process runs, the hotkey does nothing,
and there is no error anywhere. That is the worst possible outcome for a tool
whose entire promise is not to lose what you said, so `start()` probes
`AXIsProcessTrusted()` and refuses rather than pretending.

Note the grant follows the **host application**, not this script. Run from
iTerm and iTerm holds it; run from Terminal.app and that does. A packaged Utter
will need its own grant, which is why this belongs in the first-run wizard.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Literal

from pynput import keyboard

log = logging.getLogger(__name__)

Mode = Literal["push", "toggle", "double_toggle"]
MODES = ("push", "toggle", "double_toggle")

# How close two presses must be to count as a double tap. 400ms is the interval
# macOS itself uses for its double-tap dictation shortcut.
DOUBLE_TAP_SECONDS = 0.4

SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"

# Written out at length because the short version sent the author to the wrong
# pane. macOS has two settings pages called 辅助功能: the accessibility *features*
# (VoiceOver, Zoom) at the top level, and the permission list buried under
# 隐私与安全性. They are one click apart and identically named.
PERMISSION_HINT = (
    "缺少「辅助功能」权限，热键收不到任何按键。\n"
    "\n"
    "  1. 打开权限页（注意不是系统设置里那个同名的「辅助功能」功能页）：\n"
    f"       open '{SETTINGS_URL}'\n"
    "  2. 在名单里打开「终端」的开关；没有就点 + 号，从 应用程序/实用工具 里添加。\n"
    "     勾的是终端本身，不是 Python —— macOS 把权限发给发起进程的那个 app。\n"
    "  3. ⌘Q 完全退出终端再重开。权限只在启动时读取一次。\n"
    "  4. 用 `utter doctor` 确认。\n"
    "\n"
    "(Accessibility permission missing. Grant it to your terminal app, then "
    "quit and relaunch it — the permission is only read at launch.)"
)


class HotkeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class HotkeyEvent:
    kind: Literal["start", "stop"]
    mode: Mode
    at: float
    """`time.perf_counter()`, not wall clock — Task 3 measures latency from here."""


# Side-specific modifiers, which pynput's own parser cannot express usefully.
#
# `HotKey.parse("<alt_r>")` yields a bare virtual keycode, while the listener
# reports `Key.alt_r`; and `listener.canonical()` folds `Key.alt_r` down to
# `Key.alt`, so left and right become indistinguishable. Measured on macOS
# 2026-08-10: right Option arrives as `Key.alt_r`, left as `Key.alt`.
#
# This matters because a single side-specific modifier is the best push-to-talk
# key there is — nothing is bound to right Option alone, and it is comfortable to
# hold for the length of a sentence.
_SIDED = {
    "alt_r": keyboard.Key.alt_r,
    "alt_l": keyboard.Key.alt_l,
    "cmd_r": keyboard.Key.cmd_r,
    "cmd_l": keyboard.Key.cmd_l,
    "ctrl_r": keyboard.Key.ctrl_r,
    "ctrl_l": keyboard.Key.ctrl_l,
    "shift_r": keyboard.Key.shift_r,
    "shift_l": keyboard.Key.shift_l,
}

# A side-agnostic modifier is satisfied by either side.
_EITHER_SIDE = {
    keyboard.Key.alt: {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r},
    keyboard.Key.cmd: {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
    keyboard.Key.ctrl: {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
    keyboard.Key.shift: {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r},
}


def parse_combination(spec: str) -> frozenset[frozenset]:
    """Parse a spec such as "<alt_r>" or "<cmd>+<alt>+d".

    Returns one frozenset per slot, holding every raw key that satisfies it.
    "<alt>" accepts either Option; "<alt_r>" accepts only the right one.

    Raises at parse time, so a typo in the config fails at startup while the
    user is looking at the terminal rather than the first time they reach for a
    hotkey that turns out not to exist.
    """
    if not spec or not spec.strip():
        raise HotkeyError("hotkey combination is empty")

    slots = []
    for part in spec.split("+"):
        part = part.strip()
        if not part:
            raise HotkeyError(f"cannot parse hotkey {spec!r}: empty component")

        name = part[1:-1] if part.startswith("<") and part.endswith(">") else None
        if name in _SIDED:
            slots.append(frozenset({_SIDED[name]}))
            continue

        try:
            parsed = keyboard.HotKey.parse(part)
        except Exception as exc:
            raise HotkeyError(f"cannot parse hotkey {spec!r}: {exc}") from exc
        if not parsed:
            raise HotkeyError(f"cannot parse hotkey {spec!r}: {part!r} matched nothing")

        key = parsed[0]
        slots.append(frozenset(_EITHER_SIDE.get(key, {key})))

    return frozenset(slots)


class HotkeyMatcher:
    """Key events in, start/stop events out. No pynput, no threads, no I/O.

    `combination` is a set of slots, each a set of raw keys that satisfy it —
    see parse_combination. A plain set of keys is accepted too and treated as
    one exact key per slot, which is what the tests use.
    """

    def __init__(self, combination, mode: Mode, on_event: Callable[[HotkeyEvent], None]):
        if mode not in MODES:
            raise ValueError(f"unknown hotkey mode {mode!r}; expected one of {MODES}")
        if not combination:
            raise ValueError("hotkey combination cannot be empty")

        self.slots = [
            slot if isinstance(slot, (set, frozenset)) else frozenset({slot})
            for slot in combination
        ]
        self.combination = frozenset().union(*self.slots)
        self.mode = mode
        self.on_event = on_event
        self._held: set = set()
        self._engaged = False  # combination currently satisfied
        self._active = False  # dictation currently running
        # -inf, not 0.0: a zero would sit within the double-tap window of any
        # clock that happens to start near zero, making the very first single
        # tap fire. Real perf_counter values are large enough that it never
        # showed up in use, which is exactly the kind of bug that surfaces on
        # someone else's machine.
        self._last_tap = float("-inf")

    def _satisfied(self) -> bool:
        return all(slot & self._held for slot in self.slots)

    def reset(self) -> None:
        """Forget which keys are down.

        Needed after a re-grab: stale held keys would make the next single press
        look like a completed combination.
        """
        self._held.clear()
        self._engaged = False

    def press(self, key) -> None:
        self._held.add(key)
        if self._engaged or not self._satisfied():
            # Already engaged means key repeat, which fires press over and over.
            # One dictation, not fifty.
            return

        self._engaged = True
        if self.mode == "push":
            self._start()
        elif self.mode == "toggle":
            self._stop() if self._active else self._start()
        else:
            self._double_tap()

    def release(self, key) -> None:
        self._held.discard(key)
        if not self._engaged or key not in self.combination:
            return

        self._engaged = False
        if self.mode == "push" and self._active:
            # The release is the end of the utterance.
            self._stop()

    def _double_tap(self) -> None:
        """Two presses inside the window flip the state; a single press does nothing.

        Single-press toggle on a bare modifier is a trap: right Option is a key
        people brush against, and an accidental toggle silently starts recording
        everything said next. A double tap is essentially never accidental,
        which is why macOS uses one for its own dictation shortcut.
        """
        now = time.perf_counter()
        if now - self._last_tap <= DOUBLE_TAP_SECONDS:
            self._last_tap = float("-inf")
            self._stop() if self._active else self._start()
        else:
            self._last_tap = now

    def _start(self) -> None:
        self._active = True
        self._emit("start")

    def _stop(self) -> None:
        self._active = False
        self._emit("stop")

    def _emit(self, kind) -> None:
        self.on_event(HotkeyEvent(kind=kind, mode=self.mode, at=time.perf_counter()))

    @property
    def active(self) -> bool:
        return self._active


def _accessibility_trusted() -> bool:
    try:
        from ApplicationServices import AXIsProcessTrusted

        return bool(AXIsProcessTrusted())
    except Exception:  # pragma: no cover - non-macOS, or pyobjc missing
        return False


class HotkeyListener:
    """pynput wiring around a HotkeyMatcher. Use as a context manager."""

    def __init__(
        self,
        on_event: Callable[[HotkeyEvent], None],
        combination: str = "<cmd>+<alt>",
        mode: Mode = "push",
    ):
        # Parse eagerly: a bad combination should fail here, not on first press.
        self.matcher = HotkeyMatcher(
            combination=parse_combination(combination), mode=mode, on_event=on_event
        )
        self.combination = combination
        self._listener = None

    def is_available(self) -> tuple[bool, str]:
        if not _accessibility_trusted():
            return False, PERMISSION_HINT
        return True, ""

    def start(self) -> "HotkeyListener":
        if self._listener is not None:
            return self

        available, reason = self.is_available()
        if not available:
            # Refuse loudly. A listener that runs and hears nothing is worse
            # than one that never started.
            raise HotkeyError(reason)

        self.matcher.reset()
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.start()
        return self

    def close(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()

    def __enter__(self) -> "HotkeyListener":
        return self.start()

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def _canonical(self, key):
        """Normalise letters for keyboard layout, but leave modifiers alone.

        `listener.canonical()` folds Key.alt_r into Key.alt, which would make a
        right-Option hotkey fire on the left one too. Modifiers have no layout
        problem to solve, so they skip it.
        """
        if isinstance(key, keyboard.Key):
            return key
        listener = self._listener
        return listener.canonical(key) if listener is not None else key

    def _on_press(self, key) -> None:
        try:
            self.matcher.press(self._canonical(key))
        except Exception:  # pragma: no cover - a listener callback must not die
            log.warning("hotkey press handler failed", exc_info=True)

    def _on_release(self, key) -> None:
        try:
            self.matcher.release(self._canonical(key))
        except Exception:  # pragma: no cover
            log.warning("hotkey release handler failed", exc_info=True)
