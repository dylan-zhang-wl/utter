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

Mode = Literal["push", "toggle"]
MODES = ("push", "toggle")

PERMISSION_HINT = (
    "缺少「辅助功能」权限，热键收不到任何按键。"
    "请到 系统设置 → 隐私与安全性 → 辅助功能 中勾选运行 Utter 的那个程序"
    "（从终端运行时，要勾的是那个终端应用本身，不是 Python）。"
    "(Accessibility permission missing — see System Settings → Privacy & Security)"
)


class HotkeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class HotkeyEvent:
    kind: Literal["start", "stop"]
    mode: Mode
    at: float
    """`time.perf_counter()`, not wall clock — Task 3 measures latency from here."""


def parse_combination(spec: str) -> frozenset:
    """Parse a pynput-style spec such as "<cmd>+<alt>+d".

    Raises at parse time so a typo in the config fails at startup, while the
    user is looking at the terminal, rather than the first time they reach for
    a hotkey that turns out not to exist.
    """
    if not spec or not spec.strip():
        raise HotkeyError("hotkey combination is empty")
    try:
        return frozenset(keyboard.HotKey.parse(spec))
    except Exception as exc:
        raise HotkeyError(f"cannot parse hotkey {spec!r}: {exc}") from exc


class HotkeyMatcher:
    """Key events in, start/stop events out. No pynput, no threads, no I/O."""

    def __init__(self, combination: frozenset, mode: Mode, on_event: Callable[[HotkeyEvent], None]):
        if mode not in MODES:
            raise ValueError(f"unknown hotkey mode {mode!r}; expected one of {MODES}")
        if not combination:
            raise ValueError("hotkey combination cannot be empty")

        self.combination = combination
        self.mode = mode
        self.on_event = on_event
        self._held: set = set()
        self._engaged = False  # combination currently satisfied
        self._active = False  # dictation currently running

    def reset(self) -> None:
        """Forget which keys are down.

        Needed after a re-grab: stale held keys would make the next single press
        look like a completed combination.
        """
        self._held.clear()
        self._engaged = False

    def press(self, key) -> None:
        self._held.add(key)
        if self._engaged or not self.combination <= self._held:
            # Already engaged means key repeat, which fires press over and over.
            # One dictation, not fifty.
            return

        self._engaged = True
        if self.mode == "push":
            self._start()
        else:
            self._stop() if self._active else self._start()

    def release(self, key) -> None:
        self._held.discard(key)
        if not self._engaged or key not in self.combination:
            return

        self._engaged = False
        if self.mode == "push" and self._active:
            # The release is the end of the utterance.
            self._stop()

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
