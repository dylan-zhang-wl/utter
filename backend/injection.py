"""Getting dictated text to the cursor.

Three rules shape everything here, and all three are about what this module
refuses to do.

**铁律 12 — paste, never type.** 1500 words of synthetic keystrokes takes
seconds, and any application that steals focus partway leaves half a paragraph
somewhere unexpected. One clipboard write plus one ⌘V is atomic from the
application's point of view. The price is borrowing the user's clipboard, which
is why every path here restores it — including flavours we do not understand.
Restoring only `public.utf8-plain-text` would silently destroy a copied image,
and the author would never connect the loss to having dictated.

**铁律 13 — buffer, do not force.** Two other routes exist for putting text in a
non-frontmost window: `CGEventPostToPid`, and setting an Accessibility element's
value directly. Design §4.1e rejected both. They work on native Cocoa fields and
fail on Electron, browsers and terminals — which is most of what the author
writes in — and they fail *silently*. Random failure is worse than clear
failure, because the user cannot tell which segment was lost. So when focus has
moved, text waits.

**铁律 9 — one utterance, one injection.** Nothing already delivered is ever
revised. A design that injects a draft and corrects it later reads well in a
demo and destroys a document the moment the user edits between the two writes.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# How long to wait after ⌘V before putting the user's clipboard back.
#
# ⌘V only *posts* a keystroke. The target application reads the pasteboard some
# milliseconds later, on its own event loop. Restoring immediately after the
# post — which is what the first version did — means the application reads
# whatever was there before, and the author watched a previously-copied shell
# command appear where their dictation should have been.
#
# 250ms is chosen to be comfortably longer than an application's event-loop
# turn while staying invisible next to the ~1.1s the transcription already took.
PASTE_SETTLE_SECONDS = 0.25


@dataclass(frozen=True)
class Target:
    pid: int
    name: str
    bundle_id: str | None = None


@dataclass(frozen=True)
class InjectionResult:
    injected: bool
    buffered: bool = False
    reason: str = ""


# --- the macOS edges, isolated so tests can replace them ----------------------


def _frontmost() -> Target | None:
    try:
        from AppKit import NSWorkspace

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return None
        return Target(
            pid=int(app.processIdentifier()),
            name=str(app.localizedName()),
            bundle_id=str(app.bundleIdentifier()) if app.bundleIdentifier() else None,
        )
    except Exception:  # pragma: no cover - non-macOS
        return None


def _is_running(pid: int) -> bool:
    try:
        from AppKit import NSRunningApplication

        return NSRunningApplication.runningApplicationWithProcessIdentifier_(pid) is not None
    except Exception:  # pragma: no cover
        return False


ACTIVATION_TIMEOUT_SECONDS = 1.5


def _activate_and_wait(pid: int, timeout: float = ACTIVATION_TIMEOUT_SECONDS) -> bool:
    """Bring an application forward and wait until it really is forward.

    `activateWithOptions_` only *requests* activation; the application becomes
    frontmost some tens of milliseconds later. Firing ⌘V straight afterwards
    sends it to whoever is still in front — the author dictated from Claude,
    switched to WeChat, and watched the text land in WeChat while the log
    faithfully reported "injected into Claude". Both halves were telling the
    truth; nothing waited in between.

    Same shape of mistake as restoring the clipboard before the paste had
    landed. Asynchronous means asynchronous.
    """
    _activate(pid)
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        front = _frontmost()
        if front is not None and front.pid == pid:
            # Frontmost is set slightly before the window is ready for input.
            time.sleep(0.05)
            return True
        time.sleep(0.03)
    return False


def _activate(pid: int) -> None:
    try:
        from AppKit import NSRunningApplication

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if app is not None:
            app.activateWithOptions_(1 << 1)  # NSApplicationActivateIgnoringOtherApps
    except Exception:  # pragma: no cover
        log.warning("could not activate pid %s", pid, exc_info=True)


def _notify(title: str, body: str) -> None:
    """A user-visible notification. Never raises — this is the *fallback* path,
    and a fallback that can fail is not one."""
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification {body!r} with title {title!r}'],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except Exception:  # pragma: no cover
        log.warning("could not post notification: %s", body)


class Pasteboard:
    """NSPasteboard with a full-fidelity save and restore."""

    def snapshot(self) -> list[dict]:
        from AppKit import NSPasteboard

        board = NSPasteboard.generalPasteboard()
        saved = []
        for item in board.pasteboardItems() or []:
            flavours = {}
            for kind in item.types() or []:
                data = item.dataForType_(kind)
                if data is not None:
                    flavours[str(kind)] = bytes(data)
            if flavours:
                saved.append(flavours)
        return saved

    def restore(self, snapshot: list[dict]) -> None:
        from AppKit import NSPasteboard, NSPasteboardItem
        from Foundation import NSData

        board = NSPasteboard.generalPasteboard()
        board.clearContents()
        if not snapshot:
            return

        items = []
        for flavours in snapshot:
            item = NSPasteboardItem.alloc().init()
            for kind, data in flavours.items():
                item.setData_forType_(NSData.dataWithBytes_length_(data, len(data)), kind)
            items.append(item)
        board.writeObjects_(items)

    def set_text(self, text: str) -> None:
        from AppKit import NSPasteboard

        board = NSPasteboard.generalPasteboard()
        board.clearContents()
        board.setString_forType_(text, "public.utf8-plain-text")


class Keyboard:
    """⌘V, with the Controller built once and kept.

    Constructing a pynput Controller enters macOS's non-reentrant
    `keycode_context`. Doing that on every paste, while a hotkey listener thread
    is inside the same context, is how this process learned to die with SIGABRT
    and no traceback (see HotkeyListener's docstring). Build it once, at a
    moment of our choosing, and reuse it.
    """

    def __init__(self):
        self._controller = None

    def _get(self):
        if self._controller is None:
            from pynput.keyboard import Controller

            self._controller = Controller()
        return self._controller

    def warm_up(self) -> None:
        """Build the Controller now rather than mid-dictation."""
        self._get()

    def paste(self) -> None:
        from pynput.keyboard import Key

        controller = self._get()
        with controller.pressed(Key.cmd):
            controller.press("v")
            controller.release("v")


@dataclass
class Injector:
    """Delivers text to the locked target, or holds it until that is possible."""

    pasteboard: object = field(default_factory=Pasteboard)
    keyboard: object = field(default_factory=Keyboard)
    target: Target | None = None

    _buffer: list[str] = field(default_factory=list)
    _seen: set[int] = field(default_factory=set)

    # -- target --

    def lock_target(self, target: Target | None = None) -> Target | None:
        """Remember where this dictation is going. Called when the hotkey goes down."""
        self.target = target or _frontmost()
        return self.target

    def release(self) -> None:
        self.target = None

    # -- delivery --

    def inject(self, index: int, text: str) -> InjectionResult:
        """Deliver one utterance. Never raises, never delivers the same one twice."""
        if not text or not text.strip():
            return InjectionResult(injected=False, reason="empty")

        if index in self._seen:
            # 铁律 9. Whatever produced a repeat, honouring it would mean
            # writing into a document the user has since edited.
            log.warning("utterance %d already injected, refusing to repeat it", index)
            return InjectionResult(injected=False, reason="already injected")
        self._seen.add(index)

        if self.target is None:
            self._buffer.append(text)
            return InjectionResult(injected=False, buffered=True, reason="no target locked")

        if not _is_running(self.target.pid):
            return self._fall_back_to_clipboard(text)

        front = _frontmost()
        if front is None or front.pid != self.target.pid:
            # The author's expectation, stated plainly: text belongs where they
            # were when they started talking, whatever they wandered off to
            # while saying it. So bring that window back — and *verify* it came
            # back before typing into it.
            #
            # This is not the thing 铁律 13 forbids. That rule rejects posting
            # keystrokes at a window that is not in front, because it fails
            # silently and randomly. Activating the window and confirming it is
            # frontmost fails visibly and deterministically: if it does not come
            # forward, nothing is typed and the text waits.
            if not _activate_and_wait(self.target.pid):
                self._buffer.append(text)
                return InjectionResult(
                    injected=False, buffered=True,
                    reason=f"{self.target.name} 没能切回前台（当时在 {front.name if front else '未知'}）",
                )

        return self._paste(text)

    def flush(self) -> InjectionResult:
        """Deliver everything held, as one paste."""
        if not self._buffer:
            return InjectionResult(injected=False, reason="nothing pending")

        text = " ".join(self._buffer)
        if self.target is None or not _is_running(self.target.pid):
            self._buffer.clear()
            return self._fall_back_to_clipboard(text)

        if not _activate_and_wait(self.target.pid):
            return InjectionResult(injected=False, buffered=True,
                                   reason=f"{self.target.name} 没能切回前台")
        result = self._paste(text)
        if result.injected:
            self._buffer.clear()
        return result

    @property
    def pending(self) -> int:
        return len(self._buffer)

    @property
    def pending_text(self) -> str:
        return " ".join(self._buffer)

    # -- internals --

    def _paste(self, text: str) -> InjectionResult:
        saved = None
        try:
            saved = self.pasteboard.snapshot()
        except Exception:
            log.warning("could not read the clipboard; it will not be restored", exc_info=True)

        try:
            self.pasteboard.set_text(text)
            self.keyboard.paste()
            # Let the application actually read the pasteboard before taking it
            # away again. See PASTE_SETTLE_SECONDS.
            self.settle()
            return InjectionResult(injected=True)
        except Exception as exc:
            log.warning("paste failed", exc_info=True)
            return InjectionResult(injected=False, reason=f"paste failed: {exc}")
        finally:
            # Restored whatever happened above. A failed dictation must not also
            # cost the user whatever they had copied.
            if saved is not None:
                try:
                    self.pasteboard.restore(saved)
                except Exception:  # pragma: no cover
                    log.warning("could not restore the clipboard", exc_info=True)

    def settle(self) -> None:
        """Overridable so tests do not spend a quarter second each."""
        time.sleep(PASTE_SETTLE_SECONDS)

    def _fall_back_to_clipboard(self, text: str) -> InjectionResult:
        """Design §4.1c line 3. The target is gone; leave the words somewhere
        the user can reach and tell them so. Silence here would be the exact
        failure this project refuses."""
        try:
            self.pasteboard.set_text(text)
        except Exception as exc:  # pragma: no cover
            log.error("could not even reach the clipboard: %s", exc)
            return InjectionResult(injected=False, reason="clipboard unavailable")

        _notify("Utter", "目标窗口已关闭，文字已复制到剪贴板，按 ⌘V 粘贴")
        return InjectionResult(injected=False, reason="target gone; text on clipboard")
