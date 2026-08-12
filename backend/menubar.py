"""The menu bar item — so Utter is not a thing you start from a terminal.

Nothing here is functionality; every action already exists on the daemon or in
the config file. What it adds is *reachability*. A tool whose settings live in
a JSON file and whose only launcher is a shell command is a tool that gets used
for a week. v1 died of exactly that friction.
"""

from __future__ import annotations

import logging

from backend.overlay import _on_main

log = logging.getLogger(__name__)

# SF Symbol *names*, not glyph characters. Passing the private-use codepoints
# to setTitle_ renders a question mark in every font that does not carry them,
# which is what the author saw. NSImage resolves them properly, and setTemplate_
# lets macOS invert the icon for light and dark menu bars.
IDLE_SYMBOL, BUSY_SYMBOL = "waveform", "waveform.circle.fill"


class MenuBar:
    """NSStatusItem with a small menu. All AppKit; construct on the main thread."""

    def __init__(self, daemon, on_quit=None, window=None):
        self.daemon = daemon
        self.on_quit = on_quit
        self.window = window
        self._item = None
        self._delegate = None
        self._status = None
        self._menu = None

    def install(self) -> bool:
        try:
            import AppKit
            import objc

            outer = self

            # Named, not `_Delegate`: Objective-C classes share one flat
            # namespace across the process, so a second `_Delegate` anywhere
            # fails to register and takes its whole window with it.
            class UtterMenuDelegate(AppKit.NSObject):
                def openWindow_(self, _sender):
                    if outer.window is not None:
                        outer.window.show()

                def iconClicked_(self, sender):
                    import AppKit

                    event = AppKit.NSApp().currentEvent()
                    right = event is not None and event.type() in (
                        AppKit.NSEventTypeRightMouseDown,
                    )
                    if right or (event is not None
                                 and event.modifierFlags() & AppKit.NSEventModifierFlagControl):
                        outer._popup_menu()
                    elif outer.window is not None:
                        outer.window.show()

                def togglePolish_(self, _sender):
                    want = not outer.daemon.config.polish_enabled
                    on, why = outer.daemon.set_polish(want)
                    if want and not on:
                        outer._notify("润色打不开", why)
                    outer._rebuild()

                def pickPolishLevel_(self, sender):
                    outer.daemon.set_polish(True, str(sender.representedObject()))
                    outer._rebuild()

                def toggleStreaming_(self, _sender):
                    config = outer.daemon.config
                    config.stream_while_speaking = not config.stream_while_speaking
                    outer._persist()
                    outer._rebuild()

                def toggleTarget_(self, _sender):
                    current = outer.daemon.config.dictate_target
                    outer.daemon.config.dictate_target = (
                        "scratchpad" if current == "cursor" else "cursor"
                    )
                    outer._persist()

                def pickModel_(self, sender):
                    ok, message = outer.daemon.switch_provider(str(sender.representedObject()))
                    if not ok:
                        outer._notify("换不了模型", message)
                    outer._rebuild()

                def pickLanguage_(self, sender):
                    value = str(sender.representedObject())
                    outer.daemon.config.dictate_language = None if value == "auto" else value
                    outer._persist()

                def runDoctor_(self, _sender):
                    """`utter doctor` in a window.

                    Every diagnostic this project has built lives behind a
                    terminal command, and the whole point of the .app is that
                    there is no terminal any more."""
                    import subprocess, sys

                    from backend.config import DEFAULT_DIR

                    report = DEFAULT_DIR / "doctor.txt"
                    try:
                        out = subprocess.run(
                            [sys.executable, "-m", "backend.cli", "doctor"],
                            capture_output=True, text=True, timeout=120,
                        ).stdout
                        report.write_text(out or "(没有输出)", encoding="utf-8")
                        subprocess.run(["open", "-t", str(report)], check=False)
                    except Exception as exc:
                        outer._notify("自检失败", str(exc)[:120])

                def openArchive_(self, _sender):
                    import subprocess

                    path = outer.daemon.scratchpad.archive.path
                    subprocess.run(["open", "-R", str(path)], check=False)

                def openConfig_(self, _sender):
                    import subprocess

                    from backend.config import config_path

                    subprocess.run(["open", "-t", str(config_path())], check=False)

                def quit_(self, _sender):
                    if outer.on_quit:
                        outer.on_quit()
                    AppKit.NSApp().terminate_(None)

            self._delegate = UtterMenuDelegate.alloc().init()
            bar = AppKit.NSStatusBar.systemStatusBar()
            self._item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
            # A click opens the window; the menu is the right-click. The menu
            # had grown into the settings UI by accident, with explanations in
            # it, and a menu is the wrong place to read anything twice.
            button = self._item.button()
            button.setTarget_(self._delegate)
            button.setAction_("iconClicked:")
            button.sendActionOn_(
                AppKit.NSEventMaskLeftMouseDown | AppKit.NSEventMaskRightMouseDown)

            self._set_symbol(IDLE_SYMBOL)
            self._rebuild()

            # Where the icon actually landed, from inside the process that owns
            # it. The author reported "no icon" while the accessibility API
            # cheerfully reported one, and inferring the truth from outside
            # cost an hour. isVisible and the window frame are the only two
            # facts that settle it.
            # Twice: once now, once after the status bar has laid it out. The
            # frame is 32x0 at the origin immediately after creation, which
            # looks alarming and means nothing.
            # One sample, five seconds in. A status item lays out
            # asynchronously — the frame is 32x0 immediately after creation and
            # only becomes real a few seconds later — so reading it at
            # creation says nothing, which cost an hour to notice.
            AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                5.0, False, lambda _t: self._log_placement("5秒后")
            )
            return True
        except Exception:
            log.warning("could not install the menu bar item", exc_info=True)
            return False

    def _log_placement(self, when: str) -> None:
        """Where the icon actually is, from inside the process that owns it.

        The author reported "no icon" while the accessibility API cheerfully
        reported one, and inferring the truth from outside cost an hour.
        isVisible plus the window frame are the two facts that settle it.
        """
        try:
            window = self._item.button().window()
            frame = window.frame() if window else None
            log.info(
                "status item %s: visible=%s screen=%s frame=%s",
                when,
                self._item.isVisible(),
                (window.screen().frame() if window and window.screen() else None),
                frame,
            )
        except Exception:  # pragma: no cover
            log.warning("could not read the status item placement", exc_info=True)

    def _popup_menu(self) -> None:
        """Show the menu under the icon, on right-click.

        The status item no longer owns a menu — owning one would swallow the
        left click, and the left click is how the window opens now — so it is
        popped by hand.
        """
        try:
            import AppKit

            self._rebuild_now()
            if self._menu is None:
                return
            button = self._item.button()
            self._menu.popUpMenuPositioningItem_atLocation_inView_(
                None, AppKit.NSMakePoint(0, button.bounds().size.height + 4), button)
        except Exception:  # pragma: no cover
            log.warning("could not show the menu", exc_info=True)

    def set_status(self, text: str | None) -> None:
        """A line at the top of the menu, or None to clear it.

        Exists so the icon can go up before the model has finished warming.
        Warm-up has been measured between 1.7 and 77 seconds, and for all of
        that time the app previously showed nothing at all.
        """
        self._status = text
        self._rebuild()

    # -- every AppKit call below here goes to the main thread --
    #
    # menubar.py got away without this for a week because _rebuild was only
    # ever called from a menu click, which is already on the main thread. The
    # startup status line changed that: it is set from the background thread
    # that warms the model, and building an NSMenu off the main thread left the
    # status item present in the accessibility tree and absent from the screen.
    # The author saw "no icon" while `utter status` cheerfully reported one.
    #
    # overlay.py has done this correctly since it was written; this is the same
    # helper.

    def set_busy(self, busy: bool) -> None:
        self._set_symbol(BUSY_SYMBOL if busy else IDLE_SYMBOL)

    def _set_symbol(self, name: str) -> None:
        if self._item is None:
            return

        def run():
            self._set_symbol_now(name)

        _on_main(run)

    def _set_symbol_now(self, name: str) -> None:
        try:
            import AppKit

            image = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                name, "Utter"
            )
            if image is None:  # very old macOS
                self._item.button().setTitle_("U")
                return
            image.setTemplate_(True)
            self._item.button().setImage_(image)
            self._item.button().setTitle_("")
        except Exception:  # pragma: no cover
            log.warning("could not set the menu bar symbol", exc_info=True)

    def _warnings(self, config):
        """Shared with the settings window, so the two cannot disagree."""
        from backend.health import warnings_for

        return warnings_for(self.daemon, config)

    def _providers(self):
        """Every engine this build knows about, available ones first."""
        from backend.providers.stt import probe_all

        return [
            (s.id, s.display_name if s.available else f"{s.display_name}（不可用）")
            for s in probe_all()
        ]

    def _persist(self) -> None:
        self.daemon._save_config()
        self._rebuild()

    def _notify(self, title: str, body: str) -> None:
        import subprocess

        subprocess.run(
            ["osascript", "-e", f"display notification {body!r} with title {title!r}"],
            check=False, capture_output=True, timeout=5,
        )

    def _rebuild(self) -> None:
        _on_main(self._rebuild_now)

    def _rebuild_now(self) -> None:
        """Four items. It used to be fourteen, with a paragraph in one of them.

        A menu is glanced at; a settings window is read. Every control that
        needed a label longer than its own name moved to the window, and the
        shared-microphone warning became one line instead of a sentence
        explaining itself.
        """
        try:
            import AppKit

            menu = AppKit.NSMenu.alloc().init()

            def add(title, action=None, enabled=True):
                item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    title, action, "")
                if action:
                    item.setTarget_(self._delegate)
                item.setEnabled_(enabled)
                menu.addItem_(item)

            if self._status:
                add(self._status, None, False)
                menu.addItem_(AppKit.NSMenuItem.separatorItem())
            for note in self._warnings(self.daemon.config):
                add(note.short, "openWindow:")
            add("设置…", "openWindow:")
            menu.addItem_(AppKit.NSMenuItem.separatorItem())
            add("退出 Utter", "quit:")

            self._menu = menu
        except Exception:  # pragma: no cover
            log.warning("could not rebuild the menu", exc_info=True)
