"""The menu bar item — so Utter is not a thing you start from a terminal.

Nothing here is functionality; every action already exists on the daemon or in
the config file. What it adds is *reachability*. A tool whose settings live in
a JSON file and whose only launcher is a shell command is a tool that gets used
for a week. v1 died of exactly that friction.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# SF Symbol *names*, not glyph characters. Passing the private-use codepoints
# to setTitle_ renders a question mark in every font that does not carry them,
# which is what the author saw. NSImage resolves them properly, and setTemplate_
# lets macOS invert the icon for light and dark menu bars.
IDLE_SYMBOL, BUSY_SYMBOL = "waveform", "waveform.circle.fill"


class MenuBar:
    """NSStatusItem with a small menu. All AppKit; construct on the main thread."""

    def __init__(self, daemon, on_quit=None):
        self.daemon = daemon
        self.on_quit = on_quit
        self._item = None
        self._delegate = None

    def install(self) -> bool:
        try:
            import AppKit
            import objc

            outer = self

            class _Delegate(AppKit.NSObject):
                def togglePolish_(self, _sender):
                    outer.daemon.config.polish_enabled = not outer.daemon.config.polish_enabled
                    outer._rebuild()

                def toggleTarget_(self, _sender):
                    current = outer.daemon.config.dictate_target
                    outer.daemon.config.dictate_target = (
                        "scratchpad" if current == "cursor" else "cursor"
                    )
                    outer._rebuild()

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

            self._delegate = _Delegate.alloc().init()
            bar = AppKit.NSStatusBar.systemStatusBar()
            self._item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
            self._set_symbol(IDLE_SYMBOL)
            self._rebuild()
            return True
        except Exception:
            log.warning("could not install the menu bar item", exc_info=True)
            return False

    def set_busy(self, busy: bool) -> None:
        self._set_symbol(BUSY_SYMBOL if busy else IDLE_SYMBOL)

    def _set_symbol(self, name: str) -> None:
        if self._item is None:
            return
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

    def _rebuild(self) -> None:
        try:
            import AppKit

            config = self.daemon.config
            menu = AppKit.NSMenu.alloc().init()

            def add(title, action=None, enabled=True):
                item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    title, action, ""
                )
                if action:
                    item.setTarget_(self._delegate)
                item.setEnabled_(enabled)
                menu.addItem_(item)

            add(f"按住 {config.hotkey_push or '未设置'} 说话", None, False)
            menu.addItem_(AppKit.NSMenuItem.separatorItem())
            add(f"润色：{'开' if config.polish_enabled else '关'}", "togglePolish:")
            add(
                f"输出：{'光标处' if config.dictate_target == 'cursor' else '暂存区'}",
                "toggleTarget:",
            )
            menu.addItem_(AppKit.NSMenuItem.separatorItem())
            add(f"本次已听写 {len(self.daemon.scratchpad)} 段", None, False)
            add("打开存档", "openArchive:")
            add("编辑设置…", "openConfig:")
            menu.addItem_(AppKit.NSMenuItem.separatorItem())
            add("退出 Utter", "quit:")

            self._item.setMenu_(menu)
        except Exception:  # pragma: no cover
            log.warning("could not rebuild the menu", exc_info=True)
