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
            add(
                f"边说边出字：{'开' if config.stream_while_speaking else '关'}"
                f"（{config.hotkey_toggle or '未设置'} 双击）",
                "toggleStreaming:",
            )
            add(
                f"润色：{'开（' + config.polish_level + '）' if config.polish_enabled else '关'}",
                "togglePolish:",
            )
            if config.polish_enabled:
                # Only the levels 铁律 10 defines, and the name says what each
                # one is allowed to touch — "medium" tells the author nothing
                # about what a model is about to do to their argument.
                levels = AppKit.NSMenu.alloc().init()
                for value, label in (
                    ("light", "轻 —— 只补标点"),
                    ("medium", "中 —— 标点 + 删口水词"),
                    ("heavy", "重 —— 标点 + 口水词 + 分段"),
                ):
                    item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                        label, "pickPolishLevel:", ""
                    )
                    item.setTarget_(self._delegate)
                    item.setRepresentedObject_(value)
                    item.setState_(1 if config.polish_level == value else 0)
                    levels.addItem_(item)
                holder = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    "  润色档位", None, ""
                )
                menu.addItem_(holder)
                menu.setSubmenu_forItem_(levels, holder)

            # Switching engines and languages is the whole comparison the author
            # is running. Making it cost a terminal visit is how a comparison
            # quietly does not get run.
            models = AppKit.NSMenu.alloc().init()
            for pid, label in self._providers():
                item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    label, "pickModel:", ""
                )
                item.setTarget_(self._delegate)
                item.setRepresentedObject_(pid)
                item.setState_(1 if getattr(self.daemon.stt, "id", "") == pid else 0)
                models.addItem_(item)
            holder = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                f"模型：{getattr(self.daemon.stt, 'display_name', '?')}", None, ""
            )
            menu.addItem_(holder)
            menu.setSubmenu_forItem_(models, holder)

            languages = AppKit.NSMenu.alloc().init()
            for value, label in (("auto", "自动检测"), ("zh", "中文"), ("en", "English")):
                item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    label, "pickLanguage:", ""
                )
                item.setTarget_(self._delegate)
                item.setRepresentedObject_(value)
                current = config.dictate_language or "auto"
                item.setState_(1 if current == value else 0)
                languages.addItem_(item)
            holder = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                f"语言：{config.dictate_language or '自动'}", None, ""
            )
            menu.addItem_(holder)
            menu.setSubmenu_forItem_(languages, holder)
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
