"""The main window — settings you can see, rather than a JSON file.

Utter has been a menu bar item and a config file. That is enough to *use* but
not enough to *understand*: every question the author has asked over two days
("which model is it on", "is polish actually running", "why is it not hearing
me") had an answer that existed only in a log or a JSON key. This window is
those answers, arranged.

## Not a second app

AppKit, in the same process, sharing the same daemon object. The alternative
was the Tauri window that already exists in frontend/ — but that is a second
runtime, a second process and an IPC link that has been broken since v1, and a
settings pane is not worth any of those. Tauri stays for listen mode, which
genuinely needs a document-shaped window.

## Closing is not quitting

The close button hides the window and leaves the daemon running, the way every
menu bar app on macOS behaves. Quitting is a menu item, deliberately — the
author has lost a running daemon to a stray ⌘Q before.

## Layout

macOS settings convention: right-aligned labels in a fixed-width column, one
control per row, the whole thing left to size itself. No custom drawing, no
colours of our own — the point of a native window is that it does not look
like ours.
"""

from __future__ import annotations

import logging

from backend.overlay import _on_main

log = logging.getLogger(__name__)

WIDTH = 520
LABEL_WIDTH = 96
ROW_HEIGHT = 30
MARGIN = 24


class MainWindow:
    """The settings window. Construct once; `show()` is safe from any thread."""

    def __init__(self, daemon, on_quit=None):
        self.daemon = daemon
        self.on_quit = on_quit
        self._window = None
        self._delegate = None
        self._fields = {}

    # -- public --

    def show(self) -> None:
        def run():
            if self._window is None and not self._build():
                return
            self._refresh()
            self._window.makeKeyAndOrderFront_(None)
            import AppKit

            # An accessory app has to ask, or the window opens behind whatever
            # the author was typing into.
            AppKit.NSApp().activateIgnoringOtherApps_(True)

        _on_main(run)

    # -- construction --

    def _build(self) -> bool:
        try:
            import AppKit
            import objc

            outer = self

            # Objective-C has one flat class namespace for the whole process,
            # and menubar.py already registers a `_Delegate`. Two of them means
            # "overriding existing Objective-C class" and no window at all.
            class UtterWindowDelegate(AppKit.NSObject):
                def windowShouldClose_(self, _sender):
                    # Hide, never quit. This is a menu bar app; the icon stays.
                    outer._window.orderOut_(None)
                    return False

                def pickModel_(self, sender):
                    ok, message = outer.daemon.switch_provider(
                        str(sender.selectedItem().representedObject())
                    )
                    if not ok:
                        outer._say(message)
                    outer._refresh()

                def pickLanguage_(self, sender):
                    value = str(sender.selectedItem().representedObject())
                    outer.daemon.config.dictate_language = None if value == "auto" else value
                    outer.daemon._save_config()
                    outer._refresh()

                def pickPolish_(self, sender):
                    value = str(sender.selectedItem().representedObject())
                    if value == "off":
                        outer.daemon.set_polish(False)
                    else:
                        on, why = outer.daemon.set_polish(True, value)
                        if not on:
                            outer._say(why)
                    outer._refresh()

                def toggleStream_(self, sender):
                    outer.daemon.config.stream_while_speaking = bool(sender.state())
                    outer.daemon._save_config()
                    outer._refresh()

                def openVocabulary_(self, _sender):
                    import subprocess

                    from backend.config import config_path

                    subprocess.run(["open", "-t", str(config_path())], check=False)

                def openArchive_(self, _sender):
                    import subprocess

                    subprocess.run(
                        ["open", "-R", str(outer.daemon.scratchpad.archive.path)], check=False
                    )

                def runDoctor_(self, _sender):
                    outer._doctor()

                def quit_(self, _sender):
                    if outer.on_quit:
                        outer.on_quit()
                    AppKit.NSApp().terminate_(None)

            self._delegate = UtterWindowDelegate.alloc().init()

            rows = 7
            height = MARGIN * 2 + ROW_HEIGHT * rows + 96
            window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                AppKit.NSMakeRect(0, 0, WIDTH, height),
                AppKit.NSWindowStyleMaskTitled
                | AppKit.NSWindowStyleMaskClosable
                | AppKit.NSWindowStyleMaskMiniaturizable,
                AppKit.NSBackingStoreBuffered,
                False,
            )
            window.setTitle_("Utter")
            window.setDelegate_(self._delegate)
            window.setReleasedWhenClosed_(False)
            window.center()

            content = window.contentView()
            y = height - MARGIN - 20

            self._status = self._label(content, MARGIN, y, WIDTH - MARGIN * 2, "", bold=True)
            y -= 34

            self._hotkeys = self._label(
                content, MARGIN, y, WIDTH - MARGIN * 2, "", secondary=True
            )
            y -= 30
            self._separator(content, y + 8)
            y -= 12

            self._models = self._row_popup(content, y, "模型", "pickModel:")
            y -= ROW_HEIGHT
            self._languages = self._row_popup(content, y, "语言", "pickLanguage:")
            y -= ROW_HEIGHT
            self._polish = self._row_popup(content, y, "润色", "pickPolish:")
            y -= ROW_HEIGHT
            self._stream = self._row_check(content, y, "边说边出字", "toggleStream:")
            y -= ROW_HEIGHT + 6

            self._vocab = self._label(content, MARGIN, y, WIDTH - MARGIN * 2, "", secondary=True)
            y -= 34

            self._separator(content, y + 12)
            self._button(content, MARGIN, y - 16, 92, "自检…", "runDoctor:")
            self._button(content, MARGIN + 100, y - 16, 92, "打开存档", "openArchive:")
            self._button(content, MARGIN + 200, y - 16, 108, "编辑术语表", "openVocabulary:")
            self._button(content, WIDTH - MARGIN - 92, y - 16, 92, "退出 Utter", "quit:")

            self._window = window
            return True
        except Exception:
            log.warning("could not build the main window", exc_info=True)
            return False

    # -- little builders --

    def _label(self, parent, x, y, width, text, *, bold=False, secondary=False):
        import AppKit

        field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(x, y, width, 20)
        )
        field.setStringValue_(text)
        field.setBezeled_(False)
        field.setDrawsBackground_(False)
        field.setEditable_(False)
        field.setSelectable_(False)
        if bold:
            field.setFont_(AppKit.NSFont.boldSystemFontOfSize_(13))
        if secondary:
            field.setFont_(AppKit.NSFont.systemFontOfSize_(11))
            field.setTextColor_(AppKit.NSColor.secondaryLabelColor())
        parent.addSubview_(field)
        return field

    def _separator(self, parent, y):
        import AppKit

        line = AppKit.NSBox.alloc().initWithFrame_(
            AppKit.NSMakeRect(MARGIN, y, WIDTH - MARGIN * 2, 1)
        )
        line.setBoxType_(AppKit.NSBoxSeparator)
        parent.addSubview_(line)

    def _row_popup(self, parent, y, caption, action):
        import AppKit

        label = self._label(parent, MARGIN, y + 2, LABEL_WIDTH, caption)
        label.setAlignment_(AppKit.NSTextAlignmentRight)

        popup = AppKit.NSPopUpButton.alloc().initWithFrame_pullsDown_(
            AppKit.NSMakeRect(MARGIN + LABEL_WIDTH + 12, y,
                              WIDTH - MARGIN * 2 - LABEL_WIDTH - 12, 25),
            False,
        )
        popup.setTarget_(self._delegate)
        popup.setAction_(action)
        parent.addSubview_(popup)
        return popup

    def _row_check(self, parent, y, caption, action):
        import AppKit

        box = AppKit.NSButton.alloc().initWithFrame_(
            AppKit.NSMakeRect(MARGIN + LABEL_WIDTH + 12, y,
                              WIDTH - MARGIN * 2 - LABEL_WIDTH - 12, 22)
        )
        box.setButtonType_(AppKit.NSButtonTypeSwitch)
        box.setTitle_(caption)
        box.setTarget_(self._delegate)
        box.setAction_(action)
        parent.addSubview_(box)
        return box

    def _button(self, parent, x, y, width, title, action):
        import AppKit

        button = AppKit.NSButton.alloc().initWithFrame_(
            AppKit.NSMakeRect(x, y, width, 28)
        )
        button.setTitle_(title)
        button.setBezelStyle_(AppKit.NSBezelStyleRounded)
        button.setTarget_(self._delegate)
        button.setAction_(action)
        parent.addSubview_(button)
        return button

    # -- state --

    def _refresh(self) -> None:
        config = self.daemon.config

        running = getattr(self.daemon, "running", False)
        self._status.setStringValue_("● 就绪" if running else "○ 正在启动…")

        self._hotkeys.setStringValue_(
            f"按住 {config.hotkey_push or '未设置'} 说一句　　"
            f"双击 {config.hotkey_toggle or '未设置'} 长段口述"
        )

        self._fill(self._models, self._provider_choices(),
                   getattr(self.daemon.stt, "id", None))
        self._fill(self._languages,
                   [("auto", "自动检测"), ("zh", "中文"), ("en", "English")],
                   config.dictate_language or "auto")
        self._fill(self._polish, [
            ("off", "关"),
            ("light", "轻 —— 只补标点"),
            ("medium", "中 —— 标点 + 删口水词"),
            ("heavy", "重 —— 标点 + 口水词 + 分段"),
        ], config.polish_level if config.polish_enabled else "off")

        self._stream.setState_(1 if config.stream_while_speaking else 0)

        count = len(config.vocabulary)
        self._vocab.setStringValue_(
            f"术语表 {count} 条　—　这些词会在转录时被优先认出，"
            "改错的词只能在这里改，润色不会动你的字"
        )

    def _fill(self, popup, choices, selected) -> None:
        popup.removeAllItems()
        for value, title in choices:
            popup.addItemWithTitle_(title)
            popup.lastItem().setRepresentedObject_(value)
            if value == selected:
                popup.selectItem_(popup.lastItem())

    def _provider_choices(self):
        from backend.providers.stt import probe_all

        return [
            (s.id, s.display_name if s.available else f"{s.display_name}（不可用）")
            for s in probe_all()
        ]

    # -- actions that need more than a control --

    def _say(self, message: str) -> None:
        import AppKit

        panel = AppKit.NSAlert.alloc().init()
        panel.setMessageText_("Utter")
        panel.setInformativeText_(message)
        panel.addButtonWithTitle_("好")
        panel.beginSheetModalForWindow_completionHandler_(self._window, None)

    def _doctor(self) -> None:
        """`utter doctor`, in a text file. Every diagnostic this project has
        built lived behind a terminal command, and the point of the window is
        that there is no terminal."""
        import subprocess
        import sys
        import threading

        from backend.config import DEFAULT_DIR

        def run():
            report = DEFAULT_DIR / "doctor.txt"
            try:
                out = subprocess.run(
                    [sys.executable, "-m", "backend.cli", "doctor"],
                    capture_output=True, text=True, timeout=180,
                ).stdout
                report.write_text(out or "(没有输出)", encoding="utf-8")
                subprocess.run(["open", "-t", str(report)], check=False)
            except Exception as exc:  # pragma: no cover
                log.warning("doctor failed: %s", exc)

        threading.Thread(target=run, daemon=True, name="utter-doctor").start()
