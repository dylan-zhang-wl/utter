"""The main window: settings, not a wall of text.

The first version was a form with a paragraph of explanation under every
control, and it looked like a form. Two things were wrong with that.

It was ugly. Utter's other surface — the pill that appears while you speak — is
frosted, rounded and coloured, and a settings window in plain grey does not
look like part of the same application. This one uses the same vibrancy, the
same corner radius and the same teal as the icon.

And it explained too much. A settings pane is read a hundred times and every
sentence is read a hundred times; the terminology row does not need to justify
itself, it needs to say how many terms there are. Explanations belong in the
README, which is read once.

## Layout

Full-size content view under a transparent titlebar, an `NSVisualEffectView`
behind everything, and rounded cards for grouping — the shape macOS itself has
used for settings since Ventura. No custom controls: the point of a native
window is that it does not look like ours.
"""

from __future__ import annotations

import logging

from backend.overlay import _on_main

log = logging.getLogger(__name__)

WIDTH = 540
PAD = 20          # window margin
CARD_PAD = 14     # inside a card
ROW = 30
LABEL_W = 78

#: The icon's teal, so the window belongs to the same application as the icon
#: and the recording pill.
ACCENT = (0.10, 0.72, 0.68)


class MainWindow:
    """Utter's settings. Construct once; `show()` is safe from any thread."""

    def __init__(self, daemon, on_quit=None):
        self.daemon = daemon
        self.on_quit = on_quit
        self._window = None
        self._delegate = None

    def show(self) -> None:
        def run():
            if self._window is None and not self._build():
                return
            self._refresh()
            self._window.makeKeyAndOrderFront_(None)
            import AppKit

            AppKit.NSApp().activateIgnoringOtherApps_(True)

        _on_main(run)

    # -- build ------------------------------------------------------------

    def _build(self) -> bool:
        try:
            import AppKit

            outer = self

            class UtterWindowDelegate(AppKit.NSObject):
                def windowShouldClose_(self, _s):
                    outer._window.orderOut_(None)   # hide; the icon stays
                    return False

                def pickModel_(self, s):
                    ok, msg = outer.daemon.switch_provider(
                        str(s.selectedItem().representedObject()))
                    if not ok:
                        outer._say(msg)
                    outer._refresh()

                def pickLanguage_(self, s):
                    v = str(s.selectedItem().representedObject())
                    outer.daemon.config.dictate_language = None if v == "auto" else v
                    outer.daemon._save_config()

                def pickDevice_(self, s):
                    v = s.selectedItem().representedObject()
                    outer.daemon.config.input_device = None if v is None else int(v)
                    outer.daemon._save_config()
                    outer._refresh()

                def pickPolish_(self, s):
                    v = str(s.selectedItem().representedObject())
                    if v == "off":
                        outer.daemon.set_polish(False)
                    else:
                        on, why = outer.daemon.set_polish(True, v)
                        if not on:
                            outer._say(why)
                    outer._refresh()

                def toggleStream_(self, s):
                    outer.daemon.config.stream_while_speaking = bool(s.state())
                    outer.daemon._save_config()

                def saveKey_(self, _s):
                    outer._save_key()

                def editVocabulary_(self, _s):
                    outer._open_config()

                def openArchive_(self, _s):
                    import subprocess
                    subprocess.run(
                        ["open", "-R", str(outer.daemon.scratchpad.archive.path)],
                        check=False)

                def runDoctor_(self, _s):
                    outer._doctor()

                def quit_(self, _s):
                    if outer.on_quit:
                        outer.on_quit()
                    AppKit.NSApp().terminate_(None)

            self._delegate = UtterWindowDelegate.alloc().init()

            height = 520
            window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                AppKit.NSMakeRect(0, 0, WIDTH, height),
                AppKit.NSWindowStyleMaskTitled
                | AppKit.NSWindowStyleMaskClosable
                | AppKit.NSWindowStyleMaskFullSizeContentView,
                AppKit.NSBackingStoreBuffered, False)
            window.setTitle_("Utter")
            window.setTitlebarAppearsTransparent_(True)
            window.setTitleVisibility_(AppKit.NSWindowTitleHidden)
            window.setMovableByWindowBackground_(True)
            window.setDelegate_(self._delegate)
            window.setReleasedWhenClosed_(False)
            window.center()

            # Vibrancy behind everything, the same material the pill uses.
            blur = AppKit.NSVisualEffectView.alloc().initWithFrame_(
                AppKit.NSMakeRect(0, 0, WIDTH, height))
            blur.setMaterial_(AppKit.NSVisualEffectMaterialUnderWindowBackground)
            blur.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
            blur.setState_(AppKit.NSVisualEffectStateActive)
            blur.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
            window.setContentView_(blur)

            y = height - 52
            self._title = self._text(blur, PAD + 34, y, 240, "Utter", size=20, bold=True)
            self._badge(blur, PAD, y - 2)
            self._status = self._text(blur, WIDTH - PAD - 150, y + 2, 150, "",
                                      size=11, secondary=True, right=True)
            y -= 26
            self._hint = self._text(blur, PAD + 34, y, WIDTH - PAD * 2 - 34, "",
                                    size=11, secondary=True)
            y -= 22

            # -- 听写 --
            y = self._card(blur, y, 4 * ROW + CARD_PAD, "听写")
            self._models = self._popup(blur, y, "模型", "pickModel:"); y -= ROW
            self._langs = self._popup(blur, y, "语言", "pickLanguage:"); y -= ROW
            self._devices = self._popup(blur, y, "麦克风", "pickDevice:"); y -= ROW
            self._stream = self._switch(blur, y, "边说边出字", "toggleStream:")
            y -= ROW + CARD_PAD

            # -- 润色 --
            y = self._card(blur, y, 2 * ROW + CARD_PAD + 6, "润色")
            self._polish = self._popup(blur, y, "档位", "pickPolish:"); y -= ROW + 4
            self._key = self._secure(blur, y, "API Key", "saveKey:")
            y -= ROW + CARD_PAD

            # -- 术语表 --
            y = self._card(blur, y, ROW + 4, "术语表")
            self._vocab = self._text(blur, PAD + CARD_PAD, y + 4, 200, "", size=13)
            self._small(blur, WIDTH - PAD - CARD_PAD - 70, y, 70, "编辑…", "editVocabulary:")
            y -= ROW + CARD_PAD

            self._small(blur, PAD, PAD, 78, "自检", "runDoctor:")
            self._small(blur, PAD + 86, PAD, 78, "存档", "openArchive:")
            self._small(blur, WIDTH - PAD - 88, PAD, 88, "退出 Utter", "quit:")

            self._window = window
            return True
        except Exception:
            log.warning("could not build the main window", exc_info=True)
            return False

    # -- pieces -----------------------------------------------------------

    def _badge(self, parent, x, y):
        """The icon's waveform, small, in the accent colour."""
        import AppKit

        view = AppKit.NSImageView.alloc().initWithFrame_(AppKit.NSMakeRect(x, y, 26, 26))
        image = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "waveform", "Utter")
        if image is not None:
            config = AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
                20, AppKit.NSFontWeightMedium)
            view.setImage_(image.imageWithSymbolConfiguration_(config))
            view.setContentTintColor_(
                AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(*ACCENT, 1.0))
        parent.addSubview_(view)

    def _card(self, parent, top, height, caption) -> float:
        """A rounded translucent group, captioned. Returns the first row's y."""
        import AppKit

        label = self._text(parent, PAD + 2, top - 14, 200, caption, size=11, secondary=True)
        label.setFont_(AppKit.NSFont.systemFontOfSize_weight_(11, AppKit.NSFontWeightSemibold))

        box = AppKit.NSView.alloc().initWithFrame_(
            AppKit.NSMakeRect(PAD, top - 18 - height, WIDTH - PAD * 2, height))
        box.setWantsLayer_(True)
        layer = box.layer()
        layer.setCornerRadius_(10.0)
        layer.setBackgroundColor_(
            AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(1, 1, 1, 0.05).CGColor())
        layer.setBorderWidth_(1.0)
        layer.setBorderColor_(
            AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(1, 1, 1, 0.08).CGColor())
        parent.addSubview_(box)
        return top - 18 - CARD_PAD - 20

    def _text(self, parent, x, y, w, s, *, size=13, bold=False, secondary=False, right=False):
        import AppKit

        f = AppKit.NSTextField.alloc().initWithFrame_(AppKit.NSMakeRect(x, y, w, size + 8))
        f.setStringValue_(s)
        f.setBezeled_(False); f.setDrawsBackground_(False)
        f.setEditable_(False); f.setSelectable_(False)
        f.setFont_(AppKit.NSFont.boldSystemFontOfSize_(size) if bold
                   else AppKit.NSFont.systemFontOfSize_(size))
        if secondary:
            f.setTextColor_(AppKit.NSColor.secondaryLabelColor())
        if right:
            f.setAlignment_(AppKit.NSTextAlignmentRight)
        parent.addSubview_(f)
        return f

    def _popup(self, parent, y, caption, action):
        import AppKit

        lab = self._text(parent, PAD + CARD_PAD, y + 3, LABEL_W, caption, secondary=True)
        lab.setAlignment_(AppKit.NSTextAlignmentRight)
        x = PAD + CARD_PAD + LABEL_W + 10
        p = AppKit.NSPopUpButton.alloc().initWithFrame_pullsDown_(
            AppKit.NSMakeRect(x, y, WIDTH - PAD * 2 - CARD_PAD * 2 - LABEL_W - 10, 25), False)
        p.setTarget_(self._delegate); p.setAction_(action)
        p.setBezelStyle_(AppKit.NSBezelStyleRounded)
        parent.addSubview_(p)
        return p

    def _switch(self, parent, y, caption, action):
        import AppKit

        lab = self._text(parent, PAD + CARD_PAD, y + 3, LABEL_W, caption, secondary=True)
        lab.setAlignment_(AppKit.NSTextAlignmentRight)
        x = PAD + CARD_PAD + LABEL_W + 10
        try:
            s = AppKit.NSSwitch.alloc().initWithFrame_(AppKit.NSMakeRect(x, y, 40, 22))
        except AttributeError:  # pragma: no cover - very old macOS
            s = AppKit.NSButton.alloc().initWithFrame_(AppKit.NSMakeRect(x, y, 40, 22))
            s.setButtonType_(AppKit.NSButtonTypeSwitch)
        s.setTarget_(self._delegate); s.setAction_(action)
        parent.addSubview_(s)
        return s

    def _secure(self, parent, y, caption, action):
        import AppKit

        lab = self._text(parent, PAD + CARD_PAD, y + 3, LABEL_W, caption, secondary=True)
        lab.setAlignment_(AppKit.NSTextAlignmentRight)
        x = PAD + CARD_PAD + LABEL_W + 10
        w = WIDTH - PAD * 2 - CARD_PAD * 2 - LABEL_W - 10 - 74
        field = AppKit.NSSecureTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(x, y, w, 24))
        field.setBezelStyle_(AppKit.NSTextFieldRoundedBezel)
        parent.addSubview_(field)
        self._small(parent, x + w + 8, y - 2, 66, "存入", action)
        return field

    def _small(self, parent, x, y, w, title, action):
        import AppKit

        b = AppKit.NSButton.alloc().initWithFrame_(AppKit.NSMakeRect(x, y, w, 26))
        b.setTitle_(title)
        b.setBezelStyle_(AppKit.NSBezelStyleRounded)
        b.setControlSize_(AppKit.NSControlSizeSmall)
        b.setFont_(AppKit.NSFont.systemFontOfSize_(12))
        b.setTarget_(self._delegate); b.setAction_(action)
        parent.addSubview_(b)
        return b

    # -- state ------------------------------------------------------------

    def _refresh(self) -> None:
        c = self.daemon.config
        self._status.setStringValue_(
            "● 就绪" if getattr(self.daemon, "running", False) else "○ 启动中")
        self._hint.setStringValue_(
            f"按住 {c.hotkey_push or '—'}　说一句　·　双击 {c.hotkey_toggle or '—'}　长段口述")

        self._fill(self._models, self._providers(), getattr(self.daemon.stt, "id", None))
        self._fill(self._langs, [("auto", "自动"), ("zh", "中文"), ("en", "English")],
                   c.dictate_language or "auto")
        self._fill(self._devices, self._input_devices(),
                   None if c.input_device is None else str(c.input_device))
        self._fill(self._polish, [("off", "关"), ("light", "轻"),
                                  ("medium", "中"), ("heavy", "重")],
                   c.polish_level if c.polish_enabled else "off")
        self._stream.setState_(1 if c.stream_while_speaking else 0)

        from backend import secrets
        name = self._key_name()
        self._key.setPlaceholderString_(
            f"{name}　已存" if secrets.get_secret(name) else f"{name}　未设置")
        self._vocab.setStringValue_(f"{len(c.vocabulary)} 条")

    def _fill(self, popup, choices, selected) -> None:
        popup.removeAllItems()
        for value, title in choices:
            popup.addItemWithTitle_(title)
            popup.lastItem().setRepresentedObject_(value)
            if value == selected:
                popup.selectItem_(popup.lastItem())

    def _providers(self):
        from backend.providers.stt import probe_all

        return [(s.id, s.display_name if s.available else f"{s.display_name}（不可用）")
                for s in probe_all()]

    def _input_devices(self):
        """Real devices, with the shared-output problem marked rather than
        explained. One glyph says what a paragraph was saying."""
        from backend import audio_source

        try:
            shared = audio_source.shared_with_output()
            items = [(None, "跟随系统默认")]
            for d in audio_source.list_devices():
                mark = "  ⚠ 也是扬声器" if d.name == shared else ""
                items.append((str(d.index), f"{d.name}{mark}"))
            return items
        except Exception:  # pragma: no cover
            return [(None, "跟随系统默认")]

    def _key_name(self) -> str:
        return {"vertex": "google_api_key", "gemini": "google_api_key"}.get(
            self.daemon.config.llm_provider, "openai_api_key")

    # -- actions ----------------------------------------------------------

    def _save_key(self) -> None:
        """Straight into the Keychain, never onto disk (铁律 4)."""
        from backend import secrets

        value = str(self._key.stringValue()).strip()
        if not value:
            self._say("先粘贴 key 再点存入。")
            return
        name = self._key_name()
        secrets.set_secret(name, value)
        self._key.setStringValue_("")
        if secrets.get_secret(name) == value:
            self._say(f"已存进钥匙串（{len(value)} 个字符）。")
            self.daemon.set_polish(self.daemon.config.polish_enabled,
                                   self.daemon.config.polish_level)
        else:
            self._say("写进钥匙串后读回来对不上，没存成。")
        self._refresh()

    def _open_config(self) -> None:
        import subprocess

        from backend.config import config_path

        subprocess.run(["open", "-t", str(config_path())], check=False)

    def _say(self, message: str) -> None:
        import AppKit

        a = AppKit.NSAlert.alloc().init()
        a.setMessageText_("Utter")
        a.setInformativeText_(message)
        a.addButtonWithTitle_("好")
        a.beginSheetModalForWindow_completionHandler_(self._window, None)

    def _doctor(self) -> None:
        import subprocess
        import sys
        import threading

        from backend.config import DEFAULT_DIR

        def run():
            report = DEFAULT_DIR / "doctor.txt"
            try:
                out = subprocess.run([sys.executable, "-m", "backend.cli", "doctor"],
                                     capture_output=True, text=True, timeout=180).stdout
                report.write_text(out or "(没有输出)", encoding="utf-8")
                subprocess.run(["open", "-t", str(report)], check=False)
            except Exception as exc:  # pragma: no cover
                log.warning("doctor failed: %s", exc)

        threading.Thread(target=run, daemon=True, name="utter-doctor").start()
