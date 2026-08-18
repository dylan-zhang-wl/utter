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

Rows are **anchored to both edges** — label at the leading margin, control at
the trailing one — with hairlines between them. The version before this gave
every control one derived width so their edges would line up, which they did,
and it looked wrong anyway: a 382pt pop-up containing the single character 关.
Apple's forms size a control to its content and line up the *right* edge. The
lesson is that "even" is a property of the margins, not of the controls.
"""

from __future__ import annotations

import logging

from backend.overlay import _on_main

log = logging.getLogger(__name__)

WIDTH = 500
PAD = 20          # window margin
ROW_PAD = 14      # leading/trailing inset inside a card
ROW_H = 38        # System Settings' row height
CARD_GAP = 18
ALERT_BODY_W = 300   # a sheet's usable width; wider makes long lines hard to scan

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
        self._timer = None
        self._warnings = []
        #: The 听记 page. The application does two things now, and starting the
        #: second from a right-click on the menu bar is not where anyone would
        #: look — so both live here, one segmented control apart.
        self.listen_pane = None
        self._pages = None
        self._tabs = None
        #: Set by the app: start/stop and pause/resume from the 听记 page.
        self.on_listen_toggle = None
        self.on_listen_pause = None
        #: Where the window was before it became a bookmark.
        self._normal_frame = None
        #: The size the bookmark was last left at. Now that the window can be
        #: resized, snapping back to 360×560 every meeting would undo the
        #: adjustment on every start.
        self._bookmark_size = None

    def menu_target(self):
        """An object responding to `openWindow:`, for the ⌘, menu item.

        Built eagerly rather than on first show, because the shortcut has to
        work before the window has ever been opened.
        """
        if self._delegate is None:
            self._delegate = self._delegate_class().alloc().init()
        return self._delegate

    def show(self) -> None:
        def run():
            if self._window is None and not self._build():
                return
            self._refresh()
            import AppKit

            # A Dock tile while there is a window to click on, and none while
            # there is not. An Accessory app never gets one, which is right for
            # the resident daemon and wrong the moment it puts a window on
            # screen: an app with a visible window and no Dock icon cannot be
            # reached by ⌘Tab and looks like it is not running.
            AppKit.NSApp().setActivationPolicy_(
                AppKit.NSApplicationActivationPolicyRegular)
            self._window.makeKeyAndOrderFront_(None)
            # Activate on the NEXT run-loop turn, not this one.
            #
            # setActivationPolicy_ from Accessory to Regular does not take
            # effect until the run loop turns, so activating immediately after
            # it lands while the app is still an accessory — and an accessory
            # app is not "active", which means the main menu never receives key
            # equivalents. The window appeared and ⌘W did nothing until the
            # user clicked on it. Measured: frontmost=false with a window
            # visible; frontmost=true and ⌘W working once activation happened
            # a beat later.
            def activate():
                AppKit.NSApp().activateIgnoringOtherApps_(True)
                self._window.makeKeyAndOrderFront_(None)

            AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                0.0, False, lambda _t: activate())
            self._tick(True)
            # A window that is ordered front and still not visible is the
            # failure this line exists to catch; it cost a session once.
            log.info("settings window: visible=%s frame=%s",
                     self._window.isVisible(), self._window.frame())

        _on_main(run)

    def _select_page(self, name: str) -> None:
        if not self._pages:
            return
        for key, page in self._pages.items():
            page.setHidden_(key != name)

    def show_listen(self) -> None:
        """Bring the window up on the 听记 page."""
        def run():
            if self._window is None and not self._build():
                return
            if self._tabs is not None:
                self._tabs.setSelectedSegment_(0)
            self._select_page("listen")

        _on_main(run)
        self.show()

    # -- the controller talks to the pane through us -----------------------

    def append(self, index, source, translation=None):
        if self.listen_pane:
            self.listen_pane.append(index, source, translation)

    def translated(self, index, text):
        if self.listen_pane:
            self.listen_pane.translated(index, text)

    def set_clock(self, text):
        if self.listen_pane:
            self.listen_pane.set_clock(text)

    def preview(self, text):
        if self.listen_pane:
            self.listen_pane.preview(text)

    #: The bookmark: narrow, tall, and parked at the top right while a meeting
    #: runs. The author's use is watching subtitles beside something else, so
    #: the full-width settings window is the wrong shape for the only moment it
    #: is actually being read.
    BOOKMARK_SIZE = (360, 560)
    BOOKMARK_MARGIN = 24

    def set_listening(self, running: bool):
        if self.listen_pane:
            self.listen_pane.set_running(running)
        _on_main(lambda: self._set_bookmark(running))

    def _set_bookmark(self, on: bool) -> None:
        """Shrink to a bookmark while recording; restore afterwards."""
        import AppKit

        if self._window is None:
            return
        try:
            if on:
                if self._normal_frame is None:
                    self._normal_frame = self._window.frame()
                screen = (self._window.screen() or AppKit.NSScreen.mainScreen()
                          ).visibleFrame()
                width, height = self._bookmark_size or self.BOOKMARK_SIZE
                frame = AppKit.NSMakeRect(
                    screen.origin.x + screen.size.width - width - self.BOOKMARK_MARGIN,
                    screen.origin.y + screen.size.height - height - self.BOOKMARK_MARGIN,
                    width, height)
                self._window.setFrame_display_animate_(frame, True, True)
                if self._tabs is not None:
                    self._tabs.setHidden_(True)   # one page while recording
            else:
                current = self._window.frame().size
                self._bookmark_size = (current.width, current.height)
                if self._tabs is not None:
                    self._tabs.setHidden_(False)
                if self._normal_frame is not None:
                    self._window.setFrame_display_animate_(
                        self._normal_frame, True, True)
                    self._normal_frame = None
        except Exception:
            log.warning("切换书签形态出错", exc_info=True)

    def set_preparing(self, preparing: bool):
        if self.listen_pane:
            self.listen_pane.set_preparing(preparing)

    def _remember_appearance(self, size, contrast) -> None:
        """Type is a reading preference, not a per-meeting one."""
        self.daemon.config.listen_font_size = size
        self.daemon.config.listen_high_contrast = contrast
        try:
            self.daemon._save_config()
        except Exception:
            log.warning("字号设置没能存下来", exc_info=True)

    def ask(self, title: str, message: str, *, yes: str, no: str) -> bool:
        """A yes/no sheet, answered synchronously from a background thread."""
        import threading

        answer = {"yes": True}
        done = threading.Event()

        def run():
            import AppKit

            try:
                alert = AppKit.NSAlert.alloc().init()
                alert.setMessageText_(title)
                alert.setInformativeText_(message)
                alert.addButtonWithTitle_(yes)
                alert.addButtonWithTitle_(no)
                answer["yes"] = (alert.runModal() == AppKit.NSAlertFirstButtonReturn)
            except Exception:
                log.warning("提问失败，按默认继续", exc_info=True)
            finally:
                done.set()

        _on_main(run)
        done.wait(120)
        return answer["yes"]

    def set_font_size(self, size):
        if self.listen_pane:
            self.listen_pane.set_appearance(font_size=size)

    def set_status_line(self, text):
        if self.listen_pane:
            self.listen_pane.set_status_line(text)

    def say(self, title: str, message: str, *, settings_url: str | None = None) -> None:
        """A sheet on the main window, from any thread.

        `settings_url` puts the user one click from the pane that fixes it.
        Telling somebody the name of a checkbox is not the same as showing it
        to them, and macOS's privacy panes are not where their names suggest.
        """
        def run():
            import AppKit

            if self._window is None:
                return
            alert = AppKit.NSAlert.alloc().init()
            alert.setMessageText_(title)
            alert.setInformativeText_(message)
            alert.addButtonWithTitle_("打开设置" if settings_url else "好")
            if settings_url:
                alert.addButtonWithTitle_("以后再说")

            def done(response):
                if settings_url and response == AppKit.NSAlertFirstButtonReturn:
                    AppKit.NSWorkspace.sharedWorkspace().openURL_(
                        AppKit.NSURL.URLWithString_(settings_url))

            alert.beginSheetModalForWindow_completionHandler_(self._window, done)

        _on_main(run)

    # -- build ------------------------------------------------------------

    #: Registered once per process. Objective-C has one flat class namespace,
    #: so a second MainWindow raised "overriding existing Objective-C class"
    #: and _build() swallowed it and returned False — a window that silently
    #: refuses to open. The delegate therefore holds its owner rather than
    #: closing over it.
    _delegate_class_cache = None

    def _delegate_class(self):
        """The window's target for every control, registered once."""
        if MainWindow._delegate_class_cache is not None:
            return MainWindow._delegate_class_cache

        import AppKit

        outer = self  # rebound below for later instances

        def guarded(fn):
            """Log what AppKit would otherwise swallow.

            An exception raised inside an action handler does not reach the
            console, the log, or the user: the event loop eats it and the
            control simply appears inert. That has now cost two debugging
            rounds — a selector typo, then a missing forwarder — with
            identical symptoms and an empty log both times. A control that
            fails must say so.
            """
            def wrapped(self, sender=None):
                try:
                    return fn(self, sender)
                except Exception:
                    log.warning("界面动作 %s 出错", fn.__name__, exc_info=True)
                    return None
            wrapped.__name__ = fn.__name__
            return wrapped

        class UtterWindowDelegate(AppKit.NSObject):
            def windowShouldClose_(self, _s):
                outer._window.orderOut_(None)   # hide; ⌘Q is how you quit
                outer._tick(False)
                # Back to the menu bar, and the Dock tile goes with it.
                AppKit.NSApp().setActivationPolicy_(
                    AppKit.NSApplicationActivationPolicyAccessory)
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

            def pickPage_(self, sender):
                outer._select_page(
                    "listen" if sender.selectedSegment() == 0 else "settings")

            def openWindow_(self, _s):
                outer.show()

            def openArchive_(self, _s):
                import subprocess
                subprocess.run(
                    ["open", "-R", str(outer.daemon.scratchpad.archive.path)], check=False)

            def showWarnings_(self, _s):
                outer._show_warnings()

            def runDoctor_(self, _s):
                outer._doctor()

            def quit_(self, _s):
                if outer.on_quit:
                    outer.on_quit()
                AppKit.NSApp().terminate_(None)

        MainWindow._delegate_class_cache = UtterWindowDelegate
        return UtterWindowDelegate

    def _build(self) -> bool:
        """Auto Layout, not arithmetic.

        The first version positioned every control by subtracting from a
        running `y`, and it looked exactly like that: rows a pixel or two out,
        labels not sharing a baseline, the gap above one card different from
        the gap above the next. Stacks do the spacing; the margins are stated
        once and everything else is measured from them.
        """
        try:
            import AppKit

            if self._delegate is None:
                self._delegate = self._delegate_class().alloc().init()

            window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                AppKit.NSMakeRect(0, 0, WIDTH, 100),
                AppKit.NSWindowStyleMaskTitled
                | AppKit.NSWindowStyleMaskClosable
                | AppKit.NSWindowStyleMaskMiniaturizable
                # Resizable, which a settings window would not be and this one
                # has to be: it holds a live transcript now, and how much of a
                # meeting fits on screen is the whole point of the bookmark.
                # The author reported that dragging the edge was not smooth.
                # The truth was that there was no edge to drag.
                | AppKit.NSWindowStyleMaskResizable
                | AppKit.NSWindowStyleMaskFullSizeContentView,
                AppKit.NSBackingStoreBuffered, False)
            window.setTitle_("Utter")
            window.setTitlebarAppearsTransparent_(True)
            window.setTitleVisibility_(AppKit.NSWindowTitleHidden)
            window.setMovableByWindowBackground_(True)
            # The bookmark width, which every row here is known to survive.
            # Narrower and the toolbar starts dropping buttons rather than
            # getting narrower, and the button it drops could be 结束.
            window.setMinSize_(AppKit.NSMakeSize(self.BOOKMARK_SIZE[0], 360))
            window.setDelegate_(self._delegate)
            window.setReleasedWhenClosed_(False)

            blur = AppKit.NSVisualEffectView.alloc().init()
            blur.setMaterial_(AppKit.NSVisualEffectMaterialUnderWindowBackground)
            blur.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
            blur.setState_(AppKit.NSVisualEffectStateActive)
            window.setContentView_(blur)

            column = AppKit.NSStackView.alloc().init()
            column.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationVertical)
            # Width, not Leading. A leading-aligned vertical stack sizes each
            # child to its own content, so the 听写 card (which holds a long
            # device name) came out wider than the 润色 card, and the cards
            # themselves were the ragged edge — not the controls inside them.
            column.setAlignment_(AppKit.NSLayoutAttributeWidth)
            column.setSpacing_(CARD_GAP)
            column.setTranslatesAutoresizingMaskIntoConstraints_(False)
            blur.addSubview_(column)
            AppKit.NSLayoutConstraint.activateConstraints_([

            ])

            def stacked(view):
                """Add a full-width block to the column.

                The explicit width is not redundant with the stack's Width
                alignment: alignment alone let the 听写 card settle at 336pt
                beside two 460pt neighbours, because the alignment constraints
                lose to the card's own content. Saying the width outright is
                the thing that is even every time.
                """
                column.addArrangedSubview_(view)
                view.widthAnchor().constraintEqualToAnchor_(
                    column.widthAnchor()).setActive_(True)

            stacked(self._header())

            self._models = self._popup("pickModel:")
            self._langs = self._popup("pickLanguage:")
            self._devices = self._popup("pickDevice:")
            self._stream = self._switch("toggleStream:")
            stacked(self._card("听写", [
                ("模型", self._models),
                ("语言", self._langs),
                ("麦克风", self._devices),
                ("边说边出字", self._stream),
            ]))

            self._polish = self._popup("pickPolish:")
            self._key = self._secure_row()
            stacked(self._card("润色", [
                ("档位", self._polish),
                ("API Key", self._key_row),
            ]))

            self._vocab = self._text("", size=13)
            stacked(self._card("术语表", [
                ("词条", self._pair(self._vocab, self._button("编辑…", "editVocabulary:"))),
            ]))

            stacked(self._footer())

            # --- two pages, one window -----------------------------------
            settings_page = AppKit.NSView.alloc().init()
            settings_page.setTranslatesAutoresizingMaskIntoConstraints_(False)
            column.removeFromSuperview()
            settings_page.addSubview_(column)
            AppKit.NSLayoutConstraint.activateConstraints_([
                column.topAnchor().constraintEqualToAnchor_(settings_page.topAnchor()),
                column.leadingAnchor().constraintEqualToAnchor_constant_(
                    settings_page.leadingAnchor(), PAD),
                # Equal, not less-than, and no fixed width: a pinned 460pt
                # column silently set a 500pt floor on the whole window, which
                # is why 开始听记 hid the tabs and then failed to shrink into a
                # bookmark. Tracking the page instead lets the window narrow.
                column.trailingAnchor().constraintEqualToAnchor_constant_(
                    settings_page.trailingAnchor(), -PAD),
            ])

            from backend.listenwindow import TranscriptPane

            if self.listen_pane is None:
                self.listen_pane = TranscriptPane(
                    on_toggle=lambda: (self.on_listen_toggle or (lambda: None))())
                self.listen_pane.on_pause = lambda: (
                    self.on_listen_pause() if self.on_listen_pause else False)
                config = self.daemon.config
                self.listen_pane.font_size = config.listen_font_size
                self.listen_pane.high_contrast = config.listen_high_contrast
                self.listen_pane.on_appearance = self._remember_appearance
            listen_page = self.listen_pane.view()

            self._pages = {"listen": listen_page, "settings": settings_page}
            for page in self._pages.values():
                page.setTranslatesAutoresizingMaskIntoConstraints_(False)
                blur.addSubview_(page)
                AppKit.NSLayoutConstraint.activateConstraints_([
                    page.topAnchor().constraintEqualToAnchor_constant_(
                        blur.topAnchor(), 64),
                    page.leadingAnchor().constraintEqualToAnchor_(blur.leadingAnchor()),
                    page.trailingAnchor().constraintEqualToAnchor_(blur.trailingAnchor()),
                    page.bottomAnchor().constraintEqualToAnchor_constant_(
                        blur.bottomAnchor(), -PAD),
                ])

            tabs = AppKit.NSSegmentedControl.alloc().init()
            tabs.setSegmentCount_(2)
            tabs.setLabel_forSegment_("听记", 0)
            tabs.setLabel_forSegment_("设置", 1)
            tabs.setSegmentStyle_(AppKit.NSSegmentStyleTexturedRounded)
            tabs.setTarget_(self._delegate)
            tabs.setAction_("pickPage:")
            tabs.setTranslatesAutoresizingMaskIntoConstraints_(False)
            blur.addSubview_(tabs)
            AppKit.NSLayoutConstraint.activateConstraints_([
                tabs.centerXAnchor().constraintEqualToAnchor_(blur.centerXAnchor()),
                tabs.topAnchor().constraintEqualToAnchor_constant_(blur.topAnchor(), 28),
                tabs.widthAnchor().constraintEqualToConstant_(180),
            ])
            self._tabs = tabs
            tabs.setSelectedSegment_(0)
            self._select_page("listen")

            window.setContentSize_(AppKit.NSMakeSize(WIDTH, 620))
            window.center()
            # Reopen where it was left. center() above is the first-run
            # position; after that AppKit restores the saved frame over it.
            window.setFrameAutosaveName_("UtterSettings")
            self._window = window
            return True
        except Exception:
            log.warning("could not build the main window", exc_info=True)
            return False

    # -- pieces -----------------------------------------------------------

    def _header(self):
        import AppKit

        row = AppKit.NSStackView.alloc().init()
        row.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationHorizontal)
        row.setSpacing_(10)

        badge = AppKit.NSImageView.alloc().init()
        image = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "waveform", "Utter")
        if image is not None:
            config = AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
                22, AppKit.NSFontWeightMedium)
            badge.setImage_(image.imageWithSymbolConfiguration_(config))
            badge.setContentTintColor_(
                AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(*ACCENT, 1.0))
        row.addArrangedSubview_(badge)

        titles = AppKit.NSStackView.alloc().init()
        titles.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationVertical)
        titles.setAlignment_(AppKit.NSLayoutAttributeLeading)
        titles.setSpacing_(1)
        titles.addArrangedSubview_(self._text("Utter", size=17, bold=True))
        self._hint = self._text("", size=11, secondary=True)
        titles.addArrangedSubview_(self._hint)
        row.addArrangedSubview_(titles)

        spacer = AppKit.NSView.alloc().init()
        row.addArrangedSubview_(spacer)
        # Two indicators, because they answer two questions. The dot says
        # whether the app is ready; the triangle says whether anything is off.
        # Folding them into one made 就绪 turn orange for a user whose AirPods
        # are legitimately both the microphone and the speakers — which is a
        # note, not a reason to look broken.
        #
        # A button rather than a label, so the explanation has two ways out: a
        # tooltip on hover, and a sheet on click. A tooltip alone hides the one
        # thing the indicator exists to say behind a gesture nobody is told
        # about.
        self._warn = AppKit.NSButton.alloc().init()
        self._warn.setBezelStyle_(AppKit.NSBezelStyleInline)
        self._warn.setBordered_(False)
        self._warn.setTitle_("")
        self._warn.setTarget_(self._delegate)
        self._warn.setAction_("showWarnings:")
        symbol = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "exclamationmark.triangle.fill", "有需要注意的地方")
        if symbol is not None:
            self._warn.setImage_(symbol)
            self._warn.setContentTintColor_(AppKit.NSColor.systemOrangeColor())
        else:  # pragma: no cover - SF Symbols missing is not worth a blank button
            self._warn.setTitle_("⚠")
        row.addArrangedSubview_(self._warn)
        self._status = self._text("", size=11, secondary=True)
        row.addArrangedSubview_(self._status)
        return row

    def _card(self, caption, rows):
        """A captioned, rounded group of label/control rows.

        `NSBox`, not a layer-backed `NSView`. A CALayer background colour is a
        `CGColor`, which is a fixed set of numbers: it does not follow the
        system between light and dark, so the hand-mixed white-at-5% that looks
        right on a dark desktop is invisible on a light one. NSBox fills with
        an `NSColor` and repaints itself when the appearance changes.
        """
        import AppKit

        box = AppKit.NSStackView.alloc().init()
        box.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationVertical)
        box.setAlignment_(AppKit.NSLayoutAttributeLeading)
        box.setSpacing_(6)

        label = self._text(caption, size=11, secondary=True)
        label.setFont_(AppKit.NSFont.systemFontOfSize_weight_(11, AppKit.NSFontWeightSemibold))
        box.addArrangedSubview_(label)

        inner = AppKit.NSStackView.alloc().init()
        inner.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationVertical)
        # Width, so every row spans the card. Leading would size each row to
        # its own content and the hairlines would come out different lengths.
        inner.setAlignment_(AppKit.NSLayoutAttributeWidth)
        inner.setSpacing_(0)
        inner.setTranslatesAutoresizingMaskIntoConstraints_(False)
        for index, (caption_text, control) in enumerate(rows):
            if index:
                inner.addArrangedSubview_(self._hairline())
            inner.addArrangedSubview_(self._row(caption_text, control))

        card = AppKit.NSBox.alloc().init()
        card.setBoxType_(AppKit.NSBoxCustom)
        card.setTitlePosition_(AppKit.NSNoTitle)
        card.setCornerRadius_(10.0)
        card.setFillColor_(AppKit.NSColor.controlBackgroundColor())
        card.setBorderColor_(AppKit.NSColor.separatorColor())
        card.setBorderWidth_(1.0)
        card.setContentViewMargins_(AppKit.NSMakeSize(0, 0))
        card.setTranslatesAutoresizingMaskIntoConstraints_(False)
        card.setContentView_(inner)
        AppKit.NSLayoutConstraint.activateConstraints_([
            inner.topAnchor().constraintEqualToAnchor_(card.topAnchor()),
            inner.bottomAnchor().constraintEqualToAnchor_(card.bottomAnchor()),
            inner.leadingAnchor().constraintEqualToAnchor_(card.leadingAnchor()),
            inner.trailingAnchor().constraintEqualToAnchor_(card.trailingAnchor()),
        ])
        box.addArrangedSubview_(card)
        # Leading alignment positions the caption; it does not stretch the card,
        # so the card is pinned to the group's width outright.
        card.widthAnchor().constraintEqualToAnchor_(box.widthAnchor()).setActive_(True)
        return box

    def _row(self, caption, control):
        """One label/control line: label at the leading margin, control at the
        trailing one, whatever is between them empty."""
        import AppKit

        row = AppKit.NSStackView.alloc().init()
        row.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationHorizontal)
        row.setAlignment_(AppKit.NSLayoutAttributeCenterY)
        row.setSpacing_(10)
        row.setEdgeInsets_(AppKit.NSEdgeInsetsMake(0, ROW_PAD, 0, ROW_PAD))
        row.setTranslatesAutoresizingMaskIntoConstraints_(False)
        row.heightAnchor().constraintGreaterThanOrEqualToConstant_(ROW_H).setActive_(True)

        label = self._text(caption)
        row.addArrangedSubview_(label)
        row.addArrangedSubview_(self._spacer())
        row.addArrangedSubview_(control)

        # The label and the control keep their natural width; the gap between
        # them absorbs everything left over. Without this the stack distributes
        # the slack and neither edge lands on a margin.
        for view, priority in ((label, 750), (control, 750)):
            view.setContentHuggingPriority_forOrientation_(
                priority, AppKit.NSLayoutConstraintOrientationHorizontal)
        return row

    def _spacer(self):
        """An empty view that soaks up the slack in a row."""
        import AppKit

        view = AppKit.NSView.alloc().init()
        view.setContentHuggingPriority_forOrientation_(
            AppKit.NSLayoutPriorityDefaultLow - 50,
            AppKit.NSLayoutConstraintOrientationHorizontal)
        return view

    def _hairline(self):
        """A separator inset to the label, the way System Settings draws it."""
        import AppKit

        line = AppKit.NSBox.alloc().init()
        line.setBoxType_(AppKit.NSBoxSeparator)
        line.setTranslatesAutoresizingMaskIntoConstraints_(False)

        wrap = AppKit.NSStackView.alloc().init()
        wrap.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationHorizontal)
        wrap.setEdgeInsets_(AppKit.NSEdgeInsetsMake(0, ROW_PAD, 0, 0))
        wrap.addArrangedSubview_(line)
        return wrap

    def _footer(self):
        import AppKit

        row = AppKit.NSStackView.alloc().init()
        row.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationHorizontal)
        row.setSpacing_(8)
        row.addArrangedSubview_(self._button("自检", "runDoctor:"))
        row.addArrangedSubview_(self._button("存档", "openArchive:"))
        spacer = AppKit.NSView.alloc().init()
        row.addArrangedSubview_(spacer)
        row.addArrangedSubview_(self._button("退出 Utter", "quit:"))
        return row

    def _pair(self, left, right):
        """Two controls that belong together, side by side at the trailing
        margin — a field and the button that commits it."""
        import AppKit

        row = AppKit.NSStackView.alloc().init()
        row.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationHorizontal)
        row.setAlignment_(AppKit.NSLayoutAttributeCenterY)
        row.setSpacing_(8)
        row.addArrangedSubview_(left)
        row.addArrangedSubview_(right)
        return row

    def _text(self, s, *, size=13, bold=False, secondary=False):
        import AppKit

        f = AppKit.NSTextField.labelWithString_(s)
        f.setFont_(AppKit.NSFont.boldSystemFontOfSize_(size) if bold
                   else AppKit.NSFont.systemFontOfSize_(size))
        if secondary:
            f.setTextColor_(AppKit.NSColor.secondaryLabelColor())
        return f

    def _popup(self, action):
        import AppKit

        p = AppKit.NSPopUpButton.alloc().initWithFrame_pullsDown_(
            AppKit.NSMakeRect(0, 0, 120, 25), False)
        p.setTarget_(self._delegate)
        p.setAction_(action)
        # Each pop-up is as wide as its own longest item, so 关/轻/中/重 stays
        # small and the model names stay readable. The one thing that must not
        # happen is a device name pushing the control past the card, so it
        # truncates rather than resists: a name is recognisable from its front.
        p.cell().setLineBreakMode_(AppKit.NSLineBreakByTruncatingTail)
        p.setContentCompressionResistancePriority_forOrientation_(
            AppKit.NSLayoutPriorityDefaultLow,
            AppKit.NSLayoutConstraintOrientationHorizontal)
        return p

    def _switch(self, action):
        import AppKit

        try:
            s = AppKit.NSSwitch.alloc().init()  # noqa: F841 — replaced below on old macOS
        except AttributeError:  # pragma: no cover - very old macOS
            s = AppKit.NSButton.alloc().init()
            s.setButtonType_(AppKit.NSButtonTypeSwitch)
            s.setTitle_("")
        s.setTarget_(self._delegate)
        s.setAction_(action)
        return s

    def _secure_row(self):
        import AppKit

        field = AppKit.NSSecureTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, 220, 24))
        field.setBezelStyle_(AppKit.NSTextFieldRoundedBezel)
        field.setTranslatesAutoresizingMaskIntoConstraints_(False)
        # Wide enough that a pasted key looks like it landed somewhere. It
        # cannot show the key — it is a secure field — so the width is the only
        # feedback the field itself gives.
        field.widthAnchor().constraintEqualToConstant_(220).setActive_(True)
        self._key_row = self._pair(field, self._button("存入", "saveKey:"))
        return field

    def _button(self, title, action):
        import AppKit

        b = AppKit.NSButton.buttonWithTitle_target_action_(title, self._delegate, action)
        b.setBezelStyle_(AppKit.NSBezelStyleRounded)
        b.setControlSize_(AppKit.NSControlSizeRegular)
        return b

    # -- state ------------------------------------------------------------

    def _refresh(self) -> None:
        from backend.keyglyph import glyph

        c = self.daemon.config
        self._refresh_status()
        self._hint.setStringValue_(
            f"按住 {glyph(c.hotkey_push)} 说一句　·　双击 {glyph(c.hotkey_toggle)} 长段口述")

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
        # The Keychain item name is an implementation detail; what the user
        # needs to know is whether there is a key in there and, if they are
        # about to paste one, which service it will be used for. Ollama runs on
        # this machine and has no key at all — offering a field for one would
        # be inviting them to hunt for something that does not exist.
        local = c.llm_provider == "ollama"
        self._key.setEnabled_(not local)
        self._key.setPlaceholderString_(
            "本地 Ollama，不需要 key" if local
            else "已存入钥匙串　·　粘贴新的可替换" if secrets.get_secret(self._key_name())
            else f"粘贴 {self._service_name()} 的 key")
        self._vocab.setStringValue_(f"{len(c.vocabulary)} 条")

    def _refresh_status(self) -> None:
        """The dot, and only the dot.

        Separate from `_refresh` because it runs on a timer: rebuilding the
        pop-ups every two seconds would re-enumerate the audio devices and
        would slam shut a menu the user had just opened.
        """
        from backend.health import warnings_for

        ready = getattr(self.daemon, "running", False)
        if ready:
            self._dot("就绪", "green", "按住热键就能说")
        else:
            self._dot("启动中", "orange", "正在申请权限、载入模型")

        self._warnings = warnings_for(self.daemon, self.daemon.config) if ready else []
        self._warn.setHidden_(not self._warnings)
        self._warn.setToolTip_(
            "\n".join(note.short.lstrip("⚠ ") for note in self._warnings)
            + "\n（点一下看详情）" if self._warnings else None)

    def _dot(self, text, colour, tooltip) -> None:
        """A coloured bullet and a word. Green is the whole point of it: a grey
        ● said 就绪 and looked exactly like a ● that said 启动中."""
        import AppKit

        shown = f"● {text}"
        s = AppKit.NSMutableAttributedString.alloc().initWithString_(shown)
        s.addAttribute_value_range_(
            AppKit.NSForegroundColorAttributeName,
            AppKit.NSColor.systemGreenColor() if colour == "green"
            else AppKit.NSColor.systemOrangeColor(),
            AppKit.NSMakeRange(0, 1))
        s.addAttribute_value_range_(
            AppKit.NSForegroundColorAttributeName,
            AppKit.NSColor.secondaryLabelColor(),
            AppKit.NSMakeRange(1, len(shown) - 1))
        s.addAttribute_value_range_(
            AppKit.NSFontAttributeName, AppKit.NSFont.systemFontOfSize_(11),
            AppKit.NSMakeRange(0, len(shown)))
        self._status.setAttributedStringValue_(s)
        self._status.setToolTip_(tooltip)

    def _tick(self, on: bool) -> None:
        """Keep the dot honest while the window is open.

        Warm-up has been measured between 1.7 and 77 seconds. Opening the
        window during it and seeing 启动中 for ever — because nothing redraws
        it — would be the same silent failure in a new place.
        """
        import AppKit

        if on and self._timer is None:
            self._timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                2.0, True, lambda _t: self._refresh_status())
        elif not on and self._timer is not None:
            self._timer.invalidate()
            self._timer = None

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
        """The Keychain item. Not shown; used to store and to look up."""
        return {"vertex": "google_api_key", "gemini": "google_api_key"}.get(
            self.daemon.config.llm_provider, "openai_api_key")

    def _service_name(self) -> str:
        """The same thing said the way the user would say it."""
        return {"vertex": "Google", "gemini": "Google"}.get(
            self.daemon.config.llm_provider, "OpenAI")

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

    def _show_warnings(self) -> None:
        """The paragraph behind the triangle."""
        import AppKit

        if not self._warnings:
            return
        alert = AppKit.NSAlert.alloc().init()
        alert.setMessageText_(self._warnings[0].short.lstrip("⚠ ")
                              if len(self._warnings) == 1 else "有几处需要注意")

        # The body goes in an accessory view rather than in informativeText,
        # for one reason: `informativeText` is drawn at the system's 11pt and
        # there is no way to ask for anything else. A note nobody can read
        # comfortably is a note nobody reads.
        body = "\n\n".join(
            note.why if len(self._warnings) == 1
            else f"{note.short.lstrip('⚠ ')}\n{note.why}"
            for note in self._warnings)
        label = AppKit.NSTextField.wrappingLabelWithString_(body)
        label.setFont_(AppKit.NSFont.systemFontOfSize_(13))
        label.setPreferredMaxLayoutWidth_(ALERT_BODY_W)
        label.setFrameSize_(AppKit.NSMakeSize(ALERT_BODY_W, label.fittingSize().height))
        alert.setAccessoryView_(label)

        alert.addButtonWithTitle_("知道了")
        alert.beginSheetModalForWindow_completionHandler_(self._window, None)

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
