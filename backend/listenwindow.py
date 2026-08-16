"""听记窗口 — the transcript, and as little else as possible.

The author's brief: the middle should be the record, everything else should get
out of the way, it must not stutter, and it should feel like a bookmark rather
than a control panel.

## Why this is an NSTextView and not an NSTableView

A table was the obvious choice and is the wrong one. Apple's own forums have
standing reports that `usesAutomaticRowHeights` flickers on dynamic tables with
many rows, that `reloadData(forRowIndexes:)` lays out subviews wrongly with it
enabled, and that inserting rows of differing heights makes a scrolled table
jump. A meeting is hundreds of rows of differing heights being appended live —
every one of those reports describes exactly our case.

A text view has none of those problems. Appending to its storage is cheap,
there are no row heights to compute, scrolling is the system's, and selecting
and copying text comes free — which the author asked for and which a table
would have needed extra work to fake.

The cost is that a translation arriving late has to be spliced into the middle
of the text rather than handed to a row. That is one range lookup and one
replacement, and it is bounded work; the table's problems were not.
"""

from __future__ import annotations

import logging

from backend.overlay import _on_main

log = logging.getLogger(__name__)

WIDTH = 560
HEIGHT = 640

#: The teal from the icon, so the window belongs to the same application.
ACCENT = (0.10, 0.72, 0.68)

WAITING = "…"


class ListenWindow:
    """The live transcript. `append` and `translated` are safe from any thread."""

    def __init__(self, on_stop=None, on_source_change=None):
        self.on_stop = on_stop
        self.on_source_change = on_source_change
        self._window = None
        self._text = None
        self._clock = None
        self._delegate = None
        #: entry index -> the range in the text holding its translation
        self._ranges: dict[int, tuple[int, int]] = {}
        self._order: list[int] = []

    # -- public ------------------------------------------------------------

    def show(self) -> None:
        def run():
            if self._window is None and not self._build():
                return
            import AppKit

            AppKit.NSApp().setActivationPolicy_(
                AppKit.NSApplicationActivationPolicyRegular)
            self._window.makeKeyAndOrderFront_(None)
            AppKit.NSApp().activateIgnoringOtherApps_(True)

        _on_main(run)

    def append(self, index: int, source: str, translation: str | None = None) -> None:
        """Add one utterance. The Chinese may arrive later; see `translated`."""
        def run():
            if self._text is None:
                return
            import AppKit

            storage = self._text.textStorage()
            at_bottom = self._at_bottom()

            storage.beginEditing()
            storage.appendAttributedString_(self._styled(source + "\n", english=True))
            start = storage.length()
            body = translation or WAITING
            storage.appendAttributedString_(
                self._styled(body + "\n\n", english=False, faded=not translation))
            storage.endEditing()

            self._ranges[index] = (start, len(body) + 1)   # +1 for the newline
            self._order.append(index)
            if at_bottom:
                self._scroll_to_bottom()

        _on_main(run)

    def translated(self, index: int, text: str) -> None:
        """Fill in a translation that arrived after its line was drawn."""
        def run():
            if self._text is None or index not in self._ranges:
                return
            import AppKit

            start, length = self._ranges[index]
            storage = self._text.textStorage()
            if start + length > storage.length():
                return

            at_bottom = self._at_bottom()
            replacement = self._styled(text + "\n", english=False)
            storage.beginEditing()
            storage.replaceCharactersInRange_withAttributedString_(
                AppKit.NSMakeRange(start, length), replacement)
            storage.endEditing()

            # Everything after this entry shifted by the difference in length.
            delta = replacement.length() - length
            self._ranges[index] = (start, replacement.length())
            if delta:
                for other in self._order:
                    if other != index and self._ranges[other][0] > start:
                        s, n = self._ranges[other]
                        self._ranges[other] = (s + delta, n)
            if at_bottom:
                self._scroll_to_bottom()

        _on_main(run)

    def set_clock(self, text: str) -> None:
        def run():
            if self._clock is not None:
                self._clock.setStringValue_(text)

        _on_main(run)

    # -- scrolling ---------------------------------------------------------

    def _at_bottom(self, slack: float = 40.0) -> bool:
        """Is the reader already at the end?

        This is the whole of the scrolling policy. Always scrolling would yank
        the view out from under someone reading back over what was just said —
        during a meeting, which is when they can least afford to lose their
        place. Never scrolling would mean the live transcript does not follow
        the speaker. So: follow only if they are already at the bottom.
        """
        try:
            clip = self._text.enclosingScrollView().contentView()
            visible = clip.documentVisibleRect()
            total = self._text.frame().size.height
            return visible.origin.y + visible.size.height >= total - slack
        except Exception:  # pragma: no cover - defensive
            return True

    def _scroll_to_bottom(self) -> None:
        try:
            self._text.scrollRangeToVisible_(
                __import__("AppKit").NSMakeRange(self._text.textStorage().length(), 0))
        except Exception:  # pragma: no cover
            pass

    # -- text styling ------------------------------------------------------

    def _styled(self, text: str, *, english: bool, faded: bool = False):
        import AppKit

        paragraph = AppKit.NSMutableParagraphStyle.alloc().init()
        # Generous leading. A transcript is read in long runs, not scanned.
        paragraph.setLineSpacing_(3.0)
        paragraph.setParagraphSpacing_(0.0)

        if english:
            font = AppKit.NSFont.systemFontOfSize_(14)
            colour = AppKit.NSColor.labelColor()
        else:
            font = AppKit.NSFont.systemFontOfSize_(14)
            colour = (AppKit.NSColor.tertiaryLabelColor() if faded
                      else AppKit.NSColor.secondaryLabelColor())

        return AppKit.NSAttributedString.alloc().initWithString_attributes_(
            text, {
                AppKit.NSFontAttributeName: font,
                AppKit.NSForegroundColorAttributeName: colour,
                AppKit.NSParagraphStyleAttributeName: paragraph,
            })

    # -- build -------------------------------------------------------------

    #: Registered once per process. Objective-C has a single flat class
    #: namespace, so defining this class a second time raises "overriding
    #: existing Objective-C class" and takes the window with it. This project
    #: has been bitten by that once already, when the settings window and the
    #: menu bar both wanted a class called `_Delegate`.
    _delegate_class_cache = None

    def _delegate_class(self):
        if ListenWindow._delegate_class_cache is not None:
            return ListenWindow._delegate_class_cache

        import AppKit

        class UtterListenDelegate(AppKit.NSObject):
            def windowShouldClose_(self, _s):
                # Closing hides. The meeting is still being recorded, and the
                # menu bar says so.
                self.owner._window.orderOut_(None)
                AppKit.NSApp().setActivationPolicy_(
                    AppKit.NSApplicationActivationPolicyAccessory)
                return False

            def stop_(self, _s):
                if self.owner.on_stop:
                    self.owner.on_stop()

            def toggleFloat_(self, sender):
                on = bool(sender.state())
                self.owner._window.setLevel_(
                    AppKit.NSFloatingWindowLevel if on else AppKit.NSNormalWindowLevel)

        UtterListenDelegate.owner = None
        ListenWindow._delegate_class_cache = UtterListenDelegate
        return UtterListenDelegate

    def _build(self) -> bool:
        try:
            import AppKit

            if self._delegate is None:
                self._delegate = self._delegate_class().alloc().init()
                self._delegate.owner = self

            window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                AppKit.NSMakeRect(0, 0, WIDTH, HEIGHT),
                AppKit.NSWindowStyleMaskTitled
                | AppKit.NSWindowStyleMaskClosable
                | AppKit.NSWindowStyleMaskMiniaturizable
                | AppKit.NSWindowStyleMaskResizable
                | AppKit.NSWindowStyleMaskFullSizeContentView,
                AppKit.NSBackingStoreBuffered, False)
            window.setTitle_("听记")
            window.setTitlebarAppearsTransparent_(True)
            window.setTitleVisibility_(AppKit.NSWindowTitleHidden)
            window.setMovableByWindowBackground_(True)
            window.setDelegate_(self._delegate)
            window.setReleasedWhenClosed_(False)
            window.setMinSize_(AppKit.NSMakeSize(380, 300))

            blur = AppKit.NSVisualEffectView.alloc().init()
            blur.setMaterial_(AppKit.NSVisualEffectMaterialUnderWindowBackground)
            blur.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
            blur.setState_(AppKit.NSVisualEffectStateActive)
            window.setContentView_(blur)

            bar = self._toolbar()
            scroll, text = self._transcript()
            self._text = text

            for view in (bar, scroll):
                view.setTranslatesAutoresizingMaskIntoConstraints_(False)
                blur.addSubview_(view)

            AppKit.NSLayoutConstraint.activateConstraints_([
                bar.topAnchor().constraintEqualToAnchor_constant_(blur.topAnchor(), 30),
                bar.leadingAnchor().constraintEqualToAnchor_constant_(
                    blur.leadingAnchor(), 16),
                bar.trailingAnchor().constraintEqualToAnchor_constant_(
                    blur.trailingAnchor(), -16),
                bar.heightAnchor().constraintEqualToConstant_(28),

                scroll.topAnchor().constraintEqualToAnchor_constant_(
                    bar.bottomAnchor(), 10),
                scroll.leadingAnchor().constraintEqualToAnchor_(blur.leadingAnchor()),
                scroll.trailingAnchor().constraintEqualToAnchor_(blur.trailingAnchor()),
                scroll.bottomAnchor().constraintEqualToAnchor_(blur.bottomAnchor()),
            ])

            window.center()
            window.setFrameAutosaveName_("UtterListen")
            self._window = window
            return True
        except Exception:
            log.warning("听记窗口建不起来", exc_info=True)
            return False

    def _toolbar(self):
        """Four things. Everything else lives behind the ⋯."""
        import AppKit

        row = AppKit.NSStackView.alloc().init()
        row.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationHorizontal)
        row.setAlignment_(AppKit.NSLayoutAttributeCenterY)
        row.setSpacing_(10)

        dot = AppKit.NSTextField.labelWithString_("●")
        dot.setTextColor_(AppKit.NSColor.systemRedColor())
        dot.setFont_(AppKit.NSFont.systemFontOfSize_(10))
        row.addArrangedSubview_(dot)

        self._clock = AppKit.NSTextField.labelWithString_("00:00")
        # Monospaced digits, so the clock does not jitter as the numbers change.
        self._clock.setFont_(
            AppKit.NSFont.monospacedDigitSystemFontOfSize_weight_(13, AppKit.NSFontWeightRegular))
        self._clock.setTextColor_(AppKit.NSColor.secondaryLabelColor())
        row.addArrangedSubview_(self._clock)

        spacer = AppKit.NSView.alloc().init()
        spacer.setContentHuggingPriority_forOrientation_(
            1, AppKit.NSLayoutConstraintOrientationHorizontal)
        row.addArrangedSubview_(spacer)

        stop = AppKit.NSButton.buttonWithTitle_target_action_(
            "结束", self._delegate, "stop_")
        stop.setBezelStyle_(AppKit.NSBezelStyleRounded)
        row.addArrangedSubview_(stop)

        more = AppKit.NSPopUpButton.alloc().initWithFrame_pullsDown_(
            AppKit.NSMakeRect(0, 0, 34, 24), True)
        more.addItemWithTitle_("⋯")
        float_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "保持在最前面", "toggleFloat:", "")
        float_item.setTarget_(self._delegate)
        more.menu().addItem_(float_item)
        row.addArrangedSubview_(more)
        return row

    def _transcript(self):
        import AppKit

        scroll = AppKit.NSScrollView.alloc().init()
        scroll.setHasVerticalScroller_(True)
        scroll.setDrawsBackground_(False)      # let the frost through
        scroll.setAutohidesScrollers_(True)

        text = AppKit.NSTextView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, WIDTH, HEIGHT))
        text.setEditable_(False)
        # Selectable is the point: the author asked to be able to copy a line
        # mid-meeting, and none of it touches the audio thread.
        text.setSelectable_(True)
        text.setDrawsBackground_(False)
        text.setTextContainerInset_(AppKit.NSMakeSize(20, 14))

        # The full incantation for "wrap to the window and grow downwards".
        # Without every line of it the text view keeps its initial width and
        # long sentences simply run off the right edge — which is exactly what
        # the first render showed: `...makes the translator invisib`.
        huge = 1.0e7
        text.setMinSize_(AppKit.NSMakeSize(0.0, 0.0))
        text.setMaxSize_(AppKit.NSMakeSize(huge, huge))
        text.setVerticallyResizable_(True)
        text.setHorizontallyResizable_(False)
        text.setAutoresizingMask_(AppKit.NSViewWidthSizable)
        container = text.textContainer()
        container.setContainerSize_(AppKit.NSMakeSize(WIDTH, huge))
        container.setWidthTracksTextView_(True)

        scroll.setDocumentView_(text)
        # After setDocumentView_, not before: measured, the text view came out
        # 1120 wide inside a 560 clip view, so it laid out against a 1080pt
        # container and every long line ran off the right edge. Pinning the
        # width to the clip view is what makes it wrap; the autoresizing mask
        # then keeps it pinned as the window is resized.
        text.setFrameSize_(AppKit.NSMakeSize(
            scroll.contentView().bounds().size.width, HEIGHT))
        return scroll, text
