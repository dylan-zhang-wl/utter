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


class TranscriptPane:
    """The live transcript as a view, so it can live in the main window.

    Split out of the standalone window once the author pointed out the obvious:
    the application now does two things, and starting the second one from a
    right-click on the menu bar is not where anybody would look for it. The
    transcript belongs on a page of the main window next to the settings.

    `append` and `translated` are safe to call from any thread.
    """

    _delegate_class_cache = None

    def __init__(self, on_toggle=None):
        self.on_toggle = on_toggle
        self._view = None
        self._text = None
        self._clock = None
        self._dot = None
        self._source = None
        self._choices = []
        self._button = None
        self._pause = None
        self._status = None
        self.on_pause = None
        self._delegate = None
        #: entry index -> the range in the text holding its translation
        self._ranges: dict[int, tuple[int, int]] = {}
        self._order: list[int] = []
        #: index -> (english, chinese|None). Kept so changing the font or the
        #: contrast can redraw the transcript instead of only affecting what
        #: arrives next.
        self._said: dict[int, tuple[str, str | None]] = {}
        #: How many characters of speculative preview sit at the very end of
        #: the storage. Always last, so it never disturbs a committed entry's
        #: offsets — and always removed before anything is committed.
        self._preview_len = 0
        self.font_size = 14.0
        self.high_contrast = False
        #: Called when the reader changes the type, so it can be remembered.
        self.on_appearance = None
        self._type = None
        self._contrast_item = None

    # -- public ------------------------------------------------------------

    def append(self, index: int, source: str, translation: str | None = None) -> None:
        """Add one utterance. The Chinese may arrive later; see `translated`."""
        def run():
            if self._text is None:
                return
            import AppKit

            self._sync_width()
            storage = self._text.textStorage()
            at_bottom = self._at_bottom()

            storage.beginEditing()
            self._drop_preview(storage)
            storage.appendAttributedString_(self._styled(source + "\n", english=True))
            start = storage.length()
            body = translation or WAITING
            storage.appendAttributedString_(
                self._styled(body + "\n", english=False, faded=not translation))
            storage.endEditing()
            self._said[index] = (source, translation)

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
            english = self._said.get(index, ("", None))[0]
            self._said[index] = (english, text)
            if delta:
                for other in self._order:
                    if other != index and self._ranges[other][0] > start:
                        s, n = self._ranges[other]
                        self._ranges[other] = (s + delta, n)
            if at_bottom:
                self._scroll_to_bottom()

        _on_main(run)

    def preview(self, text: str) -> None:
        """The grey tail: what a small model thinks is being said right now.

        It is speculative, so it is drawn as speculation — dimmed, with an
        ellipsis — and it lives at the very end of the storage where it cannot
        shift any committed entry's offsets. `append` removes it before writing
        a real sentence, so the two can never both claim the same stretch.
        """
        def run():
            if self._text is None:
                return
            import AppKit

            storage = self._text.textStorage()
            at_bottom = self._at_bottom()
            body = (text or "").strip()
            storage.beginEditing()
            self._drop_preview(storage)
            if body:
                drawn = self._styled(body + "…\n", english=True, faded=True)
                storage.appendAttributedString_(drawn)
                self._preview_len = drawn.length()
            storage.endEditing()
            if at_bottom:
                self._scroll_to_bottom()

        _on_main(run)

    def _drop_preview(self, storage) -> None:
        """Remove the speculative tail. Caller holds the edit."""
        import AppKit

        if not self._preview_len:
            return
        start = storage.length() - self._preview_len
        if start >= 0:
            storage.deleteCharactersInRange_(
                AppKit.NSMakeRange(start, self._preview_len))
        self._preview_len = 0

    def set_appearance(self, *, font_size=None, high_contrast=None) -> None:
        """Change the type, and redraw what is already on screen.

        Applying a new size only to what arrives next would leave a transcript
        in two sizes, which is worse than not offering the control.
        """
        def run():
            if font_size is not None:
                self.font_size = max(10.0, min(28.0, float(font_size)))
            if high_contrast is not None:
                self.high_contrast = bool(high_contrast)
            self._redraw()
            if self._contrast_item is not None:
                self._contrast_item.setState_(1 if self.high_contrast else 0)
            if self.on_appearance:
                self.on_appearance(self.font_size, self.high_contrast)

        _on_main(run)

    def _redraw(self) -> None:
        """Rebuild the transcript with the current type."""
        import AppKit

        if self._text is None:
            return
        at_bottom = self._at_bottom()
        storage = self._text.textStorage()
        storage.beginEditing()
        storage.setAttributedString_(
            AppKit.NSAttributedString.alloc().initWithString_(""))
        ranges = {}
        for index in self._order:
            english, chinese = self._said.get(index, ("", None))
            storage.appendAttributedString_(self._styled(english + "\n", english=True))
            start = storage.length()
            body = chinese or WAITING
            storage.appendAttributedString_(
                self._styled(body + "\n", english=False, faded=not chinese))
            ranges[index] = (start, len(body) + 1)
        storage.endEditing()
        self._preview_len = 0      # the redraw wrote committed text only
        self._ranges = ranges
        if at_bottom:
            self._scroll_to_bottom()

    def set_clock(self, text: str) -> None:
        def run():
            if self._clock is not None:
                self._clock.setStringValue_(text)

        _on_main(run)

    def _sync_width(self) -> None:
        """Keep the text as wide as its clip view, and no wider.

        The document view of a scroll view does not follow the clip view on its
        own, and its initial frame is set before the window has laid anything
        out — measured at 980pt of text inside a 460pt window, running off the
        right edge. Doing it here is O(1) and self-healing: the next line of the
        meeting fixes the width after a resize.
        """
        try:
            import AppKit

            clip = self._text.enclosingScrollView().contentView().frame().size.width
            if clip > 1 and abs(self._text.frame().size.width - clip) > 0.5:
                self._text.setFrameSize_(AppKit.NSMakeSize(
                    clip, self._text.frame().size.height))
        except Exception:  # pragma: no cover - defensive
            pass

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
        # A blank line between entries cost half the screen — five sentences
        # visible at a time, and a fast speaker outruns that instantly. The
        # separation is now spacing *after* the Chinese line, which reads the
        # same and takes a third of the room.
        paragraph.setLineSpacing_(2.0)
        paragraph.setParagraphSpacing_(0.0 if english else self.font_size * 0.7)

        if english:
            font = AppKit.NSFont.systemFontOfSize_(self.font_size)
            colour = AppKit.NSColor.labelColor()
        else:
            # Songti for the Chinese. One typeface change does more for the
            # 书卷气 the author asked for than any amount of ornament, and it
            # also separates the translation from the transcript at a glance
            # without a second colour.
            size = self.font_size + 1        # serif reads small at the same size
            font = (AppKit.NSFont.fontWithName_size_("Songti SC", size)
                    or AppKit.NSFont.fontWithName_size_("STSong", size)
                    or AppKit.NSFont.systemFontOfSize_(self.font_size))
            if faded:
                colour = AppKit.NSColor.tertiaryLabelColor()
            elif self.high_contrast:
                colour = AppKit.NSColor.labelColor()
            else:
                colour = AppKit.NSColor.secondaryLabelColor()
            paragraph.setLineSpacing_(3.0)

        return AppKit.NSAttributedString.alloc().initWithString_attributes_(
            text, {
                AppKit.NSFontAttributeName: font,
                AppKit.NSForegroundColorAttributeName: colour,
                AppKit.NSParagraphStyleAttributeName: paragraph,
            })

    # -- the view ----------------------------------------------------------

    def view(self):
        """The 听记 page: a control row and the transcript under it."""
        import AppKit

        if self._view is not None:
            return self._view
        if self._delegate is None:
            self._delegate = self._delegate_class().alloc().init()
            self._delegate.owner = self

        holder = AppKit.NSView.alloc().init()
        holder.setTranslatesAutoresizingMaskIntoConstraints_(False)

        controls = self._controls()
        self._row = controls
        scroll, text = self._transcript()
        self._text = text
        for sub in (controls, scroll):
            sub.setTranslatesAutoresizingMaskIntoConstraints_(False)
            holder.addSubview_(sub)

        AppKit.NSLayoutConstraint.activateConstraints_([
            controls.topAnchor().constraintEqualToAnchor_(holder.topAnchor()),
            controls.leadingAnchor().constraintEqualToAnchor_constant_(
                holder.leadingAnchor(), 20),
            controls.trailingAnchor().constraintEqualToAnchor_constant_(
                holder.trailingAnchor(), -20),
            controls.heightAnchor().constraintEqualToConstant_(28),

            scroll.topAnchor().constraintEqualToAnchor_constant_(
                controls.bottomAnchor(), 10),
            scroll.leadingAnchor().constraintEqualToAnchor_(holder.leadingAnchor()),
            scroll.trailingAnchor().constraintEqualToAnchor_(holder.trailingAnchor()),
            scroll.bottomAnchor().constraintEqualToAnchor_(holder.bottomAnchor()),
        ])
        self._view = holder
        self._show_idle()
        return holder

    def _controls(self):
        """Idle: pick a source and start. Running: a clock and 结束.

        One row that changes rather than two rows where one is always dead —
        the page has exactly one thing to offer at any moment.
        """
        import AppKit

        row = AppKit.NSStackView.alloc().init()
        row.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationHorizontal)
        row.setAlignment_(AppKit.NSLayoutAttributeCenterY)
        row.setSpacing_(10)

        self._source = AppKit.NSPopUpButton.alloc().initWithFrame_pullsDown_(
            AppKit.NSMakeRect(0, 0, 230, 25), False)
        row.addArrangedSubview_(self._source)
        self.refresh_sources()

        # 朱砂, not the system red: it is the colour of a seal, and it is the
        # one spot of colour in the window.
        self._dot = AppKit.NSTextField.labelWithString_("■")
        self._dot.setTextColor_(AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
            0.78, 0.20, 0.16, 1.0))
        self._dot.setFont_(AppKit.NSFont.systemFontOfSize_(9))
        row.addArrangedSubview_(self._dot)

        self._clock = AppKit.NSTextField.labelWithString_("00:00")
        # Monospaced digits, so the clock does not jitter as it counts.
        self._clock.setFont_(AppKit.NSFont.monospacedDigitSystemFontOfSize_weight_(
            13, AppKit.NSFontWeightRegular))
        self._clock.setTextColor_(AppKit.NSColor.secondaryLabelColor())
        row.addArrangedSubview_(self._clock)

        spacer = AppKit.NSView.alloc().init()
        spacer.setContentHuggingPriority_forOrientation_(
            1, AppKit.NSLayoutConstraintOrientationHorizontal)
        row.addArrangedSubview_(spacer)

        # Type controls behind one Aa button rather than three in the row: at
        # 360pt the bookmark has no width to spare, and NSStackView answers an
        # overfull row by clipping a view — which could be 结束.
        self._type = AppKit.NSPopUpButton.alloc().initWithFrame_pullsDown_(
            AppKit.NSMakeRect(0, 0, 50, 24), True)
        self._type.setBezelStyle_(AppKit.NSBezelStyleRounded)
        self._type.setControlSize_(AppKit.NSControlSizeSmall)
        menu = AppKit.NSMenu.alloc().init()
        menu.addItemWithTitle_action_keyEquivalent_("Aa", None, "")
        smaller = menu.addItemWithTitle_action_keyEquivalent_(
            "缩小字号", "smaller:", "-")
        bigger = menu.addItemWithTitle_action_keyEquivalent_(
            "放大字号", "bigger:", "+")
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        contrast = menu.addItemWithTitle_action_keyEquivalent_(
            "高对比度译文", "contrast:", "")
        for item in (smaller, bigger, contrast):
            item.setTarget_(self._delegate)
        self._contrast_item = contrast
        self._type.setMenu_(menu)
        row.addArrangedSubview_(self._type)

        self._status = AppKit.NSTextField.labelWithString_("")
        self._status.setFont_(AppKit.NSFont.systemFontOfSize_(11))
        self._status.setTextColor_(AppKit.NSColor.secondaryLabelColor())
        row.addArrangedSubview_(self._status)

        self._pause = AppKit.NSButton.buttonWithTitle_target_action_(
            "暂停", self._delegate, "pause:")
        self._pause.setBezelStyle_(AppKit.NSBezelStyleRounded)
        row.addArrangedSubview_(self._pause)

        self._button = AppKit.NSButton.buttonWithTitle_target_action_(
            "开始听记", self._delegate, "toggle:")
        self._button.setBezelStyle_(AppKit.NSBezelStyleRounded)
        self._button.setKeyEquivalent_("\r")     # Return starts a meeting
        row.addArrangedSubview_(self._button)
        return row

    def _show_idle(self) -> None:
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        if not running:
            self.preview("")
        if self._button is None:
            return
        self._button.setTitle_("结束听记" if running else "开始听记")
        self._button.setKeyEquivalent_("" if running else "\r")
        self._source.setHidden_(running)
        self._dot.setHidden_(not running)
        self._clock.setHidden_(not running)
        if self._pause is not None:
            self._pause.setHidden_(not running)
            self._pause.setTitle_("暂停")

    def set_running(self, running: bool) -> None:
        _on_main(lambda: self._set_running(running))

    def set_status_line(self, text) -> None:
        """A word about what is happening, where the clock is.

        Ending a meeting runs several model round trips for the summary, and
        until now the window said nothing at all through them — which reads as
        the application having ignored 结束.
        """
        def run():
            if self._status is not None:
                self._status.setStringValue_(text or "")

        _on_main(run)

    def set_preparing(self, preparing: bool) -> None:
        """Loading the model and opening a device takes seconds. Say so.

        A button that stays on 开始听记 while several seconds pass reads as a
        button that did nothing — which is exactly how the author described it.
        """
        def run():
            if self._button is None:
                return
            self._button.setEnabled_(not preparing)
            if preparing:
                self._button.setTitle_("正在准备…")

        _on_main(run)

    def refresh_sources(self) -> None:
        """List what this machine actually has, right now.

        The first version offered 「线下会议（麦克风）」 and 「线上会议（系统音频）」,
        which is a category the user has to translate into a device. The author
        asked for the real thing instead: the inputs macOS currently reports,
        plus system audio when this machine can do it — and nothing that is not
        actually there. Unplug the headphones and the next open shows the
        change.
        """
        from backend import audio_source
        from backend.system_audio import SystemAudioSource

        self._choices = []
        self._source.removeAllItems()

        try:
            audio_source.refresh_devices()   # PortAudio enumerates once at init
            shared = audio_source.shared_with_output()
            for device in audio_source.list_devices():
                mark = "  ⚠ 也是扬声器" if device.name == shared else ""
                self._source.addItemWithTitle_(f"{device.name}{mark}")
                self._choices.append(("mic", device.index))
        except Exception:
            log.warning("列不出输入设备", exc_info=True)

        ok, _why = SystemAudioSource.available()
        if ok:
            self._source.addItemWithTitle_("这台电脑正在播的声音")
            self._choices.append(("system", None))

        if not self._choices:
            self._source.addItemWithTitle_("找不到可用的音频输入")
            self._source.setEnabled_(False)

    @property
    def source(self) -> tuple[str, object]:
        """(kind, device index) for whatever is selected."""
        try:
            return self._choices[self._source.indexOfSelectedItem()]
        except Exception:
            return ("mic", None)

    @property
    def source_kind(self) -> str:
        return self.source[0]

    def clear(self) -> None:
        def run():
            if self._text is not None:
                self._text.textStorage().setAttributedString_(
                    __import__("AppKit").NSAttributedString.alloc().initWithString_(""))
            self._ranges.clear()
            self._order.clear()

        _on_main(run)

    def _delegate_class(self):
        import AppKit

        if TranscriptPane._delegate_class_cache is not None:
            return TranscriptPane._delegate_class_cache

        class UtterTranscriptDelegate(AppKit.NSObject):
            def smaller_(self, _s):
                self.owner.set_appearance(font_size=self.owner.font_size - 1)

            def bigger_(self, _s):
                self.owner.set_appearance(font_size=self.owner.font_size + 1)

            def contrast_(self, _s):
                self.owner.set_appearance(
                    high_contrast=not self.owner.high_contrast)

            def pause_(self, sender):
                try:
                    owner = self.owner
                    if owner.on_pause:
                        paused = owner.on_pause()
                        sender.setTitle_("继续" if paused else "暂停")
                except Exception:
                    log.warning("暂停/继续出错", exc_info=True)

            def toggle_(self, _s):
                # Guarded for the reason in window.py: AppKit swallows whatever
                # an action handler raises, so an unguarded failure here is a
                # button that silently does nothing.
                try:
                    owner = self.owner
                    if owner.on_toggle:
                        owner.on_toggle()
                except Exception:
                    log.warning("开始/结束听记出错", exc_info=True)

        UtterTranscriptDelegate.owner = None
        TranscriptPane._delegate_class_cache = UtterTranscriptDelegate
        return UtterTranscriptDelegate

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
        # long sentences run off the right edge — which is what the first
        # render showed: `...makes the translator invisib`.
        huge = 1.0e7
        text.setMinSize_(AppKit.NSMakeSize(0.0, 0.0))
        text.setMaxSize_(AppKit.NSMakeSize(huge, huge))
        text.setVerticallyResizable_(True)
        text.setHorizontallyResizable_(False)
        text.setAutoresizingMask_(AppKit.NSViewWidthSizable)
        container = text.textContainer()
        container.setContainerSize_(AppKit.NSMakeSize(WIDTH, huge))
        container.setWidthTracksTextView_(True)

        # A sheet of paper, not a void. Warm in light mode, barely-there in
        # dark: the point is that the transcript reads as a page rather than as
        # a terminal, which is what "书签" was asking for.
        paper = AppKit.NSBox.alloc().init()
        paper.setBoxType_(AppKit.NSBoxCustom)
        paper.setTitlePosition_(AppKit.NSNoTitle)
        paper.setCornerRadius_(8.0)
        paper.setBorderWidth_(0.0)
        paper.setFillColor_(AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
            0.99, 0.98, 0.95, 0.55))
        paper.setContentViewMargins_(AppKit.NSMakeSize(0, 0))
        self._paper = paper

        # Lay out only what is on screen. Without this TextKit re-flows the
        # entire document on every width change, and a meeting's transcript
        # only grows: measured at 150 entries a resize already cost 11-21ms,
        # and a resize gesture sends a stream of those. Non-contiguous layout
        # is the supported way to make a long, append-only document behave.
        text.layoutManager().setAllowsNonContiguousLayout_(True)

        scroll.setDocumentView_(text)
        # After setDocumentView_, not before: measured, the text view came out
        # 1120 wide inside a 560 clip view and laid out against a 1080pt
        # container, so every long line ran off the right.
        text.setFrameSize_(AppKit.NSMakeSize(
            scroll.contentView().bounds().size.width or WIDTH, HEIGHT))
        return scroll, text
