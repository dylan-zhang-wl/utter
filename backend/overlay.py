"""The floating indicator: a frosted pill that says whether Utter is hearing you.

Why it exists. Until now the only sign the daemon was listening was the system
microphone icon, which looks identical whether audio is arriving or the device
is dead. The author spent a day unable to tell "the model misheard me" from
"nothing was recorded", and several of those rounds would have been one glance.

Two layers, as everywhere else in this project. `BarModel` turns an audio level
into bar heights and is pure arithmetic — no AppKit, fully testable. `Overlay`
is the NSPanel around it.

## The panel must not take focus

`NSWindowStyleMaskNonactivatingPanel`, and `becomesKeyOnlyIfNeeded`. A panel
that becomes key steals focus from the window the author is dictating into, and
since injection targets whatever is frontmost, their text would be pasted into
this overlay instead of their document. FluidVoice #401 hit the milder version
of the same bug — an overlay stealing focus made ⌘Tab sluggish. Here it would
lose text, so it is checked by a test rather than trusted.

## Everything visible happens on the main thread

AppKit requires it, and our callers do not run there: hotkey events arrive on
pynput's listener thread and audio levels on the transcription worker. Every
public method hops to the main queue, so callers never have to think about it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

WIDTH, HEIGHT = 200.0, 44.0
BAR_COUNT = 22
BOTTOM_MARGIN = 120.0  # above the Dock, clear of most document windows


@dataclass
class BarModel:
    """Audio level in, bar heights out. Pure arithmetic, no AppKit.

    Three things the shape has to do, none of them decorative:

    * **Silence must look like silence.** At level 0 the bars collapse to a
      near-flat line. That is the whole point of the overlay — if the author
      speaks and the line stays flat, the microphone is not working, and they
      learn it in the moment instead of a second later from an empty result.
    * **The envelope is fixed, the amplitude moves.** Bars are tallest in the
      middle and taper to the ends, so the row reads as one shape rising and
      falling rather than twenty-two independent meters.
    * **Falling is slower than rising.** A meter that tracks the signal exactly
      flickers. Rising fast keeps it honest about onsets; falling slowly keeps
      it readable.
    """

    count: int = BAR_COUNT
    attack: float = 0.55  # how much of a rise is taken per frame
    release: float = 0.12  # and of a fall — deliberately much slower
    floor: float = 0.06  # bars never fully vanish; the row stays legible

    _level: float = 0.0
    _phase: float = 0.0
    _envelope: list[float] = field(default_factory=list)

    def __post_init__(self):
        # cos² across the row: 1.0 at the centre, tapering to ~0.35 at the ends.
        self._envelope = [
            0.35 + 0.65 * math.cos((i / (self.count - 1) - 0.5) * math.pi) ** 2
            for i in range(self.count)
        ]

    def feed(self, level: float) -> None:
        """Take one audio level, 0..1. Call once per frame."""
        level = max(0.0, min(1.0, float(level)))
        rate = self.attack if level > self._level else self.release
        self._level += (level - self._level) * rate
        self._phase += 0.35

    def reset(self) -> None:
        self._level = 0.0
        self._phase = 0.0

    def heights(self) -> list[float]:
        """Bar heights, 0..1, in drawing order."""
        out = []
        for i, env in enumerate(self._envelope):
            # A small per-bar wobble so the row breathes instead of moving as a
            # rigid block. Deterministic, not random — random flickers.
            wobble = 0.82 + 0.18 * math.sin(self._phase + i * 0.9)
            out.append(min(1.0, self.floor + self._level * env * wobble))
        return out

    @property
    def level(self) -> float:
        return self._level


def rms_to_level(samples) -> float:
    """Turn a chunk of float32 audio into a 0..1 level for the bars.

    Speech sits far below full scale, so a linear RMS would leave the bars
    almost flat at normal volume. The curve below maps ordinary speech across
    most of the range, which is what makes the display useful rather than
    technically correct.
    """
    import numpy as np

    if samples is None or len(samples) == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms <= 0:
        return 0.0
    db = 20 * math.log10(max(rms, 1e-6))
    return max(0.0, min(1.0, (db + 55.0) / 45.0))


# --- the AppKit half ----------------------------------------------------------


def _on_main(fn) -> None:
    """Run fn on the main thread. AppKit demands it; our callers are elsewhere."""
    try:
        from Foundation import NSOperationQueue, NSThread

        if NSThread.isMainThread():
            fn()
        else:
            NSOperationQueue.mainQueue().addOperationWithBlock_(fn)
    except Exception:  # pragma: no cover - non-macOS
        log.debug("could not dispatch to the main thread", exc_info=True)


def _build_view_class():
    """Defined lazily so importing this module needs no AppKit."""
    import AppKit
    import objc

    class _BarsView(AppKit.NSView):
        def initWithFrame_(self, frame):
            self = objc.super(_BarsView, self).initWithFrame_(frame)
            if self is None:
                return None
            self.model = BarModel()
            self.state = "idle"
            self.caption = ""
            return self

        def isFlipped(self):
            return True

        def drawRect_(self, _rect):
            bounds = self.bounds()
            w, h = bounds.size.width, bounds.size.height

            colour = {
                "recording": (0.20, 0.86, 0.72),
                "working": (0.45, 0.68, 0.95),
                "error": (0.92, 0.35, 0.34),
            }.get(self.state, (0.62, 0.62, 0.60))

            heights = self.model.heights()
            n = len(heights)
            slot = (w - 84) / n
            bar_w = max(1.5, slot * 0.42)
            mid, span = h / 2, h * 0.34
            left = 16.0

            for i, value in enumerate(heights):
                bar_h = max(1.5, value * span * 2)
                x = left + i * slot + (slot - bar_w) / 2
                AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    *colour, 0.55 + 0.45 * value
                ).set()
                AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    AppKit.NSMakeRect(x, mid - bar_h / 2, bar_w, bar_h),
                    bar_w / 2, bar_w / 2,
                ).fill()

            if self.caption:
                style = AppKit.NSMutableParagraphStyle.alloc().init()
                style.setAlignment_(AppKit.NSTextAlignmentRight)
                AppKit.NSString.stringWithString_(self.caption).drawInRect_withAttributes_(
                    AppKit.NSMakeRect(w - 76, mid - 8, 60, 16),
                    {
                        AppKit.NSFontAttributeName:
                            AppKit.NSFont.monospacedDigitSystemFontOfSize_weight_(11, 0),
                        AppKit.NSForegroundColorAttributeName:
                            AppKit.NSColor.secondaryLabelColor(),
                        AppKit.NSParagraphStyleAttributeName: style,
                    },
                )

    return _BarsView


class Overlay:
    """The floating pill. Hidden while idle; every method is thread-safe."""

    def __init__(self):
        self._panel = None
        self._view = None
        self._timer = None

    # -- construction --

    def _ensure(self) -> bool:
        if self._panel is not None:
            return True
        try:
            import AppKit

            screen = AppKit.NSScreen.mainScreen().frame()
            rect = AppKit.NSMakeRect(
                (screen.size.width - WIDTH) / 2, BOTTOM_MARGIN, WIDTH, HEIGHT
            )
            # NonactivatingPanel is the load-bearing flag. A panel that becomes
            # key steals focus, and injection targets whatever is frontmost —
            # the author's text would land in this overlay.
            panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                rect,
                AppKit.NSWindowStyleMaskBorderless
                | AppKit.NSWindowStyleMaskNonactivatingPanel,
                AppKit.NSBackingStoreBuffered,
                False,
            )
            panel.setLevel_(AppKit.NSStatusWindowLevel)
            panel.setOpaque_(False)
            panel.setBackgroundColor_(AppKit.NSColor.clearColor())
            panel.setHasShadow_(True)
            panel.setBecomesKeyOnlyIfNeeded_(True)
            panel.setIgnoresMouseEvents_(True)
            panel.setCollectionBehavior_(
                AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                | AppKit.NSWindowCollectionBehaviorStationary
                | AppKit.NSWindowCollectionBehaviorIgnoresCycle
            )

            blur = AppKit.NSVisualEffectView.alloc().initWithFrame_(
                AppKit.NSMakeRect(0, 0, WIDTH, HEIGHT)
            )
            blur.setMaterial_(AppKit.NSVisualEffectMaterialHUDWindow)
            blur.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
            blur.setState_(AppKit.NSVisualEffectStateActive)
            blur.setWantsLayer_(True)
            blur.layer().setCornerRadius_(HEIGHT / 2)
            blur.layer().setMasksToBounds_(True)

            view = _build_view_class().alloc().initWithFrame_(
                AppKit.NSMakeRect(0, 0, WIDTH, HEIGHT)
            )
            blur.addSubview_(view)
            panel.setContentView_(blur)

            self._panel, self._view = panel, view
            return True
        except Exception:
            log.warning("could not create the overlay panel", exc_info=True)
            return False

    # -- public, thread-safe --

    def show(self, state: str = "recording") -> None:
        def run():
            if not self._ensure():
                return
            self._view.state = state
            self._view.model.reset()
            self._panel.orderFrontRegardless()
            self._view.setNeedsDisplay_(True)

        _on_main(run)

    def set_state(self, state: str, caption: str = "") -> None:
        def run():
            if self._view is None:
                return
            self._view.state = state
            self._view.caption = caption
            self._view.setNeedsDisplay_(True)

        _on_main(run)

    def feed(self, level: float, caption: str = "") -> None:
        def run():
            if self._view is None:
                return
            self._view.model.feed(level)
            if caption:
                self._view.caption = caption
            self._view.setNeedsDisplay_(True)

        _on_main(run)

    def hide(self) -> None:
        def run():
            if self._panel is not None:
                self._panel.orderOut_(None)

        _on_main(run)
