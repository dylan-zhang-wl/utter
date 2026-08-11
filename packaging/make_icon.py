"""Draw Utter's application icon.

    ~/.venvs/utter/bin/python packaging/make_icon.py

Drawn with CoreGraphics rather than shipped as a PNG, because the shape is not
decoration — it is the same waveform the overlay draws while you are speaking,
built from the same `BarModel` arithmetic. The icon and the thing on screen
should be recognisably one object.

## Two icons, on purpose

The menu bar item is a *template* image: macOS forces it monochrome and
inverts it for light and dark bars, and every well-behaved status item works
that way. The application icon has the opposite job — it has to be findable in
the Dock, in Spotlight and in a Finder window full of other icons, so it is the
one that carries colour.

Same silhouette, different treatment. That is how Apple's own pairs work.

## The shape

macOS icons are a rounded rect with a continuous ("squircle") corner, inset
inside the 1024pt canvas so the shadow has room. The numbers below are Apple's
grid: content occupies 824 of 1024, corner radius is 22.37% of the content.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CANVAS = 1024
CONTENT = 824  # Apple's icon grid: the art does not reach the edges
CORNER = 0.2237  # of the content width — the continuous-corner ratio

# Teal, and not an arbitrary one: it is the colour the overlay turns while it
# is recording (backend/overlay.py). Seeing the icon and then seeing the pill
# light up in the same colour should feel like one thing.
TOP = (0.28, 0.92, 0.78)
BOTTOM = (0.05, 0.62, 0.72)


def _bars(count: int = 9) -> list[float]:
    """Heights 0..1, the same cos² envelope the overlay uses.

    Nine rather than the overlay's twenty-two: at icon sizes, twenty-two bars
    turn into a grey smear. The shape has to survive being 32 points wide in a
    Dock, so it is the same curve at a lower sample rate.
    """
    return [
        0.30 + 0.70 * math.cos((i / (count - 1) - 0.5) * math.pi) ** 2
        for i in range(count)
    ]


def draw(path: Path, size: int) -> None:
    import AppKit
    import Quartz

    scale = size / CANVAS
    image = AppKit.NSImage.alloc().initWithSize_(AppKit.NSMakeSize(size, size))
    image.lockFocus()

    context = AppKit.NSGraphicsContext.currentContext().CGContext()
    Quartz.CGContextSetAllowsAntialiasing(context, True)
    Quartz.CGContextSetInterpolationQuality(context, Quartz.kCGInterpolationHigh)

    inset = (CANVAS - CONTENT) / 2 * scale
    side = CONTENT * scale
    radius = CONTENT * CORNER * scale
    body = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
        AppKit.NSMakeRect(inset, inset, side, side), radius, radius
    )

    # Vertical gradient, light at the top, the way macOS lights its own icons.
    AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(
        AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(*TOP, 1.0),
        AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(*BOTTOM, 1.0),
    ).drawInBezierPath_angle_(body, -90.0)

    # A hairline of white along the top edge. Every Apple icon has one; without
    # it a flat gradient reads as a sticker rather than an object.
    body.setLineWidth_(max(1.0, 2.0 * scale))
    AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(1, 1, 1, 0.28).set()
    body.stroke()

    # The waveform.
    heights = _bars()
    span = side * 0.62
    bar_w = span / (len(heights) * 2 - 1)
    left = inset + (side - span) / 2
    middle = inset + side / 2
    tallest = side * 0.46

    AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(1, 1, 1, 0.96).set()
    for i, value in enumerate(heights):
        height = max(bar_w, value * tallest)
        rect = AppKit.NSMakeRect(
            left + i * bar_w * 2, middle - height / 2, bar_w, height
        )
        AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            rect, bar_w / 2, bar_w / 2
        ).fill()

    image.unlockFocus()

    representation = AppKit.NSBitmapImageRep.imageRepWithData_(
        image.TIFFRepresentation()
    )
    representation.setSize_(AppKit.NSMakeSize(size, size))
    data = representation.representationUsingType_properties_(
        AppKit.NSBitmapImageFileTypePNG, {}
    )
    data.writeToFile_atomically_(str(path), True)


def build(destination: Path) -> Path:
    """Write Utter.icns next to `destination`."""
    iconset = destination.parent / "Utter.iconset"
    iconset.mkdir(parents=True, exist_ok=True)

    # The sizes iconutil expects. 16pt matters most and is the hardest: nine
    # bars at 16 points is why the envelope was resampled from twenty-two.
    for base in (16, 32, 128, 256, 512):
        draw(iconset / f"icon_{base}x{base}.png", base)
        draw(iconset / f"icon_{base}x{base}@2x.png", base * 2)

    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(destination)],
        check=True, capture_output=True, text=True,
    )
    return destination


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "packaging" / "Utter.icns"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"✓ {build(out)}")
