"""Normalise punctuation width at the boundary between scripts.

Whisper picks a punctuation style per utterance and does not hold it steady
across a code-switch, so a bilingual sentence comes back with Chinese commas
after English clauses and ASCII full stops after Chinese ones. The author
noticed it as "中英文之间的标点区分不明显".

This is deliberately **not** an LLM job. The rule is mechanical — punctuation
takes the width of the script it follows — so it can be applied with certainty,
at no latency, with polish switched off. Handing it to a model would mean
letting a model rewrite the author's text to fix a typographic detail, which is
exactly the trade 铁律 10 exists to refuse.

Only the width of punctuation changes here. No character of the author's words
is ever touched.
"""

from __future__ import annotations

# Full-width, and the half-width form each maps to.
_WIDE_TO_NARROW = {
    "，": ",", "。": ".", "？": "?", "！": "!", "：": ":", "；": ";",
    "（": "(", "）": ")",
}
_NARROW_TO_WIDE = {narrow: wide for wide, narrow in _WIDE_TO_NARROW.items()}

_PUNCT = set(_WIDE_TO_NARROW) | set(_WIDE_TO_NARROW.values())


def _is_cjk(char: str) -> bool:
    return "一" <= char <= "鿿" or "　" <= char <= "〿"


def _is_latin(char: str) -> bool:
    return char.isascii() and char.isalnum()


def normalise(text: str) -> str:
    """Give each punctuation mark the width of the script it follows.

    A mark after a Chinese character becomes full-width; after a Latin letter or
    digit, half-width. Anything ambiguous — a mark at the very start, or between
    two other marks — is left exactly as the model produced it, because guessing
    there would be changing the text on no evidence.
    """
    if not text:
        return text

    out = list(text)
    for i, char in enumerate(out):
        if char not in _PUNCT:
            continue

        previous = next((out[j] for j in range(i - 1, -1, -1) if not out[j].isspace()), "")
        if _is_cjk(previous):
            out[i] = _NARROW_TO_WIDE.get(char, char)
        elif _is_latin(previous):
            out[i] = _WIDE_TO_NARROW.get(char, char)

    result = "".join(out)

    # A full-width mark carries its own spacing, so a space in front of one is
    # always wrong; and there is never a space before a half-width mark either.
    for wide in _WIDE_TO_NARROW:
        result = result.replace(f" {wide}", wide)
    for narrow in ",.?!:;":
        result = result.replace(f" {narrow}", narrow)

    return _space_after_narrow(result)


def _space_after_narrow(text: str) -> str:
    """A half-width mark needs the space its full-width counterpart had built in.

    Converting 「，」 to 「,」 silently removes the gap the wide form carried, so
    "2001，页" would come out as "2001,页" — worse typography than what arrived.
    """
    out = []
    for i, char in enumerate(text):
        out.append(char)
        if char not in ",.?!:;":
            continue

        following = text[i + 1] if i + 1 < len(text) else ""
        if not following or following.isspace() or following in _PUNCT:
            continue
        # 3.14 and 1,000 are numbers, not sentences.
        previous = text[i - 1] if i else ""
        if char in ".," and previous.isdigit() and following.isdigit():
            continue
        out.append(" ")

    return "".join(out)


def collapse_repetition(text: str, threshold: int = 4) -> tuple[str, int]:
    """Collapse a degenerate repetition loop. Returns (text, how many removed).

    Whisper sometimes falls into repeating one short phrase until the window
    ends. The temperature ladder exists to escape that and does not always
    manage it — measured 2026-08-10, 13.4 seconds of speech came back as
    「英文是，」 forty times.

    This is not editing the author's words: they did not say it forty times.
    Anything collapsed here is reported, never removed quietly, because a
    transcript that silently drops repetitions would hide a failing model.
    """
    if not text:
        return text, 0

    import re

    removed = 0
    # Two or more characters repeated many times over — long enough not to catch
    # a real 「好好」 or 「哈哈哈」, short enough to catch a decoder loop.
    pattern = re.compile(r"(.{2,30}?)\1{%d,}" % (threshold - 1))
    while True:
        match = pattern.search(text)
        if not match:
            break
        unit = match.group(1)
        count = len(match.group(0)) // len(unit)
        removed += count - 1
        text = text[: match.start()] + unit + text[match.end() :]

    return text, removed
