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


_TERMINAL = "。？！.?!…、，,;；:："


def close_sentence(text: str) -> str:
    """Give an utterance a closing mark if it ends without one.

    Measured on the author's own dictation, 2026-08-10: one utterance in seven
    ended with any punctuation at all. The cause is the gesture — they release
    the key as the last word lands, so Whisper receives audio that stops
    mid-breath and declines to close a sentence it cannot tell has ended.

    Permitted by 铁律 10, which allows adding punctuation and forbids changing
    words. Nothing here touches a word. The mark follows the script of the last
    character, so a Chinese sentence gets 。 and an English one gets a full stop.

    Deliberately conservative about what it will not close: text already ending
    in any punctuation, and text ending mid-clause on a comma, are left alone.
    """
    stripped = text.rstrip()
    if not stripped or stripped[-1] in _TERMINAL:
        return text
    return stripped + ("。" if _is_cjk(stripped[-1]) else ".")


# Whisper's stock hallucinations, in the languages this author dictates.
#
# Found in their own archive: a near-silent recording came back as
# 「字幕志愿者 李宗盛。」 and was injected into a document. Nobody said it. It is
# a subtitle credit, learned from the video captions that make up much of
# Whisper's Chinese training data, and the model reaches for it whenever there
# is nothing to transcribe.
#
# The VAD gate is the first defence and it does most of the work, but it
# answers "is anyone talking", not "is this output real". A cough or a keyboard
# clack passes the gate and gives the model an opening.
#
# This list only ever removes text that is the WHOLE utterance. A sentence that
# happens to contain 「谢谢观看」 in the middle of real speech is real speech.
_HALLUCINATIONS = (
    "字幕志愿者", "字幕由", "字幕组", "中文字幕",
    "谢谢观看", "感谢观看", "感谢收看", "请不吝点赞", "订阅", "转发", "打赏",
    "明镜与点点栏目", "下次再见", "我们下期再见",
    "thanks for watching", "thank you for watching", "please subscribe",
    "subtitles by", "amara.org", "www.", "http",
)

#: What may be left over once every stock phrase is removed. 「字幕志愿者 李宗盛」
#: leaves 「李宗盛」; a real sentence leaves most of itself.
_HALLUCINATION_REMAINDER = 4


def is_hallucination(text: str) -> bool:
    """Is this the model filling a silence with a subtitle credit?

    Not "does it contain a stock phrase" — the first version asked that, with a
    length limit, and it would have thrown away

        这一节我要讨论字幕志愿者这个群体在数字人文里的位置

    which is exactly the kind of sentence this author dictates. Discarding
    something they actually said is far worse than letting one stray line
    through (铁律 8).

    So: strip the stock phrases out and see what is left. A hallucination is
    almost nothing but stock phrases; a real sentence that mentions one is
    still a sentence afterwards.
    """
    import re as _re

    stripped = _re.sub(r"[\s，。？！、；：,.?!;:\-—…]+", "", text).lower()
    if not stripped:
        return False

    remainder = stripped
    matched = False
    for phrase in _HALLUCINATIONS:
        # The phrases are written readably, with spaces; the text has had its
        # spaces removed. Normalise both or "thanks for watching" never matches
        # "thanksforwatching" — which it did not, in the first version.
        needle = _re.sub(r"\s+", "", phrase)
        if needle and needle in remainder:
            matched = True
            remainder = remainder.replace(needle, "")
    return matched and len(remainder) <= _HALLUCINATION_REMAINDER


# How long a pause has to be before it reads as the end of a sentence rather
# than the end of a clause. Measured against the author's own dictation: the
# breaths between clauses run 300-500ms, and the gaps where they finished a
# thought run past a second.
SENTENCE_PAUSE_MS = 800

_ENDS_A_CLAUSE = "，。？！、；：,.?!;:…—"


def punctuate_pause(text: str, gap_ms: float | None, *, final: bool = False) -> str:
    """Close a streamed clause using the pause the speaker actually left.

    Streaming produces clauses of one to three seconds, and Whisper does not
    punctuate a fragment — handed 「而且」 it returns 「而且」. Two wrong answers
    were tried before this one:

      * close every clause like a sentence, which gave
        「但是这个延迟。好像。还是比较多的。」
      * close none of them, which gave a paragraph with no punctuation at all

    Both were guesses about where the sentence ends. The speaker already
    answered that question by pausing, and the VAD measured it. A short breath
    is a comma; a long one is a full stop; the end of the session is a full
    stop. No model is involved and nothing is invented — the only thing added
    is a mark, which 铁律 10 has always allowed.
    """
    stripped = text.rstrip()
    if not stripped:
        return text
    if stripped[-1] in _ENDS_A_CLAUSE:
        return stripped  # Whisper already decided; leave it alone

    if final:
        return stripped + "。"
    if gap_ms is None:
        return stripped + "，"
    return stripped + ("。" if gap_ms >= SENTENCE_PAUSE_MS else "，")
