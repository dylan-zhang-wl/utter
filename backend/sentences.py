"""把转录文本切成句子 — 用 Whisper 自己的标点，而不是静音时长。

Two rounds of tuning the VAD produced fragments (「My father」) and then blocks
(thirty seconds of unbroken text), because silence length is being asked a
question it cannot answer: a speaker pauses the same ~500ms between phrases as
between sentences, so no threshold separates a comma from a full stop.

The information was there all along. Whisper punctuates English well — measured
2026-08-12 on real output: four sentences, six marks, proper nouns intact — and
this project was throwing that away and guessing from the audio instead.

So the VAD keeps doing what it is good at (finding *some* boundary, cheaply) and
sentences are cut from the *text*. A chunk that ends mid-sentence leaves a
remainder, which is held and joined to the front of the next chunk's text. That
is the standard shape for streaming translation — segmentation guided by
punctuation, supported by past sentences as context — rather than the hard
audio segmentation that is a known source of errors.

铁律 2 is untouched: nothing is transcribed twice. This splits text that has
already been transcribed exactly once.
"""

from __future__ import annotations

import re

#: What ends a sentence. Chinese marks included because the speaker may be
#: speaking Chinese, and the closing quote/bracket forms because a sentence
#: ending inside quotation marks still ends there.
_ENDERS = ".!?。！？…"
_SENTENCE = re.compile(
    r"[^" + re.escape(_ENDERS) + r"]*[" + re.escape(_ENDERS) + r"]+[\"'”’)）】」』]*\s*"
)

#: Abbreviations whose full stop does not end a sentence. Deliberately short:
#: every entry here is a place where a real sentence break would be missed, so
#: only the ones that actually appear in academic speech are worth the risk.
_ABBREVIATIONS = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "cf.", "et al.", "fig.", "no.", "pp.",
)

#: A held remainder longer than this is emitted even without punctuation. A
#: speaker who never lands a full stop must not accumulate for ever, and the
#: end of a meeting must not swallow the last thought.
MAX_HELD_CHARS = 400


#: A full stop between digits belongs to a number, not to a sentence. Found in
#: a real talk about 「Democracy 1.0」 and 「Democracy 2.0」: every mention was
#: cut in two, leaving 「Democracy 2.」 as one entry and 「0, Far-Right
#: extremists…」 as the next — and the translator faithfully rendered the
#: stray 「0，」.
_DECIMAL = re.compile(r"\d\s*$")

#: What follows a full stop settles most of the hard cases at once: English
#: starts a sentence with a capital, so a lower-case word after the stop means
#: the stop belonged to something else. 「the U.S. wasn't even really a
#: democracy」 arrived as three separate entries — 「…in fact the U.」, 「S.」,
#: 「wasn't even really a democracy…」 — and this rule keeps all three together
#: without needing to know that U.S. is an abbreviation.
#:
#: It also covers the abbreviations nobody listed, at the cost of joining two
#: sentences when the model forgets to capitalise the second. That trade is
#: worth taking: a run-on is readable, a sentence cut in three is not.
_CONTINUES = re.compile(r"^\s*[a-z]")

#: And the other half of the same sentence: a lone capital before the stop is
#: an initial being spelled out. 「U.」 in 「the U.S.」 is followed by a capital,
#: so the rule above cannot see it; together the two keep 「the U.S. wasn't
#: even really a democracy」 in one piece while still ending a sentence that
#: genuinely finishes on 「…visited the U.S.」
_INITIAL = re.compile(r"(?:^|[\s(\[\"'])[A-Z]\.\s*$")


def _ends_on_abbreviation(text: str) -> bool:
    """Whether the trailing full stop belongs to an abbreviation.

    The match has to be on a word boundary. A plain `endswith` treats
    「southwest.」 as the abbreviation 「st.」 and swallows the sentence — found
    by the test for the very case this module exists to fix.
    """
    tail = text.rstrip().lower()
    return any(re.search(r"(?:^|[\s(\[\"'])" + re.escape(abbr) + r"$", tail)
               for abbr in _ABBREVIATIONS)


def split_sentences(text: str) -> tuple[list[str], str]:
    """(complete sentences, the unfinished remainder).

    The remainder is whatever follows the last sentence-ending mark. It is the
    caller's job to hold it until more text arrives.
    """
    text = (text or "").strip()
    if not text:
        return [], ""

    sentences: list[str] = []
    position = 0
    for match in _SENTENCE.finditer(text):
        piece = text[position:match.end()]
        # A full stop after "Dr." is not the end of a sentence. Keep reading.
        if _ends_on_abbreviation(piece):
            continue
        rest = text[match.end():]
        if _CONTINUES.match(rest) or _INITIAL.search(piece):
            continue
        # 「2.0」: a digit on both sides of the stop.
        if _DECIMAL.search(piece[:-1]) and rest[:1].isdigit():
            continue
        cleaned = piece.strip()
        # 「...」 arrived as entry #0 of a real session: punctuation with no
        # word in it is not a sentence, and it went to the translator as one.
        if cleaned and re.search(r"\w", cleaned):
            sentences.append(cleaned)
        position = match.end()

    return sentences, text[position:].strip()


class SentenceAssembler:
    """Feed it transcribed chunks; it gives back whole sentences.

    Holds the tail of a chunk that ended mid-sentence and joins it to the front
    of the next one, so 「My father」 and 「from equity states」 arrive as the one
    sentence they were spoken as.
    """

    def __init__(self, max_held_chars: int = MAX_HELD_CHARS):
        self._held = ""
        self._max_held = max_held_chars
        #: Whether the full stop at the end of the held text was taken off
        #: because the audio was cut there. See `feed(cut=True)`.
        self._restore_stop = False

    @property
    def pending(self) -> str:
        return self._held

    def feed(self, text: str, *, final: bool = False,
             cut: bool = False) -> list[str]:
        """Sentences that are now complete. `final` flushes the remainder.

        `final` is what the end of a meeting passes, and it is why nothing is
        ever lost: whatever is still held comes out as its own entry even
        though the speaker never finished the thought.

        `cut` says the audio ran out here rather than the speaker stopping.
        Whisper finishes what it is given with a full stop whether or not one
        was spoken, so on a chunk cut at the ceiling that last stop is the
        model's habit and cannot be trusted. Measured on a real talk: fifteen
        of fifty-two entries began with a lower-case word, which is what a
        sentence chopped in half looks like — 「…would be more of a revelation
        to me.」 followed by 「to me than it was to you.」

        So the stop comes off and the tail is held. Which of the two it really
        was gets decided by the next chunk, from the one piece of evidence
        English gives away for free: a capital letter means a new sentence had
        started after all, and the stop goes back.
        """
        text = (text or "").strip()
        if self._restore_stop and self._held:
            if text[:1].isupper() or (final and not text):
                self._held = self._held.rstrip() + "."
        self._restore_stop = False

        joined = f"{self._held} {text}".strip() if self._held else text
        if not joined:
            if final:
                self._held = ""
            return []

        sentences, remainder = split_sentences(joined)

        if final:
            if remainder:
                sentences.append(remainder)
            self._held = ""
            return sentences

        if len(remainder) > self._max_held:
            # A speaker who never lands a full stop. Emit it rather than let
            # the buffer grow without bound — a late sentence is better than a
            # sentence that never arrives.
            sentences.append(remainder)
            remainder = ""

        if cut and not remainder and sentences:
            # Everything in this chunk parsed as complete, but the audio was
            # cut here — so the last one is only complete because the model
            # ended it. Hold it back, minus the stop it was given.
            last = sentences.pop()
            if last.endswith("."):
                self._held = last[:-1]
                self._restore_stop = True
            else:
                sentences.append(last)

        else:
            self._held = remainder
        return sentences

    def flush(self) -> list[str]:
        return self.feed("", final=True)
