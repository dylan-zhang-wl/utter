"""The translation slot for Listen mode — batched, serial, and order-safe.

P3 task 2. Three constraints shape all of it.

**Batched.** A ninety-minute meeting is roughly four hundred sentences. One
request each is four hundred round trips, enough to hit a rate limit and to
cost real money for no benefit. Sentences go out in small groups.

**Serial.** 铁律 11. One request in flight at a time, because two overlapping
requests can come back in either order and the transcript is an ordered
document.

**Order-safe.** Batching creates a failure mode that translating one at a time
does not have: the model returns a different number of lines than it was given
and every translation after that point attaches to the wrong sentence. A
transcript where line 12's Chinese sits under line 11's English is worse than
one with no Chinese at all, because nothing about it looks wrong. So each line
carries a number, the reply is matched back by that number, and anything that
does not match cleanly is simply left untranslated.
"""

from __future__ import annotations

import logging
import re
import threading
import time

log = logging.getLogger(__name__)

#: How many sentences travel together. Small, because a batch is also the
#: latency floor: nothing in a batch appears until the whole batch returns.
BATCH_SIZE = 3

#: And a ceiling on waiting, so the last sentence before a long silence does
#: not sit in the queue until somebody speaks again.
MAX_WAIT_SECONDS = 8.0

#: How much of the preceding conversation the model sees. Two entries, matching
#: the polish side: enough to resolve a pronoun or a cut-off clause, and fixed,
#: so request size does not grow with the length of the meeting.
CONTEXT_ENTRIES = 2

_NUMBERED = re.compile(r"^\s*(\d+)\s*[.、:：)]\s*(.+)$")


def foreign_script(translation: str, source: str) -> str:
    """Characters in the Chinese that belong to neither language.

    Seen once in 78 sentences of a real talk: 「直到我 აღარ需要再去想它」 — the
    model rendered 「不再」 into Georgian and carried on in Chinese. It is rare
    and it is unmissable on screen, so it is worth one retry.

    Deliberately narrow. This user writes about translation, and a legitimate
    sentence may well carry Greek letters, kana in a quotation, or a term in
    Cyrillic — so anything already present in the English is allowed through.
    Only script that the model introduced by itself counts.
    """
    allowed = set(source or "")
    strange = []
    for ch in translation or "":
        if ch in allowed or ch.isascii() or ch.isspace():
            continue
        if ("\u3000" <= ch <= "\u303f"        # CJK punctuation
                or "\u4e00" <= ch <= "\u9fff"  # Han
                or "\uff00" <= ch <= "\uffef"  # fullwidth forms
                or ch in "《》〈〉—…·"):
            continue
        strange.append(ch)
    return "".join(dict.fromkeys(strange))


def format_batch(sources: list[str]) -> str:
    """Number the lines going out, so the reply can be matched back."""
    return "\n".join(f"{i + 1}. {text}" for i, text in enumerate(sources))


def parse_batch(reply: str, count: int) -> dict[int, str]:
    """Match a numbered reply back to positions, dropping anything unclear.

    Returns {position: translation} for the lines that came back cleanly.
    A missing or duplicated number costs that one sentence its translation and
    leaves every other sentence correct — which is the whole point of numbering
    rather than zipping two lists together and hoping.
    """
    found: dict[int, str] = {}
    for line in (reply or "").splitlines():
        match = _NUMBERED.match(line)
        if not match:
            continue
        position = int(match.group(1)) - 1
        text = match.group(2).strip()
        if 0 <= position < count and text and position not in found:
            found[position] = text
    if found or count != 1:
        return found

    # A batch of one comes back unnumbered, and reliably so: asked to render a
    # single numbered line, the model decides the number is clutter and returns
    # the bare translation. Measured against gpt-5.4-mini — three lines came
    # back numbered perfectly, one line came back as 「因此，等值问题其实根本不是
    # 一个关于词语的问题。」 with no 「1.」 in sight.
    #
    # That case is not rare. Every max_wait flush, every 暂停 and the final
    # flush at 结束 can carry a single entry, so refusing to translate it would
    # have quietly dropped the last sentence of every meeting.
    #
    # Taking the whole reply is safe here for the reason the numbering exists:
    # with one entry there is nothing to misalign it against.
    whole = "\n".join(line.strip() for line in (reply or "").splitlines()
                      if line.strip()).strip()
    return {0: whole} if whole else {}


class TranslationQueue:
    """Collects entries, translates them in small serial batches.

    The worker thread is a loop around `drain_once`, which is also what the
    tests call — the batching decisions are therefore checkable without any
    threading at all.
    """

    def __init__(self, translate, on_translated, *,
                 batch_size: int = BATCH_SIZE,
                 max_wait: float = MAX_WAIT_SECONDS,
                 clock=time.monotonic):
        """`translate(text, context) -> str | None`, `on_translated(index, text)`."""
        self._translate = translate
        self._on_translated = on_translated
        self._batch_size = batch_size
        self._max_wait = max_wait
        self._clock = clock

        self._pending: list[tuple[int, str]] = []      # (entry index, source)
        self._history: list[tuple[str, str]] = []      # (source, translation)
        self._oldest_at: float | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- input ------------------------------------------------------------

    def submit(self, index: int, source: str) -> None:
        with self._lock:
            self._pending.append((index, source))
            if self._oldest_at is None:
                self._oldest_at = self._clock()
        self._wake.set()

    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    # -- batching ---------------------------------------------------------

    def _ready(self, *, force: bool) -> list[tuple[int, str]]:
        """Take a batch, or nothing. Called with the lock held."""
        if not self._pending:
            return []
        old_enough = (self._oldest_at is not None
                      and self._clock() - self._oldest_at >= self._max_wait)
        if not force and len(self._pending) < self._batch_size and not old_enough:
            return []
        batch, self._pending = self._pending[:self._batch_size], self._pending[self._batch_size:]
        self._oldest_at = self._clock() if self._pending else None
        return batch

    def drain_once(self, *, force: bool = False) -> int:
        """Translate at most one batch. Returns how many entries it handled.

        Serial by construction: this is only ever called from one thread, so
        there is never a second request in flight (铁律 11).
        """
        with self._lock:
            batch = self._ready(force=force)
        if not batch:
            return 0

        sources = [text for _, text in batch]
        context = "\n".join(f"{s}\n{t}" for s, t in self._history[-CONTEXT_ENTRIES:])
        reply = None
        try:
            reply = self._translate(format_batch(sources), context or None)
        except Exception:
            # Never propagate: a failed translation must not stop the meeting,
            # and the English is already safe in the session (铁律 8).
            log.warning("这一批翻译失败了，原文不受影响", exc_info=True)

        matched = parse_batch(reply, len(batch)) if reply else {}
        if reply and not matched:
            log.warning("翻译回来的内容对不上编号，这一批保持未翻译")

        for position, (index, source) in enumerate(batch):
            translation = matched.get(position)
            strange = foreign_script(translation or "", source)
            if strange:
                log.warning("译文里混进了 %r，重译一次：%s", strange, translation)
                retried = None
                try:
                    retried = self._translate(format_batch([source]), context or None)
                except Exception:
                    log.warning("重译也失败了，保留原样", exc_info=True)
                again = parse_batch(retried, 1).get(0) if retried else None
                if again and not foreign_script(again, source):
                    translation = again
            if translation:
                self._history.append((source, translation))
                try:
                    self._on_translated(index, translation)
                except Exception:  # pragma: no cover - a UI callback misbehaving
                    log.warning("回填第 %d 条译文时出错", index, exc_info=True)
        return len(batch)

    def flush(self) -> None:
        """Translate everything still waiting. For 暂停 and 结束."""
        while self.drain_once(force=True):
            pass

    # -- thread -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="utter-translate")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.drain_once() == 0:
                # Wait, but wake up periodically so max_wait can expire on a
                # queue that stopped receiving.
                self._wake.wait(timeout=0.5)
                self._wake.clear()

    def stop(self, *, flush: bool = True) -> None:
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=30)
        if flush:
            self.flush()
