"""Scratchpad mode, and the session archive underneath it.

Design §4.1f. Dictation bound to no window: text accumulates and the author
decides where it goes. It needs no injection, no target tracking and no
permission beyond the microphone, which makes it both the cheapest thing in P2a
and plausibly the most used — dictating notes while reading a PDF and pasting
them into a draft afterwards is a real workflow, and it is one the
inject-at-cursor design cannot serve at all.

The archive is where 铁律 8 becomes concrete. Entries land on disk as they
arrive, one JSON object per line. A crash nine minutes into a ten-minute
dictation costs the last sentence, not the session.
"""

from __future__ import annotations

import json
import logging
import os
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backend.config import DEFAULT_DIR
from backend.pipeline import Utterance

log = logging.getLogger(__name__)


def _is_cjk(char: str) -> bool:
    return unicodedata.east_asian_width(char) in ("W", "F") or "一" <= char <= "鿿"


def join_text(pieces) -> str:
    """Join utterances the way the script they are in expects.

    A space between two Chinese sentences is wrong typography, and a missing
    space between two English ones is wrong English. The author writes both,
    often in the same paragraph, so the separator is decided per boundary rather
    than fixed.
    """
    kept = [p.strip() for p in pieces if p and p.strip()]
    if not kept:
        return ""

    out = kept[0]
    for piece in kept[1:]:
        if _is_cjk(out[-1]) or _is_cjk(piece[0]):
            out += piece
        else:
            out += " " + piece
    return out


class SessionArchive:
    """Append-only record of everything dictated in a session.

    JSON Lines rather than a JSON array. Appending is O(1) instead of rewriting
    the whole file per utterance, and a crash mid-write costs one truncated line
    that a reader can skip — where a half-written array is unparseable in its
    entirety.
    """

    def __init__(self, base_dir: Path | None = None, session_id: str | None = None):
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_DIR
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def path(self) -> Path:
        return self.base_dir / "sessions" / f"session_{self.session_id}.jsonl"

    def append(self, utterance: Utterance) -> None:
        """Write one entry. Never raises.

        A full disk or an unwritable directory must not abort a dictation that
        is still perfectly usable in memory — the text is already in the
        scratchpad, and losing the recording of it is strictly better than
        losing the text (铁律 8).
        """
        record = {
            "index": utterance.index,
            "mode": utterance.mode,
            "text": utterance.text,
            "raw_text": utterance.raw_text,
            "translation": utterance.translation,
            "polished": utterance.polished,
            "forced": utterance.forced,
            "start_sec": round(utterance.start_sec, 3),
            "end_sec": round(utterance.end_sec, 3),
        }

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            existed = self.path.exists()
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not existed:
                # It holds whatever the author dictated, which may be an
                # unpublished draft or a confidential translation.
                os.chmod(self.path, 0o600)
        except OSError:
            log.warning("could not archive utterance %d", utterance.index, exc_info=True)


@dataclass
class Scratchpad:
    """Unbound dictation buffer. The author decides where the text goes."""

    archive: SessionArchive | None = None
    entries: list[Utterance] = field(default_factory=list)

    def add(self, utterance: Utterance) -> None:
        self.entries.append(utterance)
        if self.archive is not None:
            self.archive.append(utterance)

    def peek(self, raw: bool = False) -> str:
        """The buffer as one string, without emptying it.

        `raw=True` gives the pre-polish transcript, for diffing when a passage
        reads as though it says something the author did not say.
        """
        return join_text([e.raw_text if raw else e.text for e in self.entries])

    def take(self) -> str:
        """The buffer as one string, then empty it.

        Only the in-memory buffer is cleared. The archive keeps its copy — that
        separation is the whole reason the archive exists.
        """
        text = self.peek()
        self.clear()
        return text

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)
