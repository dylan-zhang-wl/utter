"""听记 — the session behind Listen mode.

P3 task 1. This module is only the record: what was heard, what it was
translated to, and getting both onto disk. The audio, the model and the
window are elsewhere.

Two things drive every decision here.

**A meeting cannot be repeated.** Dictation can: if a sentence comes back
wrong the author says it again. A lecture is gone. So an entry is written the
moment it exists rather than at the end, and the session survives a crash, a
flat battery, and — most likely of all — forgetting to press stop.

**Two records, not one.** The author asked for both: one verbatim, with every
filler and stumble the speaker made, and one bilingual and cleaned up for
reading. They are generated from the same line-per-entry file, which is the
real archive; the two documents are conveniences that can be rebuilt from it
at any time.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

#: Where sessions live. Read through `backend.config` at call time, never
#: imported by value — that is lesson 39, and it cost 1,299 stray files.
LISTEN_DIRNAME = "listen"

ENTRIES_FILE = "entries.jsonl"
VERBATIM_FILE = "原文.md"
BILINGUAL_FILE = "对照.md"


@dataclass
class Entry:
    """One thing somebody said.

    `source` is what the recogniser heard, kept exactly, fillers and all.
    `display` is the same sentence tidied for the screen. `translation` may be
    None for a while — it arrives on a later batch — and may stay None if the
    model was unreachable, which must never cost us the source.
    """

    index: int
    started_at: float           # seconds from the start of the session
    source: str                 # verbatim, never rewritten
    display: str = ""           # fillers stripped, for the screen
    translation: str | None = None
    #: Whisper's own confidence for this segment. Recorded from the first day
    #: and shown nowhere: the plan is to learn the real distribution from a
    #: real meeting before deciding what "low" means (§8.4).
    confidence: float | None = None
    forced: bool = False        # the cut was imposed, so the text may be cut mid-word
    source_edited: bool = False
    translation_edited: bool = False

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "started_at": round(self.started_at, 3),
            "source": self.source,
            "display": self.display,
            "translation": self.translation,
            "confidence": self.confidence,
            "forced": self.forced,
            "source_edited": self.source_edited,
            "translation_edited": self.translation_edited,
        }

    @classmethod
    def from_json(cls, raw: dict) -> "Entry":
        """Tolerant on the way in: a field this version does not know about is
        not a reason to lose the sentence."""
        return cls(
            index=int(raw.get("index", 0)),
            started_at=float(raw.get("started_at", 0.0)),
            source=raw.get("source") or "",
            display=raw.get("display") or "",
            translation=raw.get("translation"),
            confidence=raw.get("confidence"),
            forced=bool(raw.get("forced", False)),
            source_edited=bool(raw.get("source_edited", False)),
            translation_edited=bool(raw.get("translation_edited", False)),
        )


@dataclass
class ListenSession:
    """A meeting, from 开始 to 结束.

    States are `idle` → `recording` ⇄ `paused` → `stopped`. The state matters
    to the caller for one reason only: entries are accepted while recording and
    refused otherwise, so a stray transcription arriving after the user pressed
    stop cannot append itself to a finished document.
    """

    directory: Path
    title: str = ""
    entries: list[Entry] = field(default_factory=list)
    state: str = "idle"

    # -- construction -----------------------------------------------------

    @classmethod
    def create(cls, base_dir: Path | None = None, *, title: str = "",
               now: datetime | None = None) -> "ListenSession":
        """A new session in its own directory, named for when it started."""
        from backend import config as _config

        root = Path(base_dir) if base_dir else _config.DEFAULT_DIR / LISTEN_DIRNAME
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        return cls(directory=root / stamp, title=title or stamp)

    @classmethod
    def load(cls, directory: Path) -> "ListenSession":
        """Rebuild a session from its entries file.

        This is the recovery path, and it is why the jsonl is the archive of
        record: a truncated final line — the shape a crash leaves — costs that
        one sentence and nothing else.
        """
        session = cls(directory=Path(directory), title=Path(directory).name,
                      state="stopped")
        path = session.entries_path
        if not path.exists():
            return session
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                session.entries.append(Entry.from_json(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError):
                log.warning("%s 第 %d 行读不出来，跳过这一条", path.name, number)
        return session

    # -- paths ------------------------------------------------------------

    @property
    def entries_path(self) -> Path:
        return self.directory / ENTRIES_FILE

    @property
    def verbatim_path(self) -> Path:
        return self.directory / VERBATIM_FILE

    @property
    def bilingual_path(self) -> Path:
        return self.directory / BILINGUAL_FILE

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self.state = "recording"
        self._ensure_directory()

    def pause(self) -> None:
        """Pausing writes both documents.

        The author asked for this specifically. Pause is when someone steps out
        of the room and looks at what has been captured so far, and a record
        that only exists after 结束 is not a record during a two-hour meeting.
        """
        if self.state == "recording":
            self.state = "paused"
        self.write_documents()

    def resume(self) -> None:
        if self.state == "paused":
            self.state = "recording"

    def stop(self) -> None:
        self.state = "stopped"
        self.write_documents()

    @property
    def is_recording(self) -> bool:
        return self.state == "recording"

    # -- entries ----------------------------------------------------------

    def add(self, entry: Entry) -> bool:
        """Append one entry and put it on disk immediately.

        Returns whether it was accepted. A transcription that finishes after
        the user pressed stop is dropped rather than appended, but says so in
        the log — the pipeline is asynchronous and this is a real race, not a
        theoretical one.
        """
        if self.state not in ("recording", "paused"):
            log.info("会话已结束，第 %d 条不再收录", entry.index)
            return False
        self.entries.append(entry)
        self._append_to_disk(entry)
        return True

    def set_translation(self, index: int, text: str) -> bool:
        """Fill in a translation that arrived later.

        Rewrites the whole entries file rather than appending, because a
        translation belongs to an entry already written. That is O(n) per
        batch and n is a few hundred — cheaper than any scheme that would let
        the file and memory disagree.
        """
        for entry in self.entries:
            if entry.index == index:
                entry.translation = text
                self._rewrite_entries()
                return True
        return False

    # -- disk -------------------------------------------------------------

    def _ensure_directory(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError:
            log.warning("建不了会话目录 %s", self.directory, exc_info=True)

    def _append_to_disk(self, entry: Entry) -> None:
        """Never raises. Losing the file is bad; losing the meeting is worse,
        and the entry is already in memory (铁律 8)."""
        try:
            self._ensure_directory()
            with self.entries_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")
            os.chmod(self.entries_path, 0o600)
        except OSError:
            log.warning("第 %d 条写不进 %s，只在内存里", entry.index,
                        self.entries_path, exc_info=True)

    def _rewrite_entries(self) -> None:
        try:
            self._ensure_directory()
            body = "".join(json.dumps(e.to_json(), ensure_ascii=False) + "\n"
                           for e in self.entries)
            tmp = self.entries_path.with_suffix(".jsonl.tmp")
            tmp.write_text(body, encoding="utf-8")
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.entries_path)
        except OSError:
            log.warning("重写 %s 失败", self.entries_path, exc_info=True)

    def write_documents(self) -> None:
        """The two records the author asked for.

        Both are derived from `entries`, so either can be deleted and rebuilt.
        Never raises: a session that cannot write its documents still has its
        jsonl, and that is the one that matters.
        """
        if not self.entries:
            return
        try:
            self._ensure_directory()
            self._write(self.verbatim_path, self.render_verbatim())
            self._write(self.bilingual_path, self.render_bilingual())
        except OSError:
            log.warning("写会议记录失败，entries.jsonl 仍然完好", exc_info=True)

    def _write(self, path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)

    # -- rendering --------------------------------------------------------

    def render_verbatim(self) -> str:
        """Every word, exactly as heard.

        No filler stripping, no punctuation repair, no translation. This is the
        one anybody goes to when they need to know what the speaker actually
        said, so nothing may be tidied out of it.
        """
        lines = [f"# {self.title}", "", "> 逐字记录，未做任何清理。", ""]
        lines += [entry.source for entry in self.entries if entry.source]
        return "\n\n".join(lines).rstrip() + "\n"

    def render_bilingual(self) -> str:
        """English and Chinese, tidied for reading."""
        lines = [f"# {self.title}", "", "> 中英对照。英文已去口水词；逐字版见 原文.md。", ""]
        for entry in self.entries:
            shown = entry.display or entry.source
            if not shown:
                continue
            lines.append(shown)
            lines.append(entry.translation if entry.translation else "*（未翻译）*")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def list_sessions(base_dir: Path | None = None) -> list[Path]:
    """Session directories, newest first."""
    from backend import config as _config

    root = Path(base_dir) if base_dir else _config.DEFAULT_DIR / LISTEN_DIRNAME
    if not root.exists():
        return []
    return sorted((d for d in root.iterdir() if d.is_dir()), reverse=True)
