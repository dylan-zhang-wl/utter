"""会后纪要 — the summary, generated once when the meeting ends.

P3 task 5. Ninety minutes of speech does not fit in one prompt, so this is the
usual two passes: summarise each stretch, then summarise the summaries.

Two rules make it safe to let a model write freely here, where 铁律 10 forbids
it two modules away.

**A summary is new text by definition.** It is supposed to compress and
rephrase — that is the whole request. So the guard is not "did the wording
change" but "can the reader get back to what was actually said": every section
carries the entry numbers it was drawn from.

**A summary never replaces anything.** 原文.md and 对照.md are already on disk
before this runs and are not touched by it. If the model invents an agenda item
that nobody raised, the transcript beside it still says otherwise.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

#: How much source text goes into one map-step request. Characters rather than
#: entries: entries vary from three words to three sentences, and it is the
#: prompt size that has a limit.
CHUNK_CHARS = 6000

SUMMARY_FILE = "纪要.md"

_MAP_PROMPT = (
    "你在整理一场学术会议的记录。下面是会议的一段。"
    "提炼这一段里真正被讨论的内容：议题、观点、分歧、结论、待办。"
    "用中文，分条写，每条一句话。"
    "\n\n"
    "**只写记录里确实说过的**。没有讨论到的不要补，"
    "听起来像是会议该有的内容但记录里没有的，一律不要写。"
    "这一段如果没有实质内容，就只回一句「（无实质内容）」。"
)

_REDUCE_PROMPT = (
    "下面是同一场会议各段的要点。把它们合成一份完整纪要。"
    "用中文，分「议题」「结论」「待办」三部分；某一部分没有内容就写「无」。"
    "\n\n"
    "**不要引入任何各段要点里没有的内容。** 合并重复的，保留分歧，"
    "不要把没有结论的讨论写成有结论。"
)


def plan_chunks(entries, max_chars: int = CHUNK_CHARS) -> list[list]:
    """Split entries into stretches small enough to summarise in one request.

    Never splits an entry, and never returns an empty chunk. A single entry
    longer than the budget gets a chunk to itself rather than being cut —
    truncating somebody's sentence to fit a limit is how a summary comes to
    say something the speaker did not.
    """
    chunks: list[list] = []
    current: list = []
    size = 0
    for entry in entries:
        length = len(entry.source or "")
        if current and size + length > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(entry)
        size += length
    if current:
        chunks.append(current)
    return chunks


def _range_label(chunk) -> str:
    first, last = chunk[0].index, chunk[-1].index
    return f"条目 {first}" if first == last else f"条目 {first}–{last}"


def _timestamp(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def summarise(session, complete, *, max_chars: int = CHUNK_CHARS) -> str | None:
    """Write 纪要.md for a finished session. Returns the text, or None.

    `complete(system, user) -> str` is the raw model call — the same shape the
    LLM providers already expose.

    Returns None rather than raising when there is nothing to summarise or the
    model is unreachable: the meeting is already recorded twice over by the
    time this runs, and a missing summary is an inconvenience where a lost
    transcript would not be (铁律 8).
    """
    entries = [e for e in session.entries if (e.source or "").strip()]
    if not entries:
        return None

    chunks = plan_chunks(entries, max_chars)
    partials: list[tuple[str, str]] = []      # (range label, points)
    for chunk in chunks:
        body = "\n".join(
            f"[{_timestamp(e.started_at)}] {e.source}" for e in chunk
        )
        try:
            points = (complete(_MAP_PROMPT, body) or "").strip()
        except Exception:
            log.warning("这一段纪要生成失败，跳过：%s", _range_label(chunk), exc_info=True)
            continue
        if points and "无实质内容" not in points:
            partials.append((_range_label(chunk), points))

    if not partials:
        log.info("没有可写进纪要的内容")
        return None

    if len(partials) == 1:
        combined = partials[0][1]
    else:
        joined = "\n\n".join(f"【{label}】\n{points}" for label, points in partials)
        try:
            combined = (complete(_REDUCE_PROMPT, joined) or "").strip()
        except Exception:
            # Better the per-stretch points than nothing: they are already a
            # usable record, just not merged.
            log.warning("合并纪要失败，改用分段要点", exc_info=True)
            combined = joined

    document = _render(session, combined, partials)
    try:
        path = session.directory / SUMMARY_FILE
        path.write_text(document, encoding="utf-8")
        import os

        os.chmod(path, 0o600)
    except OSError:
        log.warning("纪要写不进盘，只返回文本", exc_info=True)
    return document


def _render(session, combined: str, partials) -> str:
    lines = [f"# {session.title} — 纪要", ""]
    lines.append("> 由记录自动生成。**下面每一条都可能是模型的概括**，")
    lines.append("> 原话见同目录的 原文.md 与 对照.md。")
    lines.append("")
    lines.append(combined)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 依据")
    lines.append("")
    # The point of this section: any claim above can be traced to a stretch of
    # the transcript, so a reader who doubts a line can go and read it.
    for label, points in partials:
        lines.append(f"### {label}")
        lines.append("")
        lines.append(points)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
