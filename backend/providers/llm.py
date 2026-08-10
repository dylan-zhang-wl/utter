"""Language-model providers, and the prompts they are given.

Providers do one thing — `complete(system, user)`. The prompts live here rather
than in each provider, because the prompt is where 铁律 10 is actually enforced
and it must be identical whichever model runs it.

On 铁律 10. The author dictates academic argument. A model that turns

    翻译不是复制，而是一种重写

into

    翻译是一种创造性的转换过程

has produced something more fluent, more "paper-like", and no longer the
author's claim — and because it reads well, re-reading will not catch it. So
every level of polish, including the heaviest, forbids rewording and forbids
adding or removing content. Only punctuation, filler removal and paragraphing
are ever on the table.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

import httpx  # re-exported so tests can patch one place

log = logging.getLogger(__name__)

__all__ = [
    "LlmProvider",
    "LlmError",
    "polish",
    "safe_polish",
    "translate",
    "safe_translate",
    "polish_prompt",
    "TRANSLATE_PROMPT",
    "httpx",
]

# A polished result this much shorter than the input means the model answered
# instead of editing ("OK", "Sure, here you go") or silently dropped half the
# text. Either way the original is the safer thing to keep.
_MIN_LENGTH_RATIO = 0.5

# Only these may disappear. Everything else in the author's speech is content.
#
# Measured 2026-08-10 against gemini-2.5-flash-lite, with a prompt that says in
# bold that rewording and deleting are forbidden:
#
#   light   deleted nothing                                      ✓
#   medium  deleted 呃 AND 因为   ← a causal connective
#   heavy   turned 这个我觉得 into 我觉得这个  ← reordered
#
# Both of those are precisely what 铁律 10 exists to stop, and both read
# perfectly well, which is why re-reading would not catch them. A prompt is a
# request; this is the enforcement.
_FILLERS = (
    "就是说", "然后就是", "那个那个", "这个这个",
    "嗯", "呃", "啊", "哦", "唉", "呐", "額", "额",
    "um", "uh", "erm", "you know", "i mean",
)

# Punctuation and whitespace are the one thing polish IS allowed to change, so
# they are removed before comparing. Covers both widths, plus the paragraph
# breaks the heavy level adds.
_IGNORABLE = re.compile(r"[\s，。？！、；：「」『』（）《》,.?!;:'\"()\[\]—…·-]+")


def content_signature(text: str) -> str:
    """What the sentence says, with everything polish may legally touch removed.

    Two texts with the same signature differ only in punctuation, spacing and
    filler words. Two texts with different signatures differ in *content* —
    a word was added, dropped, swapped or moved — and 铁律 10 forbids all four.
    """
    stripped = _IGNORABLE.sub("", text)
    lowered = stripped.lower()
    for filler in _FILLERS:
        lowered = lowered.replace(filler.lower(), "")
    return lowered


def content_changed(before: str, after: str) -> str | None:
    """None if only punctuation and filler moved; else a short description.

    The description goes into a log line and a warning, because 铁律 10's other
    half is that the author must be able to tell it happened.
    """
    a, b = content_signature(before), content_signature(after)
    if a == b:
        return None

    import difflib

    ops = difflib.SequenceMatcher(None, a, b).get_opcodes()
    dropped = "".join(a[i:j] for tag, i, j, _, _ in ops if tag in ("delete", "replace"))
    added = "".join(b[i:j] for tag, _, _, i, j in ops if tag in ("insert", "replace"))
    parts = []
    if dropped:
        parts.append(f"删掉了「{dropped[:30]}」")
    if added:
        parts.append(f"加上了「{added[:30]}」")
    return "，".join(parts) or "内容变了"


class LlmError(RuntimeError):
    pass


@runtime_checkable
class LlmProvider(Protocol):
    id: str
    display_name: str

    def is_available(self) -> tuple[bool, str]: ...

    def complete(self, system: str, user: str) -> str: ...


_COMMON_RULES = (
    "严格约束：**不得改写措辞、不得增删内容、不得改变任何论断**。"
    "你面对的是学术论证的口述稿，任何看似「更通顺」的改写都可能替换掉作者的论点。"
    "只输出处理后的正文，不要解释、不要加引号、不要写前言。"
)

# The model is never asked to delete anything. Not at any level.
#
# It used to be. "medium" told it to remove 「嗯」「呃」「就是说」, and against a
# real model that instruction also cost the author 「因为」 — a causal
# connective, deleted from an academic argument, in an output that read
# perfectly. Deletion and punctuation have completely different risk profiles:
# punctuation genuinely needs a language model, while removing 「呃」 is a string
# operation we can do exactly, verifiably, and with a list the author can read.
#
# So the model punctuates; `strip_fillers` below does the deleting. The only
# thing that changes between levels is how much *we* then do.
_LEVELS = {
    "light": "你是一个口述稿的标点整理工具。只做一件事：为下面的文字补上正确的标点与断句。",
    "medium": "你是一个口述稿的标点整理工具。只做一件事：为下面的文字补上正确的标点与断句。",
    "heavy": (
        "你是一个口述稿的整理工具。只做两件事：补上正确的标点与断句；"
        "在语义明显转折的地方分段（用空行）。"
    ),
}

#: Removed by us, not by the model. Deliberately tiny.
#:
#: Every entry has to be a string that cannot appear inside an ordinary word,
#: because this runs as a blind substring replace. 「额」 was in an earlier draft
#: and would have turned 额度 into 度; 「呐」 would have broken 呐喊; 「就是说」
#: is a real discourse marker in academic Chinese and removing it changes the
#: prose. All three are gone. When in doubt, leave it in the transcript — a
#: stray 「呃」 is a blemish, a mangled word is a lie.
_STRIP_FILLERS = ("嗯", "呃", "唔")

#: Stammers, collapsed to one rather than deleted.
_STAMMERS = ("那个", "这个", "就是", "然后", "我觉得")

_ENGLISH_FILLER = re.compile(r"\b(um+|uh+|erm+)\b[,.]?\s*", re.IGNORECASE)
#: Punctuation left stranded by a removal: a comma that now opens a clause, or
#: two in a row where a filler used to sit between them.
_ORPHAN_LEAD = re.compile(r"(^|[\n。！？.!?])\s*[，、,]+")
_DOUBLED = re.compile(r"([，。？！、；：,.?!;:])[，、,]+")


def strip_fillers(text: str) -> str:
    """Remove filler words mechanically. Deterministic, and auditable by eye.

    Runs after the content guard has approved the model's punctuation, so the
    only difference between what was checked and what is delivered is this
    function — which is a fixed list and a few regexes, not a model.
    """
    out = text
    for filler in _STRIP_FILLERS:
        out = out.replace(filler, "")
    out = _ENGLISH_FILLER.sub("", out)

    for word in _STAMMERS:
        # 那个那个那个 -> 那个. Only immediate repeats; a word used twice in a
        # sentence is not a stammer.
        pattern = re.compile(f"(?:{re.escape(word)}[，、,]?\\s*){{2,}}")
        out = pattern.sub(word, out)

    out = _DOUBLED.sub(r"\1", out)
    out = _ORPHAN_LEAD.sub(r"\1", out)
    return out.strip()

TRANSLATE_PROMPT = (
    "你是一个学术翻译工具。把下面的英文译成中文。"
    "保持术语一致，保持原文的论证结构与语气。"
    "只输出译文，不要解释、不要附上原文。"
)


def polish_prompt(level: str = "light") -> str:
    """The system prompt for a polish level.

    An unknown level — a hand-edited config, a newer build's value — falls back
    to the most conservative one. Degrading toward the risky end would be a poor
    trade for a typo.
    """
    return f"{_LEVELS.get(level, _LEVELS['light'])}\n{_COMMON_RULES}"


def _build_user_message(text: str, context: str | None, vocabulary: list[str] | None) -> str:
    parts = []
    if vocabulary:
        parts.append(
            "以下是作者的专有名词与常用术语，出现时必须保持这些写法："
            + "、".join(vocabulary)
        )
    if context:
        # §4.1b: the previous paragraph only. Request size therefore stays
        # constant whether the dictation runs three minutes or thirty.
        parts.append(f"【上文，仅供参考，不要重复输出】\n{context}")
    parts.append(f"【需要处理的文字】\n{text}")
    return "\n\n".join(parts)


def _complete(provider, system: str, user: str) -> str:
    if provider is None or not hasattr(provider, "complete"):
        raise LlmError(f"{getattr(provider, 'id', provider)!r} cannot run a prompt")
    try:
        return provider.complete(system, user)
    except LlmError:
        raise
    except Exception as exc:
        raise LlmError(f"{getattr(provider, 'id', '?')}: {exc}") from exc


def polish(
    provider,
    text: str,
    *,
    level: str = "light",
    context: str | None = None,
    vocabulary: list[str] | None = None,
) -> str:
    return _complete(
        provider, polish_prompt(level), _build_user_message(text, context, vocabulary)
    ).strip()


def safe_polish(
    provider,
    text: str,
    *,
    level: str = "light",
    context: str | None = None,
    vocabulary: list[str] | None = None,
) -> tuple[str, bool]:
    """Polish if possible, otherwise hand back exactly what came in.

    铁律 8. The caller finalises the utterance either way; losing a sentence
    because a language model timed out is precisely the failure this project
    refuses. Returns (text, whether it was actually polished).
    """
    if provider is None:
        return text, False

    try:
        result = polish(provider, text, level=level, context=context, vocabulary=vocabulary)
    except LlmError as exc:
        log.warning("polish failed, keeping the raw transcript: %s", exc)
        return text, False

    if not result.strip():
        log.warning("polish returned nothing, keeping the raw transcript")
        return text, False

    if len(result) < len(text) * _MIN_LENGTH_RATIO:
        # The model answered the prompt instead of doing the job, or dropped
        # half the content. Either way, do not silently replace what was said.
        log.warning(
            "polish returned %d chars for %d in, keeping the raw transcript",
            len(result),
            len(text),
        )
        return text, False

    # 铁律 10, enforced rather than requested. Measured on a real model with a
    # prompt that forbids this in bold: medium dropped 「因为」 from a sentence
    # and heavy reordered a clause. Both read beautifully. That is the danger.
    changed = content_changed(text, result)
    if changed is not None:
        log.warning("polish changed the content (%s), keeping the raw transcript", changed)
        return text, False

    if level in ("medium", "heavy"):
        # Ours, after the guard, so the delivered text differs from the checked
        # text only by a fixed list the author can read in the source.
        result = strip_fillers(result)

    return result, True


def translate(provider, text: str, *, context: str | None = None) -> str:
    # A translate-only fallback (deep-translator) has no prompt interface at
    # all, so it is called directly.
    if provider is not None and not hasattr(provider, "complete"):
        if hasattr(provider, "translate"):
            try:
                return provider.translate(text)
            except Exception as exc:
                raise LlmError(f"{getattr(provider, 'id', '?')}: {exc}") from exc

    return _complete(
        provider, TRANSLATE_PROMPT, _build_user_message(text, context, None)
    ).strip()


def safe_translate(provider, text: str, *, context: str | None = None) -> str | None:
    """Translate, or return None. Design §5: a failed translation must never
    take the English with it — v1 wrote "[translation error]" into the
    transcript instead, which is worse than an empty column."""
    try:
        return translate(provider, text, context=context)
    except LlmError as exc:
        log.warning("translation failed: %s", exc)
        return None
