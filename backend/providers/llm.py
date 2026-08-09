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

_LEVELS = {
    "light": "你是一个口述稿的标点整理工具。只做一件事：为下面的文字补上正确的标点与断句。",
    "medium": (
        "你是一个口述稿的清理工具。只做三件事：补标点与断句；"
        "删除「嗯」「呃」「就是说」这类口水词；删除说话人自我重复的部分。"
    ),
    "heavy": (
        "你是一个口述稿的整理工具。只做四件事：补标点与断句；删除口水词；"
        "删除自我重复；按语义分段。"
    ),
}

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
