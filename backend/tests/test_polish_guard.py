"""铁律 10, enforced mechanically rather than requested politely.

Every string here came out of gemini-2.5-flash-lite on 2026-08-10, run against
a prompt that forbids rewording and deleting **in bold**. It did both anyway,
and both results read beautifully — which is the entire problem. The author
dictates academic argument; a dropped 「因为」 changes a causal claim into a
juxtaposition, and re-reading will not catch it.
"""

import pytest

from backend.providers.llm import content_changed, content_signature, safe_polish

RAW = (
    "所以我这一节想讨论的其实是异化和归化这两个概念在数字人文的语境下会不会失效"
    "呃因为韦努蒂当年谈的是印刷时代的翻译而我们现在面对的是机器翻译的输出"
    "这个我觉得是需要重新界定的"
)
LIGHT = (
    "所以我这一节想讨论的其实是异化和归化这两个概念，在数字人文的语境下会不会失效。"
    "呃，因为韦努蒂当年谈的是印刷时代的翻译，而我们现在面对的是机器翻译的输出，"
    "这个我觉得是需要重新界定的。"
)
MEDIUM = (  # dropped 因为
    "所以我这一节想讨论的其实是异化和归化这两个概念在数字人文的语境下会不会失效？"
    "韦努蒂当年谈的是印刷时代的翻译，而我们现在面对的是机器翻译的输出，"
    "这个我觉得是需要重新界定的。"
)
HEAVY = (  # reordered 这个我觉得 -> 我觉得这个
    "所以我这一节想讨论的其实是异化和归化这两个概念在数字人文的语境下会不会失效？"
    "因为韦努蒂当年谈的是印刷时代的翻译，而我们现在面对的是机器翻译的输出。"
    "我觉得这个是需要重新界定的。"
)


def test_adding_punctuation_is_allowed():
    assert content_changed(RAW, LIGHT) is None


def test_removing_filler_is_allowed():
    assert content_changed("嗯我觉得这个说法有问题", "我觉得这个说法有问题。") is None


def test_paragraph_breaks_are_allowed():
    assert content_changed("第一点如此第二点如彼", "第一点如此。\n\n第二点如彼。") is None


def test_dropping_a_connective_is_caught():
    """The real medium output. 「因为」 is the argument, not filler."""
    changed = content_changed(RAW, MEDIUM)
    assert changed is not None and "因为" in changed


def test_reordering_a_clause_is_caught():
    """The real heavy output. Nothing was added or lost — the words moved, and
    a length check or a word-set check would both have waved it through."""
    assert content_changed(RAW, HEAVY) is not None


def test_a_substituted_word_is_caught():
    """The failure mode from the docstring of llm.py, made a test."""
    assert content_changed("翻译不是复制，而是一种重写", "翻译是一种创造性的转换过程") is not None


def test_a_helpfully_corrected_term_is_still_caught():
    """Even when the model is RIGHT. 铁律 10 has no exception for improvements:
    an author who cannot trust the text to be theirs has to re-read every word
    against the audio, and the audio is gone."""
    assert content_changed("异化和规化是韦努蒂的概念", "异化和归化是韦努蒂的概念") is not None


def test_signature_ignores_punctuation_width():
    assert content_signature("他说,好的.") == content_signature("他说，好的。")


class _Model:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user):
        return self.reply


def test_safe_polish_keeps_the_original_when_content_moved():
    result, polished = safe_polish(_Model(MEDIUM), RAW)
    assert result == RAW
    assert polished is False


def test_safe_polish_accepts_a_clean_punctuation_pass():
    result, polished = safe_polish(_Model(LIGHT), RAW)
    assert result == LIGHT
    assert polished is True
