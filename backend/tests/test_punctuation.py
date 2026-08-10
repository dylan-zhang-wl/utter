"""Punctuation width at a script boundary — the author's "中英文之间的标点
区分不明显".

Mechanical on purpose. Handing this to an LLM would mean letting a model rewrite
the text to fix a typographic detail, which is the trade 铁律 10 refuses.
"""

from backend.punctuation import normalise


def test_chinese_takes_full_width():
    assert normalise("这句话是什么意思?") == "这句话是什么意思？"


def test_english_takes_half_width():
    assert normalise("What does this mean？") == "What does this mean?"


def test_a_mixed_sentence_gets_both():
    """The actual complaint: one utterance, two scripts, one punctuation style."""
    got = normalise("他说,I want to demonstrate that，and then 他停下了.")
    assert got == "他说，I want to demonstrate that, and then 他停下了。"


def test_english_inside_chinese_keeps_its_own_marks():
    assert normalise("作者引用了 Kress and van Leeuwen(2001)，然后展开") == (
        "作者引用了 Kress and van Leeuwen(2001)，然后展开"
    )


def test_space_before_a_full_width_mark_is_removed():
    assert normalise("这句话 。") == "这句话。"


def test_space_before_a_half_width_mark_is_removed():
    assert normalise("this clause , and then") == "this clause, and then"


def test_a_leading_mark_is_left_alone():
    """Nothing precedes it, so there is no evidence either way."""
    assert normalise("，开头") == "，开头"


def test_words_are_never_touched():
    """The only thing this may change is the width of a punctuation mark."""
    source = "多模态语篇的 semiotic potential 指的是什么"
    assert normalise(source) == source


def test_empty_and_none_safe():
    assert normalise("") == ""


def test_digits_count_as_latin():
    assert normalise("见 2001，页 45") == "见 2001, 页 45"


def test_idempotent():
    once = normalise("他说,I want to demonstrate，and 他停下了.")
    assert normalise(once) == once


def test_decimals_are_not_broken_up():
    """3.14 and 1,000 are numbers, not sentences."""
    assert normalise("误差 3.14 以内") == "误差 3.14 以内"
    assert normalise("共 1,000 条") == "共 1,000 条"
