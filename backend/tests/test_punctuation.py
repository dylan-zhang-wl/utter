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


# --- repetition collapse (a real loop, 2026-08-10) ---------------------------


from backend.punctuation import collapse_repetition


def test_collapses_a_decoder_loop():
    """13.4 seconds of speech came back as 「英文是，」 forty times."""
    text, removed = collapse_repetition("好像没有英文呢。" + "英文是， " * 40)
    assert text.count("英文是") == 1
    assert removed == 39


def test_leaves_normal_text_alone():
    source = "多模态语篇的符号资源具有不同的意义潜势，这一点很重要。"
    assert collapse_repetition(source) == (source, 0)


def test_does_not_touch_a_real_short_repeat():
    """哈哈哈 and 好好好 are things people say."""
    assert collapse_repetition("他说哈哈哈然后就走了")[1] == 0


def test_reports_how_much_was_removed():
    _, removed = collapse_repetition("of the model sign " * 30)
    assert removed >= 20


def test_empty_is_safe():
    assert collapse_repetition("") == ("", 0)


# --- closing an utterance (1 in 7 had punctuation, 2026-08-10) ---------------


from backend.punctuation import close_sentence


def test_chinese_gets_a_full_width_stop():
    assert close_sentence("这句话没有标点") == "这句话没有标点。"


def test_english_gets_a_half_width_stop():
    assert close_sentence("this sentence has no stop") == "this sentence has no stop."


def test_text_that_already_ends_properly_is_left_alone():
    for ending in ("好的。", "really?", "什么！", "wait..."):
        assert close_sentence(ending) == ending


def test_a_trailing_comma_is_left_alone():
    """Ending on a comma means the thought is unfinished; closing it would be
    asserting something the author did not."""
    assert close_sentence("首先，") == "首先，"


def test_words_are_never_changed():
    source = "多模态语篇的 semiotic potential"
    assert close_sentence(source).startswith(source)


def test_empty_is_safe():
    assert close_sentence("") == ""
    assert close_sentence("   ") == "   "
