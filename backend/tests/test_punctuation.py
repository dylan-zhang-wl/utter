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


# --- Whisper's stock hallucinations --------------------------------------------


import pytest

from backend.punctuation import is_hallucination


@pytest.mark.parametrize("text", [
    "字幕志愿者 李宗盛。",      # found in the author's own archive, injected into a document
    "谢谢观看",
    "感谢观看，下次再见",
    "请不吝点赞订阅转发打赏",
    "Thanks for watching!",
    "Please subscribe.",
])
def test_a_subtitle_credit_is_not_something_the_author_said(text):
    """Whisper's Chinese training data is largely video captions, so when there
    is nothing to transcribe it reaches for the credits. The VAD gate answers
    "is anyone talking", not "is this output real", and a cough gets through."""
    assert is_hallucination(text) is True


@pytest.mark.parametrize("text", [
    "这一节我要讨论字幕志愿者这个群体在数字人文里的位置",
    "谢谢观看我的论文答辩，下面进入提问环节",
    "我要订阅这本期刊",
    "我刚才说的大语言模型，它就还是根据那个听写的质量变的。",
])
def test_a_real_sentence_that_mentions_one_is_kept(text):
    """The first version asked "does it contain a stock phrase" with a length
    limit, and would have thrown the first of these away. Discarding something
    the author actually said is far worse than letting one stray line through
    (铁律 8), so the test comes with the rule."""
    assert is_hallucination(text) is False


def test_an_empty_transcript_is_not_a_hallucination():
    assert is_hallucination("   ") is False


# --- streamed clauses are closed by the speaker's own pause ---------------------


def test_a_breath_is_a_comma():
    from backend.punctuation import punctuate_pause

    assert punctuate_pause("而且", 400) == "而且，"


def test_a_long_pause_is_a_full_stop():
    from backend.punctuation import punctuate_pause

    assert punctuate_pause("还是比较多的", 1200) == "还是比较多的。"


def test_the_end_of_a_session_always_closes():
    from backend.punctuation import punctuate_pause

    assert punctuate_pause("是什么情况", 200, final=True) == "是什么情况。"


def test_whispers_own_punctuation_is_left_alone():
    """It knows more than the pause does when it bothers to answer."""
    from backend.punctuation import punctuate_pause

    assert punctuate_pause("好的。", 400) == "好的。"
    assert punctuate_pause("是吗？", 1500) == "是吗？"


def test_an_unknown_gap_takes_the_conservative_mark():
    """The first clause of a session has nothing before it. A comma can be
    followed by more; a full stop cannot be taken back (铁律 9)."""
    from backend.punctuation import punctuate_pause

    assert punctuate_pause("那现在", None) == "那现在，"


def test_an_empty_clause_is_left_empty():
    from backend.punctuation import punctuate_pause

    assert punctuate_pause("   ", 400).strip() == ""
