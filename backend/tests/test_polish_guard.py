"""铁律 10, enforced mechanically rather than requested politely.

Every string here came out of gemini-2.5-flash-lite on 2026-08-10, run against
a prompt that forbids rewording and deleting **in bold**. It did both anyway,
and both results read beautifully — which is the entire problem. The author
dictates academic argument; a dropped 「因为」 changes a causal claim into a
juxtaposition, and re-reading will not catch it.
"""

import pytest

from backend.providers.llm import (
    apply_punctuation,
    content_changed,
    content_signature,
    safe_polish,
)

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


def test_safe_polish_keeps_the_words_and_takes_the_punctuation():
    """It used to reject the whole polish when content moved, which in practice
    rejected almost every one — the author switched polish on and saw nothing
    change. Now the model's punctuation is merged onto the transcript's words."""
    result, polished = safe_polish(_Model(MEDIUM), RAW)

    assert polished is True
    assert content_signature(result) == content_signature(RAW), "因为 came back"
    assert "，" in result or "？" in result, "and the punctuation was still taken"


def test_a_model_that_answers_instead_of_editing_still_yields_the_raw_text():
    """No punctuation to merge means nothing to gain, so the transcript wins."""
    result, polished = safe_polish(_Model("好的"), RAW)
    assert result == RAW
    assert polished is False


def test_content_is_never_invented_even_when_the_model_adds_a_sentence():
    reply = LIGHT + "另外我补充一句作者没说过的话。"
    result, _ = safe_polish(_Model(reply), RAW)
    assert content_signature(result) == content_signature(RAW)
    assert "补充一句" not in result


def test_safe_polish_accepts_a_clean_punctuation_pass():
    result, polished = safe_polish(_Model(LIGHT), RAW)
    assert result == LIGHT
    assert polished is True


# --- filler removal, done by us rather than by the model -----------------------


def test_a_leading_filler_takes_its_comma_with_it():
    from backend.providers.llm import strip_fillers

    assert strip_fillers("呃，因为韦努蒂谈的是印刷时代。") == "因为韦努蒂谈的是印刷时代。"


def test_a_stammer_collapses_rather_than_vanishing():
    """那个那个那个 was one word said three times, not three words."""
    from backend.providers.llm import strip_fillers

    assert strip_fillers("他还是说出了那个那个那个我是故意说的。") == (
        "他还是说出了那个我是故意说的。"
    )


def test_english_fillers_go_too():
    from backend.providers.llm import strip_fillers

    assert strip_fillers("Um, I will uh lock in candidate two.") == (
        "I will lock in candidate two."
    )


@pytest.mark.parametrize(
    "text",
    [
        "这个额度还有多少？",          # 额 — an earlier draft would have left 「度」
        "前额也疼。",                  # 额 again, at the other end of a word
        "他呐喊了一声。",              # 呐
        "也就是说，这个概念要重新界定。",  # 就是说 — a real discourse marker
        "啊里巴巴和四十大盗。",         # 啊
    ],
)
def test_words_that_merely_contain_a_filler_are_untouched(text):
    """This runs as a blind substring replace, so every entry on the list has
    to be a string that cannot appear inside an ordinary word. A stray 「呃」 is
    a blemish; a mangled word is a lie."""
    from backend.providers.llm import strip_fillers

    assert strip_fillers(text) == text


def test_light_leaves_filler_alone():
    """By design, and the author should not have to guess: 「轻」 is punctuation
    only. This is why their 嗯 and 呃 survived the first real test."""
    result, polished = safe_polish(_Model("呃，我觉得是这样。"), "呃我觉得是这样", level="light")
    assert result == "呃，我觉得是这样。"
    assert polished is True


def test_medium_removes_filler_without_asking_the_model_to():
    """The model returns punctuation only; the deletion is ours. Its prompt no
    longer contains the instruction that cost the author 「因为」."""
    result, polished = safe_polish(_Model("呃，我觉得是这样。"), "呃我觉得是这样", level="medium")
    assert result == "我觉得是这样。"
    assert polished is True


def test_the_model_is_never_told_to_delete_anything():
    from backend.providers.llm import polish_prompt

    for level in ("light", "medium", "heavy"):
        assert "删除" not in polish_prompt(level), level


# --- merging rather than judging ----------------------------------------------


@pytest.mark.parametrize(
    "raw,model",
    [
        # Every one of these came out of a real run.
        ("这个我觉得是需要重新界定的", "我觉得这个是需要重新界定的。"),      # reordered
        ("他说这样不行", "他说：“这样不行。”"),                            # added quotes
        ("我现在用那个用的那个但是如果是用ADC的", "我现在用的？如果是用ADC的。"),  # deleted
    ],
)
def test_the_merged_result_always_has_the_transcripts_content(raw, model):
    from backend.providers.llm import content_signature as sig

    assert sig(apply_punctuation(raw, model)) == sig(raw)


def test_curly_quotes_do_not_count_as_content():
    """They were missing from the ignore set, so every utterance the model put
    a 「“」 into was reported as content added and thrown away."""
    assert content_changed("他说这样不行", "他说：“这样不行。”") is None


def test_whisper_punctuation_is_not_lost_to_a_worse_model_output():
    """Whisper's Chinese punctuation is sparse but real. A model that returns
    less of it has nothing to offer."""
    raw = "第一句。第二句。第三句。"
    assert apply_punctuation(raw, "第一句 第二句 第三句") == raw


# --- translation levels (P3 task 2) ---------------------------------------------


def test_every_translation_level_forbids_inventing_text():
    """The boundary that separates an honest translation from a fluent
    invention. Dictation already produced the cautionary case: the nonsense
    「我元册圆」 came back as the plausible 「语言核语言」. Garbled text
    announces itself; a fluent invention does not."""
    from backend.providers.llm import translate_prompt

    for level in ("literal", "fluent", "explain"):
        prompt = translate_prompt(level)
        assert "听不清" in prompt, f"{level} 没说听不清怎么办"
        assert "不要编" in prompt, f"{level} 没禁止编造"


def test_the_levels_differ_in_how_much_they_may_join_up():
    from backend.providers.llm import translate_prompt

    literal, fluent, explain = (translate_prompt(x)
                                for x in ("literal", "fluent", "explain"))
    assert "保持残缺" in literal
    assert "补成通顺" in fluent
    assert "括号里加" in explain
    assert literal != fluent != explain


def test_an_unknown_level_falls_back_to_the_middle_one():
    """A typo in a config file must not silently turn off the guard rails."""
    from backend.providers.llm import translate_prompt

    assert translate_prompt("nonsense") == translate_prompt("fluent")


def test_translation_carries_the_vocabulary_through():
    """Second use of the same glossary: it is what stops `foreignisation`
    coming back transliterated instead of as 异化."""
    from backend.providers import llm

    seen = {}

    class Fake:
        id = "fake"

        def complete(self, system, user):
            seen["system"], seen["user"] = system, user
            return "异化"

    out = llm.translate(Fake(), "foreignisation", vocabulary=["异化"], level="fluent")

    assert out == "异化"
    assert "异化" in seen["user"]
    assert "听不清" in seen["system"]


def test_a_failed_translation_returns_none_rather_than_poisoning_the_record():
    """v1 wrote "[translation error]" into the transcript. An empty column is
    honest; a sentence that is not a translation is not."""
    from backend.providers import llm

    class Broken:
        id = "broken"

        def complete(self, system, user):
            raise RuntimeError("no network")

    assert llm.safe_translate(Broken(), "hello") is None
