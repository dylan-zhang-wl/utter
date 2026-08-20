"""P3 task 2 — batched, serial, order-safe translation.

The tests that matter most here are the misalignment ones. Batching buys the
round trips back and introduces exactly one new way to be wrong: line 12's
Chinese ending up under line 11's English. That failure is invisible — both
lines look like perfectly good sentences — so it gets more tests than the
happy path.
"""

import threading

import pytest

from backend.translator import TranslationQueue, format_batch, parse_batch


class Clock:
    """A hand-wound clock, so 'has it waited long enough' is not a sleep()."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def build(translate, **kwargs):
    got = {}
    q = TranslationQueue(translate, lambda i, t: got.__setitem__(i, t), **kwargs)
    return q, got


# --- numbering, which is what keeps the lines attached to the right English ------


def test_a_batch_goes_out_numbered():
    assert format_batch(["one", "two"]) == "1. one\n2. two"


def test_a_clean_reply_matches_back_by_number():
    assert parse_batch("1. 第一\n2. 第二", 2) == {0: "第一", 1: "第二"}


def test_a_reply_that_skips_a_number_costs_only_that_line():
    """The whole reason for numbering. Zipping two lists would shift every
    translation after the gap onto the wrong sentence."""
    assert parse_batch("1. 第一\n3. 第三", 3) == {0: "第一", 2: "第三"}


def test_a_reply_with_extra_numbers_cannot_write_past_the_batch():
    assert parse_batch("1. 第一\n2. 第二\n3. 多出来的", 2) == {0: "第一", 1: "第二"}


def test_a_duplicated_number_keeps_the_first_and_ignores_the_rest():
    assert parse_batch("1. 第一\n1. 又来一个", 1) == {0: "第一"}


def test_an_unnumbered_reply_translates_nothing_rather_than_guessing():
    """A model that ignores the format gives us no way to know which sentence
    is which, and a wrong pairing is worse than an empty column."""
    assert parse_batch("第一\n第二", 2) == {}


def test_a_batch_of_one_is_accepted_unnumbered():
    """Measured against gpt-5.4-mini, not assumed: three numbered lines come
    back numbered, and a single numbered line comes back bare — the model
    decides the number is clutter. Every max_wait flush, every 暂停 and the
    final flush at 结束 can carry exactly one entry, so rejecting this would
    have silently dropped the last sentence of every meeting.

    Safe for the same reason the numbering exists: with one entry there is
    nothing to misalign it against.
    """
    assert parse_batch("因此，等值问题其实根本不是一个关于词语的问题。", 1) == {
        0: "因此，等值问题其实根本不是一个关于词语的问题。"}


def test_a_batch_of_one_still_prefers_the_number_when_there_is_one():
    assert parse_batch("1. 第一", 1) == {0: "第一"}


def test_a_wrapped_single_translation_is_joined_rather_than_truncated():
    assert parse_batch("因此，等值问题\n并不是关于词语的。", 1) == {
        0: "因此，等值问题\n并不是关于词语的。"}


def test_an_empty_reply_for_one_entry_translates_nothing():
    assert parse_batch("   \n  ", 1) == {}


def test_two_entries_still_refuse_an_unnumbered_reply():
    """The leniency is for batches of one only. With two, pairing by position
    is exactly the mistake the numbering exists to prevent."""
    assert parse_batch("第一\n第二", 2) == {}


def test_several_punctuation_styles_of_numbering_are_accepted():
    """Models emit 1. / 1、/ 1: fairly interchangeably."""
    assert parse_batch("1、第一\n2: 第二\n3) 第三", 3) == {
        0: "第一", 1: "第二", 2: "第三"}


# --- batching -------------------------------------------------------------------


def test_nothing_goes_out_until_the_batch_is_full():
    calls = []
    q, _ = build(lambda text, ctx: calls.append(text) or "1. 一", batch_size=3)

    q.submit(0, "one")
    q.submit(1, "two")
    assert q.drain_once() == 0, "不该为了两句就发一次请求"
    assert calls == []

    q.submit(2, "three")
    assert q.drain_once() == 3
    assert len(calls) == 1


def test_a_partial_batch_goes_out_once_it_has_waited_long_enough():
    """Otherwise the last sentence before a long silence sits in the queue
    until somebody speaks again."""
    clock = Clock()
    calls = []
    q, _ = build(lambda text, ctx: calls.append(text) or "1. 一",
                 batch_size=3, max_wait=8.0, clock=clock)

    q.submit(0, "one")
    assert q.drain_once() == 0

    clock.advance(9.0)
    assert q.drain_once() == 1
    assert calls


def test_flush_empties_the_queue_for_pause_and_stop():
    q, got = build(lambda text, ctx: "\n".join(
        f"{i + 1}. 译{i + 1}" for i in range(len(text.splitlines()))), batch_size=3)
    for i in range(5):
        q.submit(i, f"line {i}")

    q.flush()

    assert q.pending() == 0
    assert len(got) == 5


# --- 铁律 8 and 11 --------------------------------------------------------------


def test_a_failing_model_costs_the_translation_and_nothing_else():
    def boom(text, ctx):
        raise RuntimeError("no network")

    q, got = build(boom, batch_size=1)
    q.submit(0, "one")

    assert q.drain_once() == 1, "这一批应该被消化掉，而不是永远卡在队列里"
    assert got == {}, "翻译失败时不该回填任何东西"


def test_only_one_request_is_ever_in_flight(monkeypatch):
    """铁律 11. Two overlapping requests can return in either order, and the
    transcript is an ordered document."""
    overlap = []
    live = threading.Semaphore(1)

    def slow(text, ctx):
        got = live.acquire(blocking=False)
        overlap.append(got)
        try:
            return "1. 一"
        finally:
            if got:
                live.release()

    q, _ = build(slow, batch_size=1)
    for i in range(5):
        q.submit(i, f"line {i}")
    q.flush()

    assert all(overlap), "有两个请求同时在飞"


def test_the_model_sees_the_previous_exchange_as_context():
    """A cut-off clause is often only resolvable from what came before."""
    seen = []

    def translate(text, ctx):
        seen.append(ctx)
        return "\n".join(f"{i + 1}. 译" for i in range(len(text.splitlines())))

    q, _ = build(translate, batch_size=1)
    q.submit(0, "Venuti argues that")
    q.flush()
    q.submit(1, "fluency is itself an ideology")
    q.flush()

    assert seen[0] is None, "第一批没有上文"
    assert "Venuti argues that" in seen[1]


def test_context_does_not_grow_with_the_length_of_the_meeting():
    """Request size must stay flat whether the meeting runs ten minutes or
    two hours."""
    sizes = []

    def translate(text, ctx):
        sizes.append(len(ctx or ""))
        return "1. 译"

    q, _ = build(translate, batch_size=1)
    for i in range(30):
        q.submit(i, f"sentence number {i} with some length to it")
        q.flush()

    assert max(sizes) < 400, f"上文长到了 {max(sizes)} 字符"
    assert sizes[-1] <= sizes[5] * 1.5


def test_entries_are_filled_back_against_their_own_index():
    """The index is the session's, not the batch's."""
    q, got = build(lambda text, ctx: "1. 一\n2. 二", batch_size=2)
    q.submit(41, "one")
    q.submit(42, "two")
    q.flush()

    assert got == {41: "一", 42: "二"}


# --- script contamination -------------------------------------------------------


def test_a_word_the_model_rendered_into_another_script_is_caught():
    """Seen once in 78 sentences of a real talk: the model turned 「不再」 into
    Georgian and carried on in Chinese — 「直到我 აღარ需要再去想它」."""
    from backend.translator import foreign_script

    assert foreign_script("直到我 აღარ需要再去想它。",
                          "until I no longer have to think about it.") == "აღრ"


@pytest.mark.parametrize("translation, source", [
    ("演员和教练们也都对我很有耐心。", "The cast and coaches were also patient."),
    ("剧场本身就是一个自成一体的世界。", "The theatre is a world of its own."),
    # ASCII, digits and CJK punctuation are all ordinary
    ("50年前，就在离这里不到一英里的地方。", "50 years ago, just under a mile."),
    # a term the English itself carried is not the model's invention
    ("这就是所谓的 Sprachgefühl，语感。", "This is what is called Sprachgefühl."),
    ("用 α 表示显著性水平。", "Significance is denoted α."),
])
def test_ordinary_translations_are_left_alone(translation, source):
    """The author writes about translation; a quotation in another script is
    normal. Only script the model introduced by itself counts."""
    from backend.translator import foreign_script

    assert foreign_script(translation, source) == ""


def test_a_contaminated_line_is_translated_again():
    from backend.translator import TranslationQueue

    replies = iter(["1. 直到我 აღარ需要再去想它。", "1. 直到我不再需要去想它。"])
    got = {}
    queue = TranslationQueue(lambda text, context=None: next(replies),
                             lambda i, t: got.__setitem__(i, t),
                             batch_size=1, max_wait=0.0)
    queue.submit(0, "until I no longer have to think about it.")
    queue.drain_once(force=True)

    assert got[0] == "直到我不再需要去想它。"


def test_a_retry_that_is_still_wrong_keeps_the_first_answer():
    """铁律 8's shape: a flawed translation beats none, and the English is safe
    either way."""
    from backend.translator import TranslationQueue

    got = {}
    queue = TranslationQueue(lambda text, context=None: "1. 直到我 აღარ需要再去想它。",
                             lambda i, t: got.__setitem__(i, t),
                             batch_size=1, max_wait=0.0)
    queue.submit(0, "until I no longer have to think about it.")
    queue.drain_once(force=True)

    assert "需要再去想它" in got[0]
