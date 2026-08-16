"""Sentences cut from the text, not from the silence.

Two rounds of tuning the VAD produced fragments and then blocks, because a
speaker pauses the same length between phrases as between sentences. Whisper
already punctuates; these tests are about using that instead of guessing.
"""

import pytest

from backend.sentences import SentenceAssembler, split_sentences


# --- splitting ------------------------------------------------------------------


def test_a_finished_sentence_comes_out_whole():
    done, rest = split_sentences("So the question of equivalence is not about words.")
    assert done == ["So the question of equivalence is not about words."]
    assert rest == ""


def test_several_sentences_are_separated():
    done, rest = split_sentences("First one. Second one! Third one?")
    assert done == ["First one.", "Second one!", "Third one?"]
    assert rest == ""


def test_an_unfinished_tail_is_returned_separately():
    done, rest = split_sentences("That is settled. But my father")
    assert done == ["That is settled."]
    assert rest == "But my father"


def test_chinese_punctuation_counts_too():
    done, rest = split_sentences("这是第一句。这是第二句！还有")
    assert done == ["这是第一句。", "这是第二句！"]
    assert rest == "还有"


def test_an_abbreviation_does_not_end_a_sentence():
    """The full stop in 「Dr. Venuti」 is not a sentence break, and cutting
    there would hand the translator half a name."""
    done, rest = split_sentences("As Dr. Venuti argues, fluency is an ideology.")
    assert done == ["As Dr. Venuti argues, fluency is an ideology."]


def test_a_closing_quote_stays_with_its_sentence():
    done, _rest = split_sentences('He said "that is the point." Then he stopped.')
    assert done[0].endswith('"')


def test_nothing_in_nothing_out():
    assert split_sentences("") == ([], "")
    assert split_sentences("   ") == ([], "")


# --- assembling across chunks ---------------------------------------------------


def test_a_sentence_split_across_two_chunks_is_rejoined():
    """The exact failure from a real session: 「My father」 and 「from equity
    states」 arrived as separate entries because the speaker breathed."""
    assembler = SentenceAssembler()

    assert assembler.feed("My father") == []
    assert assembler.feed("came from a quiet state in the southwest.") == [
        "My father came from a quiet state in the southwest."]


def test_complete_sentences_are_released_immediately():
    """Holding everything would trade one latency problem for another."""
    assembler = SentenceAssembler()

    out = assembler.feed("This one is finished. And this one is not")
    assert out == ["This one is finished."]
    assert assembler.pending == "And this one is not"


def test_the_end_of_a_meeting_releases_whatever_is_held():
    """铁律 8's shape here: the speaker stopping mid-thought must not cost the
    thought."""
    assembler = SentenceAssembler()
    assembler.feed("A thought that never finished")

    assert assembler.flush() == ["A thought that never finished"]
    assert assembler.pending == ""


def test_a_speaker_who_never_stops_is_not_held_for_ever():
    """Some people talk in one long clause. A late sentence beats one that
    never arrives."""
    assembler = SentenceAssembler(max_held_chars=60)

    out = []
    for _ in range(5):
        out += assembler.feed("and then we kept going without ever stopping")

    assert out, "到了上限还不放出来"
    assert len(assembler.pending) <= 60


def test_flushing_an_empty_assembler_is_harmless():
    assert SentenceAssembler().flush() == []


def test_three_chunks_of_a_talk_come_out_as_sentences():
    """End to end on the shape a real session produces."""
    assembler = SentenceAssembler()
    chunks = [
        "So the question of equivalence is not really about words. Venuti",
        "argues that fluency is itself an ideology. If we accept that,",
        "then foreignisation is a political choice.",
    ]
    out = []
    for chunk in chunks:
        out += assembler.feed(chunk)
    out += assembler.flush()

    assert out == [
        "So the question of equivalence is not really about words.",
        "Venuti argues that fluency is itself an ideology.",
        "If we accept that, then foreignisation is a political choice.",
    ]
