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


# --- what a real talk about 「Democracy 2.0」 turned up --------------------------


@pytest.mark.parametrize("text, expected", [
    # the stop belongs to the number, not to the thought
    ("Democracy 2.0 is the phase where we expand who is included.",
     ["Democracy 2.0 is the phase where we expand who is included."]),
    ("It cost 3.5 million. That is a lot.",
     ["It cost 3.5 million.", "That is a lot."]),
    # an initial being spelled out, and the lower-case word that follows it
    ("Anderson argues that in fact the U.S. was not a democracy until 1965.",
     ["Anderson argues that in fact the U.S. was not a democracy until 1965."]),
    # but a sentence that genuinely ends on an abbreviation still ends
    ("We visited the U.S. Then we flew home.",
     ["We visited the U.S.", "Then we flew home."]),
    ("J. R. R. Tolkien wrote it. Then he revised it.",
     ["J. R. R. Tolkien wrote it.", "Then he revised it."]),
    # ordinary sentences are untouched
    ("She left. He stayed.", ["She left.", "He stayed."]),
    ("Dr. Smith arrived. Then we began.", ["Dr. Smith arrived.", "Then we began."]),
    ("Who counts? That is the question.", ["Who counts?", "That is the question."]),
])
def test_a_full_stop_is_not_always_the_end_of_a_sentence(text, expected):
    """Every one of these arrived broken in a 52-entry session: 「Democracy 2.」
    as one entry and 「0, Far-Right extremists…」 as the next, with the
    translator dutifully rendering the stray 「0，」."""
    from backend.sentences import split_sentences

    assert split_sentences(text)[0] == expected


# --- the full stop the model adds because the audio ran out ---------------------


def test_a_sentence_cut_at_the_ceiling_waits_for_the_rest_of_itself():
    """Whisper finishes whatever it is given with a full stop, spoken or not.
    Fifteen of fifty-two entries in a real session began with a lower-case
    word, which is what a sentence chopped in half looks like."""
    from backend.sentences import SentenceAssembler

    assembler = SentenceAssembler()

    assert assembler.feed("It would be more of a revelation to me.", cut=True) == []
    assert assembler.feed("than it was to you.") == [
        "It would be more of a revelation to me than it was to you."]


def test_a_capital_letter_says_the_full_stop_was_real_after_all():
    """The other half of the same guess. The speaker did finish at the ceiling,
    and the next chunk proves it by starting a new sentence."""
    from backend.sentences import SentenceAssembler

    assembler = SentenceAssembler()

    assert assembler.feed("Democracy needs to evolve again.", cut=True) == []
    assert assembler.feed("I want us to think about democracy in phases.") == [
        "Democracy needs to evolve again.",
        "I want us to think about democracy in phases."]


def test_only_the_last_sentence_of_a_cut_chunk_waits():
    """Everything before it ended where the speaker ended it."""
    from backend.sentences import SentenceAssembler

    assembler = SentenceAssembler()

    got = assembler.feed("They're not. We were used to it. But separate.", cut=True)

    assert got == ["They're not.", "We were used to it."]
    assert assembler.pending == "But separate"


def test_a_natural_pause_is_still_trusted_immediately():
    """The whole point of waiting for the speaker to stop. A chunk that ended
    because they stopped talking must not gain a chunk of latency."""
    from backend.sentences import SentenceAssembler

    assembler = SentenceAssembler()

    assert assembler.feed("The cast and coaches were also patient with me.") == [
        "The cast and coaches were also patient with me."]


def test_the_end_of_a_meeting_gives_the_full_stop_back():
    """铁律 8: nothing is lost, and it should not read as unfinished either."""
    from backend.sentences import SentenceAssembler

    assembler = SentenceAssembler()
    assembler.feed("We still tend to focus more.", cut=True)

    assert assembler.flush() == ["We still tend to focus more."]


def test_a_cut_chunk_ending_on_a_question_mark_is_left_alone():
    """A question mark is not the model's default filler; a full stop is."""
    from backend.sentences import SentenceAssembler

    assembler = SentenceAssembler()

    assert assembler.feed("Who counts?", cut=True) == ["Who counts?"]


@pytest.mark.parametrize("junk", ["...", ". . .", "—", "?!"])
def test_punctuation_alone_is_not_a_sentence(junk):
    """「...」 was entry #0 of a real session, and went to the translator."""
    from backend.sentences import split_sentences

    assert split_sentences(junk)[0] == []


def test_a_real_sentence_around_the_punctuation_still_survives():
    from backend.sentences import split_sentences

    assert split_sentences("... And then she spoke.")[0] == ["And then she spoke."]
