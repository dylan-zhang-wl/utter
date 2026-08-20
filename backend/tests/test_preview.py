"""The grey tail — a preview that is never allowed to become the record.

Two earlier designs put the big model on a fast tick and both failed in ways
that reached the transcript: text from truncated audio got committed with
invented words (「Chrissy was weak」 for 「when democracy was weak」), and a
buffer trimmed at an unknown audio offset lost whole sentences. This layer
cannot do either, and these tests are what holds that.
"""

import numpy as np
import pytest

from backend.preview import (MIN_SECONDS, SAMPLE_RATE, PreviewStream,
                             trim_partial_tail)


def speech(seconds: float) -> np.ndarray:
    return np.full(int(seconds * SAMPLE_RATE), 0.1, dtype=np.float32)


def test_it_shows_what_the_small_model_heard():
    shown = []
    stream = PreviewStream(lambda audio, language=None: "so the question of it.",
                           lambda: speech(3), shown.append)

    stream.once()

    assert shown == ["so the question of it."]


def test_nothing_is_shown_when_nobody_is_speaking():
    shown = []
    stream = PreviewStream(lambda audio, language=None: "phantom",
                           lambda: None, shown.append)

    assert stream.once() == ""
    assert shown == []


def test_a_fragment_too_short_to_read_is_not_transcribed():
    """Whisper on a quarter of a second mostly invents, and there is nothing
    worth reading there anyway."""
    calls = []

    def transcribe(audio, language=None):
        calls.append(1)
        return "you"

    stream = PreviewStream(transcribe, lambda: speech(MIN_SECONDS / 2), lambda _t: None)
    stream.once()

    assert not calls, "太短的片段不该送进模型"


def test_the_same_text_twice_does_not_redraw():
    """A preview that repaints every second with identical text makes the
    transcript flicker and steals the selection."""
    shown = []
    stream = PreviewStream(lambda audio, language=None: "steady text.",
                           lambda: speech(3), shown.append)

    stream.once(); stream.once(); stream.once()

    assert shown == ["steady text."]


def test_growing_text_redraws():
    texts = iter(["so the question", "so the question of", "so the question of x"])
    shown = []
    stream = PreviewStream(lambda audio, language=None: next(texts),
                           lambda: speech(3), shown.append)

    stream.once(); stream.once(); stream.once()

    # each round shows one word less than it heard; see trim_partial_tail
    assert shown == ["so the", "so the question", "so the question of"]


def test_clearing_tells_the_screen_to_drop_it():
    shown = []
    stream = PreviewStream(lambda audio, language=None: "half a sentence.",
                           lambda: speech(3), shown.append)
    stream.once()

    stream.clear()

    assert shown == ["half a sentence.", ""]


def test_a_very_long_buffer_is_cut_to_whisper_s_window():
    """Past thirty seconds the model chunks internally and the preview starts
    disagreeing with itself."""
    seen = []

    def transcribe(audio, language=None):
        seen.append(len(audio) / SAMPLE_RATE)
        return "text"

    PreviewStream(transcribe, lambda: speech(90), lambda _t: None).once()

    assert seen and seen[0] <= 28.0


def test_a_model_that_throws_does_not_stop_the_meeting():
    """铁律 8's spirit: the record must not depend on the luxury."""
    def transcribe(audio, language=None):
        raise RuntimeError("model fell over")

    stream = PreviewStream(transcribe, lambda: speech(3), lambda _t: None)
    stream.start()
    try:
        pass
    finally:
        stream.stop()          # must return, not hang or raise


def test_the_cost_is_recorded():
    stream = PreviewStream(lambda audio, language=None: "text",
                           lambda: speech(3), lambda _t: None)
    stream.once()

    assert stream.rounds == 1
    assert stream.model_seconds >= 0.0


# --- the controller's half ------------------------------------------------------


def _controller():
    from backend.config import AppConfig
    from backend.listencontroller import ListenController

    return ListenController(AppConfig())


def test_audio_accumulates_and_a_boundary_clears_it():
    """The pause is the safe reset point — that is the whole reason the
    committed path waits for it, and it is why the preview can be thrown away
    without losing a word."""
    controller = _controller()
    controller._reset_heard()

    controller._remember_heard(speech(1))
    controller._remember_heard(speech(1))
    assert len(controller._heard_so_far()) == 2 * SAMPLE_RATE

    controller._reset_heard()
    assert len(controller._heard_so_far()) == 0


def test_a_speaker_who_never_pauses_does_not_grow_the_buffer_forever():
    controller = _controller()
    controller._reset_heard()

    for _ in range(60):
        controller._remember_heard(speech(1))

    held = len(controller._heard_so_far()) / SAMPLE_RATE
    assert held <= controller.HEARD_CEILING_SECONDS


def test_the_preview_can_be_switched_off():
    from backend.config import AppConfig
    from backend.listencontroller import ListenController

    config = AppConfig()
    config.listen_preview = False
    assert ListenController(config)._build_preview() is None


def test_a_provider_with_no_smaller_tier_simply_has_no_preview():
    """SenseVoice ships one model. That is a missing luxury, not an error."""
    from backend.providers.stt import draft_provider

    class OneModel:
        id = "onemodel"
        display_name = "single"

        def __init__(self, hardware=None):
            pass

        def is_available(self):
            return True, ""

    assert draft_provider("minimal", providers=[OneModel()]) is None


# --- the two things the real replay caught ---------------------------------------


@pytest.mark.parametrize("heard, shown", [
    # the last word is where the audio ran out, so it is where the model guesses
    ("just under a month", "just under a"),
    ("Fifty years ago, just under a mile from", "Fifty years ago, just under a mile"),
    # a finished sentence was not cut off; keep all of it
    ("The country it created wasn't perfect either.", 
     "The country it created wasn't perfect either."),
    ("Did you get chills?", "Did you get chills?"),
    # nothing left once the guess goes
    ("Reelionable", ""),
    # a language written without spaces is left alone
    ("这句话还没说完", "这句话还没说完"),
])
def test_the_guessed_last_word_is_left_to_the_ellipsis(heard, shown):
    assert trim_partial_tail(heard) == shown


def test_a_round_still_in_the_model_when_the_boundary_lands_does_not_repaint():
    """Caught by replaying a real talk: the utterance closed at 8.0s, and a
    preview round that had started at 7.9s came back at 8.1s and repainted the
    old text — which the committed sentences then printed again underneath. A
    guess must not outlive the thing it was guessing about."""
    shown = []
    texts = iter(["what was being said.", "a later guess entirely."])
    stream = PreviewStream(lambda audio, language=None: next(texts),
                           lambda: speech(3), shown.append)
    stream.once()
    assert shown == ["what was being said."]

    slow = stream._transcribe

    def interrupted(audio, language=None):
        text = slow(audio, language=language)
        stream.clear()               # the boundary lands while this is running
        return text

    stream._transcribe = interrupted
    stream.once()

    assert shown == ["what was being said.", ""], \
        f"过期的那一轮又画上去了：{shown}"


def test_a_ceiling_cut_keeps_a_little_audio_so_the_next_read_starts_on_a_word():
    """A cut imposed at the ceiling lands mid-word. Starting the next preview
    buffer exactly there made the small model read the fragment as a word of
    its own — 「unalienable rights」 came back as 「I'm not a highly in-n-able
    rights」. A natural pause needs no carry: the speaker stopped."""
    controller = _controller()

    controller._reset_heard()
    controller._remember_heard(speech(10))
    controller._reset_heard(carry=controller.CARRY_SECONDS)
    carried = len(controller._heard_so_far()) / SAMPLE_RATE
    assert abs(carried - controller.CARRY_SECONDS) < 0.01

    controller._remember_heard(speech(10))
    controller._reset_heard(carry=0.0)
    assert len(controller._heard_so_far()) == 0


def test_it_stops_guessing_while_the_committed_layer_finishes_that_stretch():
    """Right after a boundary the only audio available is the fragment after
    the cut, which reads as nonsense on its own — a real replay showed
    「Reliable R…」 sitting on screen for a second while the big model was
    already a heartbeat away from printing the sentence properly."""
    shown = []
    stream = PreviewStream(lambda audio, language=None: "a wild guess.",
                           lambda: speech(3), shown.append)
    stream.once()
    assert shown == ["a wild guess."]

    stream.clear()                       # boundary
    assert stream.once() == "", "边界之后不该继续猜"
    assert stream.once() == ""

    stream.resume()                      # committed sentences have landed
    assert stream.once() == "a wild guess."


def test_a_chunk_thrown_away_as_a_hallucination_still_releases_the_tail():
    """Silence transcribed as 「You」 is dropped before it reaches the record.
    If that path forgot to resume, the preview would stay frozen for the rest
    of the meeting."""
    import types

    from backend.config import AppConfig
    from backend.listencontroller import ListenController

    controller = ListenController(AppConfig())
    released = []
    controller._preview = types.SimpleNamespace(
        resume=lambda: released.append(1), clear=lambda: None)
    controller._sentences = None

    controller._on_utterance(types.SimpleNamespace(
        raw_text="Thanks for watching!", start_sec=0.0, forced=False,
        confidence=-0.8))

    assert released, "幻觉被丢掉之后预览就再也不动了"


# --- what the first real meeting with a preview turned up -----------------------


def test_listen_has_its_own_language_and_does_not_borrow_dictation_s():
    """They are opposite ends of the same person. The author dictates in
    Chinese, so dictate_language is 'zh'; the lecture they are listening to is
    in English. Listen mode read the dictation setting until 2026-08-20 and so
    told Whisper an English talk was Chinese — the big model shrugged it off,
    the small preview model put 「它有色彩的…」 under the English."""
    from backend.config import AppConfig

    config = AppConfig()
    assert config.listen_language == "en"
    assert config.dictate_language == "zh"
    assert config.listen_language != config.dictate_language


def test_the_controller_listens_in_the_listen_language(monkeypatch):
    from backend.config import AppConfig
    from backend.listencontroller import ListenController

    config = AppConfig()
    config.dictate_language = "zh"
    config.listen_language = "en"
    controller = ListenController(config)

    seen = {}
    monkeypatch.setattr(
        "backend.providers.stt.draft_provider",
        lambda tier, **kw: type("P", (), {"transcribe": staticmethod(lambda *a, **k: "")})())
    stream = controller._build_preview()
    if stream is not None:
        seen["language"] = stream._language
        assert seen["language"] == "en", "预览用了听写的语言"


def test_a_meeting_that_never_started_leaves_no_folder(tmp_path):
    """Four empty dated folders appeared in thirteen seconds while the
    aggregate audio device was being retried by hand."""
    import types

    from backend.config import AppConfig
    from backend.listencontroller import ListenController

    controller = ListenController(AppConfig())
    directory = tmp_path / "20260820-081935"
    directory.mkdir()
    controller.session = types.SimpleNamespace(directory=directory)

    controller._discard_empty_session()

    assert not directory.exists()
    assert controller.session is None


def test_a_meeting_that_wrote_something_keeps_its_folder(tmp_path):
    import types

    from backend.config import AppConfig
    from backend.listencontroller import ListenController

    controller = ListenController(AppConfig())
    directory = tmp_path / "20260820-081948"
    directory.mkdir()
    (directory / "entries.jsonl").write_text("{}")
    controller.session = types.SimpleNamespace(directory=directory)

    controller._discard_empty_session()

    assert directory.exists(), "写过东西的目录不能删"
