

# --- optional detailed transcription (P3 task 2b) -------------------------------


def test_transcribe_detailed_falls_back_for_providers_without_it():
    """Optional by design: no existing provider had to change, and dictation
    goes on calling transcribe() exactly as before."""
    from backend.providers.stt import transcribe_detailed

    class Plain:
        def transcribe(self, audio, language=None, initial_prompt=None):
            return "hello"

    got = transcribe_detailed(Plain(), None)
    assert got.text == "hello"
    assert got.confidence is None and got.no_speech is None


def test_transcribe_detailed_uses_the_richer_call_when_there_is_one():
    from backend.providers.stt import Detailed, transcribe_detailed

    class Rich:
        def transcribe(self, audio, language=None, initial_prompt=None):
            raise AssertionError("应该走详细接口")

        def transcribe_detailed(self, audio, language=None, initial_prompt=None):
            return Detailed(text="hello", confidence=-0.31, no_speech=0.01)

    got = transcribe_detailed(Rich(), None)
    assert got.confidence == -0.31


def test_a_broken_detailed_call_still_returns_the_text():
    """铁律 8 again: confidence is a nicety, the words are not."""
    from backend.providers.stt import transcribe_detailed

    class Half:
        def transcribe(self, audio, language=None, initial_prompt=None):
            return "hello"

        def transcribe_detailed(self, audio, language=None, initial_prompt=None):
            raise RuntimeError("boom")

    assert transcribe_detailed(Half(), None).text == "hello"


def test_confidence_is_weighted_by_how_long_each_segment_ran():
    """A two-second aside and a twenty-second sentence are not equally
    informative about how well the utterance as a whole was heard."""
    from backend.providers.mlx import _weighted

    segments = [
        {"start": 0, "end": 20, "avg_logprob": -0.2},
        {"start": 20, "end": 22, "avg_logprob": -1.2},
    ]
    plain = (-0.2 + -1.2) / 2
    weighted = _weighted(segments, "avg_logprob")

    assert weighted > plain, "长句应该占更大权重"
    assert -0.35 < weighted < -0.25


def test_weighted_survives_segments_with_nothing_in_them():
    from backend.providers.mlx import _weighted

    assert _weighted([], "avg_logprob") is None
    assert _weighted([{"start": 0, "end": 1}], "avg_logprob") is None
