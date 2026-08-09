"""P1 Task 11 — the five-slot pipeline.

    AudioSource -> Segmenter -> SttProvider -> PostProcessor -> Sink

Design §4: both modes have the same shape and differ only in what fills each
slot. That is the claim that makes Utter one platform rather than two programs
sharing a repo, so these tests check the shape holds — including that the
pipeline module never learns the name of a concrete backend.

Everything is faked. Real providers have their own tests.
"""

import ast

import numpy as np
import pytest

from backend import pipeline as pipe
from backend.vad import SpeechEnd


class FakeStt:
    def __init__(self, texts=None, fail_on=()):
        self.texts = list(texts or ["one", "two", "three"])
        self.fail_on = set(fail_on)
        self.calls = []

    def transcribe(self, audio, language=None, initial_prompt=None):
        index = len(self.calls)
        self.calls.append({"n": len(audio), "language": language, "prompt": initial_prompt})
        if index in self.fail_on:
            raise RuntimeError("model exploded")
        return self.texts[index % len(self.texts)]


class RecordingSink:
    def __init__(self):
        self.received = []

    def __call__(self, utterance):
        self.received.append(utterance)


def utterance(seconds=1.0, start=0, forced=False):
    n = int(16000 * seconds)
    return SpeechEnd(
        start_sample=start,
        end_sample=start + n,
        audio=np.full(n, 0.1, dtype=np.float32),
        forced=forced,
    )


def run(p, count=2):
    for i in range(count):
        p.handle(utterance(start=i * 16000))
    return p


# --- shape -------------------------------------------------------------------


def test_pipeline_does_not_import_any_concrete_backend():
    """铁律 6, at the layer most likely to break it. The pipeline is where a
    tired afternoon reaches for `import mlx_whisper` to 'just get it working'."""
    tree = ast.parse(open(pipe.__file__).read())
    names = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    for banned in ("mlx_whisper", "faster_whisper", "ctranslate2", "openai", "httpx"):
        assert banned not in names


def test_listen_and_dictate_share_a_class():
    """Design §4: same pipeline, different slots. Two classes would mean two
    programs."""
    listen = pipe.listen_pipeline(stt=FakeStt(), sink=RecordingSink())
    dictate = pipe.dictate_pipeline(stt=FakeStt(), sink=RecordingSink())
    assert type(listen) is type(dictate)


# --- listen mode -------------------------------------------------------------


def test_listen_translates_each_utterance():
    calls = []
    sink = RecordingSink()
    p = pipe.listen_pipeline(
        stt=FakeStt(), sink=sink, translate=lambda text, context=None: calls.append(text) or f"[中] {text}"
    )
    run(p, 2)

    assert calls == ["one", "two"]
    assert sink.received[0].translation == "[中] one"


def test_listen_does_not_polish():
    polished = []
    p = pipe.listen_pipeline(
        stt=FakeStt(),
        sink=RecordingSink(),
        translate=lambda text, context=None: text,
        polish=lambda text, context=None: polished.append(text) or text,
    )
    run(p, 2)

    assert polished == []


def test_silence_is_never_transcribed():
    """A zero-length segment costs a full second of model time for nothing."""
    stt = FakeStt()
    p = pipe.listen_pipeline(stt=stt, sink=RecordingSink())
    p.handle(SpeechEnd(0, 0, np.zeros(0, dtype=np.float32)))

    assert stt.calls == []


def test_empty_transcript_is_not_emitted():
    """Whisper returns "" or " " for a segment that turned out to be noise."""
    sink = RecordingSink()
    p = pipe.listen_pipeline(stt=FakeStt(texts=["   "]), sink=sink)
    run(p, 1)

    assert sink.received == []


# --- dictate mode ------------------------------------------------------------


def test_dictate_polishes_and_never_translates():
    polished, translated = [], []
    sink = RecordingSink()
    p = pipe.dictate_pipeline(
        stt=FakeStt(),
        sink=sink,
        polish=lambda text, context=None: polished.append(text) or f"{text}.",
        translate=lambda text, context=None: translated.append(text) or text,
    )
    run(p, 2)

    assert polished == ["one", "two"]
    assert translated == []
    assert sink.received[0].text == "one."


def test_dictate_without_a_polisher_emits_the_raw_transcript():
    """P2a ships with polish off (design §8). That is a supported mode, not a
    degraded one."""
    sink = RecordingSink()
    p = pipe.dictate_pipeline(stt=FakeStt(), sink=sink, polish=None)
    run(p, 1)

    assert sink.received[0].text == "one"


# --- 铁律 8 and 9 ------------------------------------------------------------


def test_stt_failure_skips_one_utterance_without_killing_the_pipeline():
    sink = RecordingSink()
    p = pipe.listen_pipeline(stt=FakeStt(fail_on=[0]), sink=sink)
    run(p, 3)

    assert len(sink.received) == 2, "the failed one is dropped, the rest continue"


def test_a_failing_sink_does_not_kill_the_pipeline():
    """A blocked injection target must not stop transcription — the text still
    has to reach the archive."""
    def angry_sink(_utterance):
        raise RuntimeError("target window vanished")

    p = pipe.listen_pipeline(stt=FakeStt(), sink=angry_sink)
    run(p, 2)  # must not raise


def test_raw_transcript_survives_a_polish_failure():
    """铁律 8. The whole point: never lose text to a language model."""
    def broken_polish(text, context=None):
        raise RuntimeError("ollama is down")

    sink = RecordingSink()
    p = pipe.dictate_pipeline(stt=FakeStt(), sink=sink, polish=broken_polish)
    run(p, 1)

    assert sink.received[0].text == "one"
    assert sink.received[0].polished is False


def test_raw_is_always_kept_alongside_the_processed_text():
    """铁律 10's audit trail — the only way the author can later notice that a
    polish changed their meaning."""
    sink = RecordingSink()
    p = pipe.dictate_pipeline(
        stt=FakeStt(), sink=sink, polish=lambda text, context=None: "COMPLETELY DIFFERENT"
    )
    run(p, 1)

    assert sink.received[0].text == "COMPLETELY DIFFERENT"
    assert sink.received[0].raw_text == "one"


def test_translation_failure_keeps_the_english():
    def broken_translate(text, context=None):
        raise RuntimeError("rate limited")

    sink = RecordingSink()
    p = pipe.listen_pipeline(stt=FakeStt(), sink=sink, translate=broken_translate)
    run(p, 1)

    assert sink.received[0].text == "one"
    assert sink.received[0].translation is None


# --- ordering and metadata ---------------------------------------------------


def test_utterances_are_emitted_in_order():
    sink = RecordingSink()
    p = pipe.listen_pipeline(stt=FakeStt(texts=["a", "b", "c"]), sink=sink)
    run(p, 3)

    assert [u.text for u in sink.received] == ["a", "b", "c"]


def test_indices_are_monotonic():
    sink = RecordingSink()
    p = pipe.listen_pipeline(stt=FakeStt(), sink=sink)
    run(p, 3)

    assert [u.index for u in sink.received] == [0, 1, 2]


def test_timestamps_are_monotonic():
    sink = RecordingSink()
    p = pipe.listen_pipeline(stt=FakeStt(), sink=sink)
    run(p, 3)

    times = [u.start_sec for u in sink.received]
    assert times == sorted(times)


def test_skipped_utterances_do_not_consume_an_index():
    sink = RecordingSink()
    p = pipe.listen_pipeline(stt=FakeStt(fail_on=[1]), sink=sink)
    run(p, 3)

    assert [u.index for u in sink.received] == [0, 1]


def test_forced_flag_reaches_the_sink():
    sink = RecordingSink()
    p = pipe.listen_pipeline(stt=FakeStt(), sink=sink)
    p.handle(utterance(forced=True))

    assert sink.received[0].forced is True


# --- context and vocabulary --------------------------------------------------


def test_previous_text_is_offered_as_context():
    seen = []
    p = pipe.dictate_pipeline(
        stt=FakeStt(texts=["first", "second"]),
        sink=RecordingSink(),
        polish=lambda text, context=None: seen.append(context) or text,
    )
    run(p, 2)

    assert seen[0] is None
    assert seen[1] == "first"


def test_vocabulary_is_passed_to_the_model_as_a_prompt():
    """§4.1g layer 1 — free, and it works with polish switched off."""
    stt = FakeStt()
    p = pipe.dictate_pipeline(stt=stt, sink=RecordingSink(), vocabulary=["Venuti", "异化"])
    run(p, 1)

    assert "Venuti" in stt.calls[0]["prompt"]


def test_no_prompt_when_the_vocabulary_is_empty():
    stt = FakeStt()
    p = pipe.dictate_pipeline(stt=stt, sink=RecordingSink(), vocabulary=[])
    run(p, 1)

    assert stt.calls[0]["prompt"] is None


def test_language_is_forwarded():
    stt = FakeStt()
    p = pipe.listen_pipeline(stt=stt, sink=RecordingSink(), language="en")
    run(p, 1)

    assert stt.calls[0]["language"] == "en"


# --- driving from a segmenter ------------------------------------------------


def test_run_stream_drives_the_whole_chain():
    from backend.vad import VadSegmenter

    sr = 16000
    t = np.arange(sr) / sr
    voiced = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    silence = np.zeros(int(sr * 1.0), dtype=np.float32)
    audio = np.concatenate([silence[: sr // 5], voiced, silence, voiced, silence])

    sink = RecordingSink()
    p = pipe.listen_pipeline(stt=FakeStt(texts=["a", "b"]), sink=sink)
    segmenter = VadSegmenter(
        speech_prob=lambda f: 1.0 if float(np.abs(f).mean()) > 0.01 else 0.0
    )

    p.run_stream(chunks(audio, 1600), segmenter)
    assert [u.text for u in sink.received] == ["a", "b"]


def chunks(audio, size):
    for i in range(0, len(audio), size):
        yield audio[i : i + size]


def test_run_stream_flushes_at_the_end():
    """A stream that stops mid-utterance — end of file, hotkey release — must
    still deliver what was buffered."""
    from backend.vad import VadSegmenter

    sr = 16000
    t = np.arange(sr) / sr
    voiced = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    sink = RecordingSink()
    p = pipe.listen_pipeline(stt=FakeStt(texts=["a"]), sink=sink)
    segmenter = VadSegmenter(
        speech_prob=lambda f: 1.0 if float(np.abs(f).mean()) > 0.01 else 0.0
    )

    p.run_stream(chunks(voiced, 1600), segmenter)
    assert len(sink.received) == 1


def test_mode_is_recorded_on_the_utterance():
    sink = RecordingSink()
    pipe.listen_pipeline(stt=FakeStt(), sink=sink).handle(utterance())
    assert sink.received[0].mode == "listen"

    sink2 = RecordingSink()
    pipe.dictate_pipeline(stt=FakeStt(), sink=sink2).handle(utterance())
    assert sink2.received[0].mode == "dictate"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        pipe.Pipeline(stt=FakeStt(), sink=RecordingSink(), mode="karaoke")
