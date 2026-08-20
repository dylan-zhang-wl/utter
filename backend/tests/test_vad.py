"""P1 Task 9 — the VAD segmenter.

Two layers, tested separately on purpose:

  * VadSegmenter is a pure state machine over speech probabilities. All the
    interesting behaviour — when a pause ends an utterance, when a monologue is
    force-cut, how much audio comes back — is here, and none of it needs a
    neural network to test.
  * SileroVad turns 512-sample frames into probabilities via onnxruntime. One
    integration test, marked `model`, checks it against real audio.

There is deliberately no `partial` event. Design §3(b) measured v2's
re-transcribe-every-second strategy at 172-221% duty cycle and killed it; an
event that only a partial pipeline could consume would invite it back.
"""

import numpy as np
import pytest

from backend import vad

SR = 16000
FRAME = 512  # silero's window at 16 kHz — 32 ms


def frames(*spec):
    """Build audio from (kind, seconds) pairs. 'v' is voiced, 's' is silence."""
    out = []
    for kind, seconds in spec:
        n = int(SR * seconds)
        if kind == "v":
            t = np.arange(n) / SR
            out.append((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
        else:
            out.append(np.zeros(n, dtype=np.float32))
    return np.concatenate(out)


def loud_is_speech(frame: np.ndarray) -> float:
    """Stand-in for silero: anything with energy counts as speech."""
    return 1.0 if float(np.abs(frame).mean()) > 0.01 else 0.0


def segmenter(**kwargs):
    kwargs.setdefault("speech_prob", loud_is_speech)
    return vad.VadSegmenter(**kwargs)


def run(seg, audio, chunk_samples=1600):
    events = []
    for i in range(0, len(audio), chunk_samples):
        events.extend(seg.feed(audio[i : i + chunk_samples]))
    return events


def starts(events):
    return [e for e in events if isinstance(e, vad.SpeechStart)]


def ends(events):
    return [e for e in events if isinstance(e, vad.SpeechEnd)]


# --- the basics --------------------------------------------------------------


def test_pure_silence_emits_nothing():
    assert run(segmenter(), frames(("s", 3.0))) == []


def test_one_utterance_emits_start_then_end():
    audio = frames(("s", 0.3), ("v", 1.0), ("s", 1.2))
    events = run(segmenter(), audio)

    assert len(starts(events)) == 1
    assert len(ends(events)) == 1
    assert isinstance(events[0], vad.SpeechStart)
    assert isinstance(events[-1], vad.SpeechEnd)


def test_no_partial_events_exist():
    """Design §3(b). The type must not exist, so no consumer can grow to want it."""
    assert not hasattr(vad, "SpeechPartial")
    assert not hasattr(vad, "Partial")


def test_start_lands_within_100ms():
    audio = frames(("s", 0.5), ("v", 1.0), ("s", 1.2))
    event = starts(run(segmenter(), audio))[0]

    assert abs(event.at_sample - int(0.5 * SR)) < 0.1 * SR


def test_end_lands_within_100ms_of_the_pause_start():
    audio = frames(("s", 0.3), ("v", 1.0), ("s", 1.2))
    event = ends(run(segmenter(), audio))[0]

    assert abs(event.end_sample - int(1.3 * SR)) < 0.1 * SR


# --- the silence threshold ---------------------------------------------------


def test_short_pause_does_not_split():
    """300ms is a breath, not a boundary. Splitting here would cut clauses in
    half and hand Whisper fragments with no context."""
    audio = frames(("s", 0.2), ("v", 0.8), ("s", 0.3), ("v", 0.8), ("s", 1.2))
    events = run(segmenter(vad_silence_ms=600), audio)

    assert len(ends(events)) == 1


def test_long_pause_splits():
    audio = frames(("s", 0.2), ("v", 0.8), ("s", 0.9), ("v", 0.8), ("s", 1.2))
    events = run(segmenter(vad_silence_ms=600), audio)

    assert len(ends(events)) == 2
    assert len(starts(events)) == 2


def test_silence_threshold_is_configurable():
    audio = frames(("s", 0.2), ("v", 0.8), ("s", 0.5), ("v", 0.8), ("s", 1.2))
    assert len(ends(run(segmenter(vad_silence_ms=300), audio))) == 2
    assert len(ends(run(segmenter(vad_silence_ms=900), audio))) == 1


# --- the hard ceiling --------------------------------------------------------


def test_long_monologue_is_force_finalised():
    """Design §5's error handling. A speaker who never pauses — or a VAD false
    negative that never sees one — must not buffer forever."""
    audio = frames(("s", 0.2), ("v", 12.0), ("s", 1.2))
    events = run(segmenter(max_utterance_sec=5), audio)

    assert len(ends(events)) >= 2
    assert events[1].forced is True


def test_forced_cut_is_marked():
    audio = frames(("s", 0.2), ("v", 12.0), ("s", 1.2))
    forced = [e for e in ends(run(segmenter(max_utterance_sec=5), audio)) if e.forced]

    assert forced, "a forced cut must be distinguishable — it may split a word"


def test_natural_end_is_not_marked_forced():
    audio = frames(("s", 0.2), ("v", 1.0), ("s", 1.2))
    assert ends(run(segmenter(max_utterance_sec=30), audio))[0].forced is False


def test_forced_cut_reopens_immediately():
    """The speaker is still talking; the next utterance starts at the cut."""
    audio = frames(("s", 0.2), ("v", 12.0), ("s", 1.2))
    events = run(segmenter(max_utterance_sec=5), audio)

    assert len(starts(events)) == len(ends(events))


# --- the audio that comes back -----------------------------------------------


def test_end_carries_the_audio():
    audio = frames(("s", 0.3), ("v", 1.0), ("s", 1.2))
    event = ends(run(segmenter(), audio))[0]

    assert isinstance(event.audio, np.ndarray)
    assert event.audio.dtype == np.float32
    assert 0.9 * SR < len(event.audio) < 1.8 * SR


def test_audio_includes_pre_roll():
    """VAD notices speech a frame or two after it starts, so a segment cut at
    the detection point clips the first consonant. Keep a little of what came
    before."""
    audio = frames(("s", 0.5), ("v", 1.0), ("s", 1.2))
    seg = segmenter(pre_roll_ms=200)
    event = ends(run(seg, audio))[0]

    assert event.start_sample < int(0.5 * SR), "must reach back before detection"
    assert len(event.audio) > 1.0 * SR


def test_pre_roll_cannot_reach_before_the_stream():
    """Speech starting at sample 0 must not produce a negative index."""
    audio = frames(("v", 1.0), ("s", 1.2))
    event = ends(run(segmenter(pre_roll_ms=200), audio))[0]

    assert event.start_sample >= 0
    assert len(event.audio) > 0


def test_audio_is_actually_the_voiced_part():
    audio = frames(("s", 0.3), ("v", 1.0), ("s", 1.2))
    event = ends(run(segmenter(), audio))[0]

    assert float(np.abs(event.audio).mean()) > 0.01, "should not be mostly silence"


# --- streaming invariants ----------------------------------------------------


@pytest.mark.parametrize("chunk", [512, 1600, 4096, 16000])
def test_result_is_independent_of_chunk_size(chunk):
    """The audio source delivers whatever size it likes; segmentation must not
    depend on it."""
    audio = frames(("s", 0.2), ("v", 0.8), ("s", 0.9), ("v", 0.8), ("s", 1.2))
    events = run(segmenter(), audio, chunk_samples=chunk)

    assert len(ends(events)) == 2


def test_a_partial_frame_is_buffered_not_dropped():
    seg = segmenter()
    seg.feed(np.zeros(100, dtype=np.float32))  # less than one 512-sample frame
    assert seg.feed(np.zeros(100, dtype=np.float32)) == []


def test_flush_closes_an_open_utterance():
    """Hotkey release, or end of file. Whatever is buffered must come out."""
    seg = segmenter()
    run(seg, frames(("s", 0.2), ("v", 1.0)))

    events = seg.flush()
    assert len(ends(events)) == 1
    assert ends(events)[0].forced is True


def test_flush_on_silence_emits_nothing():
    seg = segmenter()
    run(seg, frames(("s", 1.0)))
    assert seg.flush() == []


def test_flush_is_idempotent():
    seg = segmenter()
    run(seg, frames(("s", 0.2), ("v", 1.0)))
    seg.flush()
    assert seg.flush() == []


def test_reset_clears_state():
    seg = segmenter()
    run(seg, frames(("s", 0.2), ("v", 1.0)))
    seg.reset()
    assert seg.flush() == []


def test_is_synchronous():
    """No asyncio. Design keeps this pure so it stays testable and so the
    pipeline can decide its own threading."""
    import inspect

    assert not inspect.iscoroutinefunction(vad.VadSegmenter.feed)
    assert not inspect.isasyncgenfunction(vad.VadSegmenter.feed)


def test_sample_positions_are_monotonic():
    audio = frames(("s", 0.2), ("v", 0.8), ("s", 0.9), ("v", 0.8), ("s", 1.2))
    events = run(segmenter(), audio)
    positions = [
        e.at_sample if isinstance(e, vad.SpeechStart) else e.end_sample for e in events
    ]

    assert positions == sorted(positions)


# --- the ONNX layer ----------------------------------------------------------


def test_no_torch_import_in_the_module():
    """铁律 5. silero-vad's pip package drags 328M of torch for a 1.2M model."""
    import ast

    tree = ast.parse(open(vad.__file__).read())
    names = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(n and n.startswith("torch") for n in names)
    assert not any(n and n.startswith("silero") for n in names)


@pytest.mark.model
def test_silero_scores_speech_above_silence():
    """The one test that runs the real network."""
    model = vad.SileroVad()
    voiced = frames(("v", 0.5))
    silent = frames(("s", 0.5))

    def mean_prob(audio):
        model.reset()
        n = len(audio) // FRAME * FRAME
        probs = [model.speech_prob(audio[i : i + FRAME]) for i in range(0, n, FRAME)]
        return sum(probs) / len(probs)

    assert mean_prob(silent) < 0.3
    assert mean_prob(voiced) > mean_prob(silent)


@pytest.mark.model
def test_segments_real_speech():
    """End to end on the generated fixture: real audio, real model."""
    import os

    import soundfile as sf

    path = f"{os.environ.get('TMPDIR', '/tmp')}/utter-fixtures/bench-say-en.wav"
    if not os.path.exists(path):
        pytest.skip("run docs/benchmarks/make-fixture.sh first")

    audio, sr = sf.read(path, dtype="float32")
    assert sr == SR

    seg = vad.VadSegmenter(vad_silence_ms=500)
    events = run(seg, audio) + seg.flush()
    utterances = ends(events)

    # The fixture is ten sentences separated by explicit 700ms silences.
    assert 6 <= len(utterances) <= 14, f"got {len(utterances)}"
    assert all(len(u.audio) > 0.2 * SR for u in utterances)


# --- the silence gate --------------------------------------------------------


def test_has_speech_is_false_for_silence():
    """Push-to-talk has no VAD in its segmentation path, so this is the only
    thing standing between an empty room and a hallucinated sentence in the
    author's document."""
    assert vad.has_speech(frames(("s", 2.0)), speech_prob=loud_is_speech) is False


def test_has_speech_is_true_for_speech():
    assert vad.has_speech(frames(("v", 1.0)), speech_prob=loud_is_speech) is True


def test_has_speech_ignores_a_single_stray_frame():
    """A door closing is one loud frame, not an utterance."""
    audio = frames(("s", 0.5))
    audio[8000:8200] = 0.5
    assert vad.has_speech(audio, speech_prob=loud_is_speech) is False


def test_has_speech_is_false_for_a_buffer_shorter_than_a_frame():
    assert vad.has_speech(np.zeros(100, dtype=np.float32), speech_prob=loud_is_speech) is False


def test_has_speech_is_false_for_nothing():
    assert vad.has_speech(None, speech_prob=loud_is_speech) is False


@pytest.mark.model
def test_has_speech_rejects_real_silence():
    assert vad.has_speech(frames(("s", 2.0))) is False


@pytest.mark.model
def test_has_speech_accepts_real_speech():
    import os
    import soundfile as sf

    path = f"{os.environ.get('TMPDIR', '/tmp')}/utter-fixtures/bench-say-en.wav"
    if not os.path.exists(path):
        pytest.skip("run docs/benchmarks/make-fixture.sh first")

    audio, _ = sf.read(path, dtype="float32")
    assert vad.has_speech(audio[: SR * 3]) is True


def test_speech_duration_counts_only_the_talking():
    """A recording is 20 seconds; how much of it is anyone speaking? Without
    this the author cannot tell a model that dropped their words from a key
    held down over silence."""
    audio = frames(("v", 2.0), ("s", 6.0), ("v", 2.0))
    assert vad.speech_duration(audio, speech_prob=loud_is_speech) == pytest.approx(4.0, abs=0.05)


def test_speech_duration_is_zero_for_silence():
    assert vad.speech_duration(frames(("s", 3.0)), speech_prob=loud_is_speech) == 0.0


def test_speech_duration_is_zero_for_a_buffer_shorter_than_a_frame():
    assert vad.speech_duration(np.zeros(100, dtype=np.float32), speech_prob=loud_is_speech) == 0.0


# --- the floor that keeps a sentence together (P3, 2026-08-16) -------------------


def _speech(seconds):
    """Audio the VAD will call speech."""
    import numpy as np

    from backend.vad import SAMPLE_RATE

    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (np.sin(2 * np.pi * 200 * t) * 0.4).astype(np.float32)


def _silence(seconds):
    import numpy as np

    from backend.vad import SAMPLE_RATE

    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def _run(segmenter, audio):
    import numpy as np

    from backend.vad import SpeechEnd

    out = []
    for i in range(0, len(audio), 8000):
        out += [e for e in segmenter.feed(audio[i:i + 8000]) if isinstance(e, SpeechEnd)]
    out += [e for e in segmenter.flush() if isinstance(e, SpeechEnd)]
    return out


def _talk():
    """One sentence of three phrases, then a real sentence break, then more.

    The phrase pauses are 600ms — just over the 500ms threshold, which is the
    whole problem — and the sentence pause is 1.2s. This is the shape that
    produced 「My father」 and 「from equity states」 as separate entries in a
    real session.
    """
    import numpy as np

    return np.concatenate([
        _speech(1.5), _silence(0.6), _speech(1.5), _silence(0.6), _speech(1.5),
        _silence(1.2),
        _speech(1.5), _silence(0.6), _speech(1.5),
        _silence(1.2),
    ])


def test_without_a_floor_every_breath_cuts_a_fragment():
    """The bug, reproduced: 500ms alone treats a phrase break as a sentence
    end, which is how a transcript came back as 「My father」."""
    from backend.vad import VadSegmenter

    pieces = _run(VadSegmenter(vad_silence_ms=500, max_utterance_sec=12,
                               speech_prob=lambda f: 1.0 if abs(f).max() > 0.1 else 0.0),
                  _talk())
    assert len(pieces) >= 4, f"应该被切碎，实际 {len(pieces)} 段"


def test_a_floor_keeps_the_phrases_of_one_sentence_together():
    from backend.vad import SAMPLE_RATE, VadSegmenter

    pieces = _run(VadSegmenter(vad_silence_ms=500, max_utterance_sec=12,
                               min_utterance_sec=4.0,
                               speech_prob=lambda f: 1.0 if abs(f).max() > 0.1 else 0.0),
                  _talk())

    assert len(pieces) <= 2, f"三个词组应该合成一句，实际切了 {len(pieces)} 段"
    assert all(len(p.audio) / SAMPLE_RATE >= 3.0 for p in pieces), \
        "还有短于 3 秒的碎片"


def test_the_ceiling_still_wins_over_the_floor():
    """A speaker who never pauses must not be held for ever."""
    from backend.vad import SAMPLE_RATE, VadSegmenter

    pieces = _run(VadSegmenter(vad_silence_ms=500, max_utterance_sec=6,
                               min_utterance_sec=4.0,
                               speech_prob=lambda f: 1.0 if abs(f).max() > 0.1 else 0.0),
                  _speech(20.0))

    assert pieces, "一直说话也必须出字"
    assert max(len(p.audio) / SAMPLE_RATE for p in pieces) <= 7.0


def test_dictation_is_unaffected_because_the_floor_defaults_to_off():
    from backend.vad import VadSegmenter

    assert VadSegmenter().min_utterance_sec == 0.0


def test_a_ceiling_cut_does_not_hand_the_same_audio_out_twice():
    """Pre-roll protects the first syllable of a fresh utterance. Applied to a
    segment that opened by reopening at the ceiling it reaches back into audio
    the previous chunk already transcribed, and the words arrive twice —
    「…be more of a revelation to me.」 then 「to me than it was to you.」"""
    import numpy as np

    from backend.vad import FRAME_SAMPLES, SAMPLE_RATE, SpeechEnd, VadSegmenter

    segmenter = VadSegmenter(max_utterance_sec=1, vad_silence_ms=10_000,
                             speech_prob=lambda frame: 1.0)
    ends = []
    for _ in range(int(3 * SAMPLE_RATE) // FRAME_SAMPLES):
        ends += [e for e in segmenter.feed(np.ones(FRAME_SAMPLES, dtype="float32"))
                 if isinstance(e, SpeechEnd)]

    assert len(ends) >= 2, "应该撞到天花板不止一次"
    for earlier, later in zip(ends, ends[1:]):
        assert later.start_sample >= earlier.end_sample, (
            f"第二段从 {later.start_sample} 开始，而第一段到 {earlier.end_sample} "
            f"才结束，中间 {earlier.end_sample - later.start_sample} 个样本被转录了两次")


def test_a_fresh_utterance_still_gets_its_pre_roll():
    """The overlap only comes off where the audio was already sent; a speaker
    starting to talk still needs the moment before the detector noticed."""
    import numpy as np

    from backend.vad import FRAME_SAMPLES, SpeechEnd, VadSegmenter

    voiced = iter([False] * 20 + [True] * 60 + [False] * 60)
    segmenter = VadSegmenter(vad_silence_ms=200, min_utterance_sec=0.0,
                             speech_prob=lambda frame: 1.0 if next(voiced) else 0.0)
    ends = []
    for _ in range(140):
        ends += [e for e in segmenter.feed(np.zeros(FRAME_SAMPLES, dtype="float32"))
                 if isinstance(e, SpeechEnd)]

    assert ends, "没切出任何一段"
    speech_began = 20 * FRAME_SAMPLES
    assert ends[0].start_sample < speech_began, "第一段没有留出前摇"
