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
