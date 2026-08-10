"""P2b — 边说边出字, on the toggle gesture.

The shape is forced by 铁律 9: injected text is never revised, so there are no
partial hypotheses being corrected on screen. Each clause is transcribed once
its pause has arrived and lands final. What the author gets is their sentences
appearing one behind the other, a beat late — not a live caption.

The microphone here is a fake that plays a scripted stream, so these run in
milliseconds and test the wiring rather than silero.
"""

import threading
import time

import numpy as np
import pytest

from backend.config import AppConfig
from backend.tests.test_daemon import FakeHotkey, FakeInjector, FakeStt
from backend import daemon as daemon_mod
from backend.vad import SpeechEnd


class StreamingMic:
    """Hands out chunks over time, the way a real microphone does."""

    def __init__(self, chunks=None):
        # Enough chunks that the scripted segmenter gets asked often enough to
        # emit every clause the test set up. One chunk meant one feed() call
        # and therefore one clause, whatever the test asked for.
        self._pending = list(
            chunks if chunks is not None
            else [np.zeros(1600, dtype=np.float32) for _ in range(20)]
        )
        self.started = 0
        self.stopped = 0
        self.dropped = 0
        self.overflows = 0
        self.first_chunk_at = None

    def start(self):
        self.started += 1
        return self

    def stop(self):
        self.stopped += 1

    def chunks(self):
        while self._pending:
            yield self._pending.pop(0)


def build(clauses, config=None, **kwargs):
    """A daemon whose segmenter emits `clauses` and nothing else."""
    emitted = iter(clauses)

    class ScriptedSegmenter:
        def __init__(self, **_kw):
            pass

        def feed(self, _chunk):
            try:
                return [SpeechEnd(start_sample=0, end_sample=16000, audio=next(emitted))]
            except StopIteration:
                return []

        def flush(self):
            return []

    daemon_mod.VadSegmenter = ScriptedSegmenter  # replaced per-test, restored by fixture
    return daemon_mod.DictationDaemon(
        config=config or AppConfig(),
        stt=kwargs.pop("stt", None) or FakeStt(texts=["一", "二", "三"]),
        make_mic=lambda mic=kwargs.pop("mic", None): mic or StreamingMic(),
        injector=kwargs.pop("injector", None) or FakeInjector(),
        hotkey_factory=lambda on_event: FakeHotkey(on_event),
        speech_check=kwargs.pop("speech_check", None) or (lambda audio: True),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def restore_segmenter():
    real = daemon_mod.VadSegmenter
    yield
    daemon_mod.VadSegmenter = real


def clause(seconds=1.0):
    return np.full(int(16000 * seconds), 0.2, dtype=np.float32)


def test_a_clause_is_injected_before_the_session_ends():
    """The whole point. Under the old behaviour nothing arrived until the
    author stopped talking."""
    injector = FakeInjector()
    d = build([clause(), clause()], injector=injector)
    d.start()
    d.begin_utterance(stream=True)

    deadline = time.time() + 5
    while len(injector.injected) < 2 and time.time() < deadline:
        time.sleep(0.05)

    assert len(injector.injected) >= 2, "clauses should arrive while the mic is open"
    d.end_utterance()
    d.stop()


def test_clauses_arrive_in_the_order_they_were_spoken():
    """铁律 11. One worker, one queue — but worth a test, because a pool would
    be the obvious 'optimisation' for someone who did not know why."""
    injector = FakeInjector()
    d = build([clause(), clause(), clause()],
              stt=FakeStt(texts=["第一句", "第二句", "第三句"]), injector=injector)
    d.start()
    d.begin_utterance(stream=True)

    deadline = time.time() + 5
    while len(injector.injected) < 3 and time.time() < deadline:
        time.sleep(0.05)
    d.end_utterance()
    d.wait_idle()
    d.stop()

    indices = [i for i, _ in injector.injected]
    assert indices == sorted(indices), "delivered out of order"


def test_push_to_talk_does_not_stream():
    """The finger is already the segmenter there, and a phrase does not want
    to be cut into clauses."""
    d = build([clause()])
    d.start()
    d.begin_utterance(stream=False)
    assert d._streaming is False
    d.end_utterance()
    d.stop()


def test_streaming_can_be_turned_off_entirely():
    d = build([clause()], config=AppConfig(stream_while_speaking=False))
    d.start()
    d.begin_utterance(stream=True)
    assert d._streaming is False, "the config has the final say"
    d.end_utterance()
    d.stop()


def test_ending_a_streaming_session_does_not_double_drain():
    """The reader owns the microphone queue while streaming. Draining it again
    in end_utterance would race the reader and lose whichever chunks it won."""
    mic = StreamingMic([clause(0.1) for _ in range(5)])
    d = build([clause()], mic=mic)
    d.start()
    d.begin_utterance(stream=True)
    time.sleep(0.2)
    before = d._index
    d.end_utterance()
    d.wait_idle()
    d.stop()

    # end_utterance must not have enqueued a job of its own on top of the
    # reader's; the only growth allowed is the reader's own flush.
    assert d._index - before <= 1


def test_the_stream_reader_stops_with_the_session():
    d = build([clause()])
    d.start()
    before = threading.active_count()
    d.begin_utterance(stream=True)
    d.end_utterance()
    d.wait_idle()
    time.sleep(0.4)
    d.stop()

    assert threading.active_count() <= before + 1, "reader thread leaked"


def test_a_silent_clause_is_still_gated():
    """The silence gate applies to every clause, not just to whole holds —
    silero can close a segment on a cough, and Whisper answers a cough with a
    sentence."""
    injector = FakeInjector()
    d = build([clause()], injector=injector, speech_check=lambda audio: False)
    d.start()
    d.begin_utterance(stream=True)
    time.sleep(0.3)
    d.end_utterance()
    d.wait_idle()
    d.stop()

    assert injector.injected == []


def test_the_menu_toggle_for_streaming_survives_a_restart(tmp_path, monkeypatch):
    """_save_config only writes the fields it owns, so a new menu item that is
    not on that list flips in memory and reverts on the next launch — the
    third time this project would have shipped v1's Save button."""
    from backend import config as cfg
    from backend.daemon import DictationDaemon

    monkeypatch.setattr("backend.config.DEFAULT_DIR", tmp_path)
    cfg.save(AppConfig())

    assert "stream_while_speaking" in DictationDaemon.MENU_OWNED

    d = build([clause()])
    d.config.stream_while_speaking = False
    d._save_config()

    assert cfg.load().stream_while_speaking is False
