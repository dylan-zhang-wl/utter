"""P2a Task 6 — the resident dictation daemon.

Everything at the edges is faked: no microphone, no hotkey listener, no model,
no injection. What is under test is the wiring and the failure behaviour.

Resident rather than launched per use, because P1 measured 2.76s for the first
utterance against 1.20s for every one after — model load plus Metal kernel
compile. A per-use launch would charge that to every first sentence.
"""

import numpy as np
import pytest

from backend import daemon as daemon_mod
from backend.config import AppConfig
from backend.injection import InjectionResult, Target


class FakeMic:
    def __init__(self, seconds=1.0):
        self.audio = np.full(int(16000 * seconds), 0.2, dtype=np.float32)
        self.started = 0
        self.stopped = 0
        self.dropped = 0

    def start(self):
        self.started += 1
        return self

    def stop(self):
        self.stopped += 1

    def chunks(self):
        yield self.audio


class FakeStt:
    def __init__(self, texts=None, fail=False):
        self.texts = list(texts or ["Translation is rewriting."])
        self.calls = []
        self.fail = fail

    def transcribe(self, audio, language=None, initial_prompt=None):
        self.calls.append({"n": len(audio), "prompt": initial_prompt, "language": language})
        if self.fail:
            raise RuntimeError("model exploded")
        return self.texts[(len(self.calls) - 1) % len(self.texts)]


class FakeInjector:
    """Mirrors the real Injector's attribute names.

    It did not, until now: the fake exposed `locked` while the real one exposes
    `target`, and daemon code reading `.target` therefore saw None in tests and
    a real value in production. Fakes that drift from the thing they stand in
    for hide exactly the bug they were written to catch.
    """

    def __init__(self):
        self.injected = []
        self.target = None
        self.pending = 0

    @property
    def locked(self):
        return self.target

    def lock_target(self, target=None):
        self.target = target or Target(pid=1, name="TestApp")
        return self.target

    def release(self):
        self.target = None

    def inject(self, index, text):
        self.injected.append((index, text))
        return InjectionResult(injected=True)

    def flush(self):
        return InjectionResult(injected=False, reason="nothing pending")


def build(config=None, stt=None, mic=None, injector=None, **kwargs):
    # The fake microphone emits a constant DC level, which the real silero VAD
    # correctly rejects as not-speech. These tests are about wiring, so the gate
    # is stubbed open unless a test is specifically about it.
    kwargs.setdefault("speech_check", lambda audio: True)
    return daemon_mod.DictationDaemon(
        config=config or AppConfig(),
        stt=stt or FakeStt(),
        make_mic=lambda: mic or FakeMic(),
        injector=injector or FakeInjector(),
        hotkey_factory=lambda on_event: FakeHotkey(on_event),
        **kwargs,
    )


class FakeHotkey:
    def __init__(self, on_event):
        self.on_event = on_event
        self.started = False
        self.closed = False

    def is_available(self):
        return True, ""

    def start(self):
        self.started = True
        return self

    def close(self):
        self.closed = True


# --- the happy path ----------------------------------------------------------


def test_one_dictation_runs_the_whole_chain():
    stt, injector = FakeStt(), FakeInjector()
    d = build(stt=stt, injector=injector)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert stt.calls, "the model was asked"
    assert injector.injected == [(0, "Translation is rewriting.")]
    d.stop()


def test_target_is_locked_when_dictation_begins():
    """Not when it ends — by then the user may have switched windows, and the
    text belongs where they started talking (design §4.1e)."""
    injector = FakeInjector()
    d = build(injector=injector)
    d.start()
    d.begin_utterance()

    assert injector.locked is not None
    d.stop()


def test_microphone_only_runs_during_an_utterance():
    mic = FakeMic()
    d = build(mic=mic)
    d.start()
    # start() opens one throwaway stream to warm CoreAudio, so count from here.
    idle, idle_stops = mic.started, mic.stopped

    d.begin_utterance()
    assert mic.started == idle + 1

    d.end_utterance()
    d.wait_idle()
    assert mic.stopped >= idle_stops + 1
    d.stop()


def test_no_hot_microphone_while_idle():
    """The warm-up stream is opened and closed again, not left running — an
    always-on microphone lights the system indicator for as long as the daemon
    lives, and that is a trade to make deliberately, not by accident."""
    mic = FakeMic()
    d = build(mic=mic)
    d.start()

    assert mic.stopped == mic.started, "every opened stream was closed again"
    d.stop()


def test_indices_increase_across_dictations():
    injector = FakeInjector()
    d = build(stt=FakeStt(texts=["one", "two"]), injector=injector)
    d.start()
    for _ in range(2):
        d.begin_utterance()
        d.end_utterance()
        d.wait_idle()

    assert [i for i, _ in injector.injected] == [0, 1]
    d.stop()


# --- warm-up -----------------------------------------------------------------


def test_model_is_warmed_at_startup():
    """P1 measured 2.76s for the first utterance against 1.20s after. That cost
    belongs at daemon start, not on the author's first sentence."""
    stt = FakeStt()
    d = build(stt=stt)
    d.start()

    assert stt.calls, "startup should have run a throwaway transcription"
    d.stop()


def test_warm_up_output_is_discarded():
    injector = FakeInjector()
    d = build(injector=injector)
    d.start()

    assert injector.injected == []
    d.stop()


def test_warm_up_failure_does_not_stop_the_daemon():
    """A cold model is slow, not broken."""
    d = build(stt=FakeStt(fail=True))
    d.start()  # must not raise
    d.stop()


# --- queueing ----------------------------------------------------------------


def test_a_second_press_during_transcription_is_queued_not_dropped():
    """铁律 8 in its most literal form: the user spoke, so the words exist."""
    injector = FakeInjector()
    d = build(stt=FakeStt(texts=["one", "two", "three"]), injector=injector)
    d.start()
    for _ in range(3):
        d.begin_utterance()
        d.end_utterance()
    d.wait_idle()

    assert len(injector.injected) == 3
    d.stop()


def test_ending_without_beginning_is_ignored():
    stt = FakeStt()
    d = build(stt=stt)
    d.start()
    before = len(stt.calls)
    d.end_utterance()
    d.wait_idle()

    assert len(stt.calls) == before
    d.stop()


def test_beginning_twice_does_not_start_two_microphones():
    mic = FakeMic()
    d = build(mic=mic)
    d.start()
    baseline = mic.started

    d.begin_utterance()
    d.begin_utterance()

    assert mic.started == baseline + 1
    d.stop()


# --- failure paths -----------------------------------------------------------


def test_a_transcription_failure_loses_one_utterance_not_the_daemon():
    d = build(stt=FakeStt(fail=True))
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert d.running is True
    d.stop()


def test_empty_transcript_is_not_injected():
    injector = FakeInjector()
    d = build(stt=FakeStt(texts=["   "]), injector=injector)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert injector.injected == []
    d.stop()


def test_scratchpad_keeps_everything_even_in_cursor_mode():
    """The archive is the safety net under injection, not an alternative to it."""
    d = build()
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert len(d.scratchpad) == 1
    d.stop()


# --- configuration -----------------------------------------------------------


def test_scratchpad_mode_does_not_inject():
    injector = FakeInjector()
    d = build(config=AppConfig(dictate_target="scratchpad"), injector=injector)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert injector.injected == []
    assert len(d.scratchpad) == 1
    d.stop()


def test_polish_is_off_unless_configured():
    polished = []
    d = build(polish=lambda text, context=None: polished.append(text) or text.upper())
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert polished == [], "铁律 10: the safe setting is the one you get by default"
    d.stop()


def test_polish_runs_when_enabled():
    polished = []
    injector = FakeInjector()
    d = build(
        config=AppConfig(polish_enabled=True),
        injector=injector,
        polish=lambda text, context=None: polished.append(text) or text.upper(),
    )
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert polished == ["Translation is rewriting."]
    assert injector.injected[0][1] == "TRANSLATION IS REWRITING."
    d.stop()


def test_polish_failure_injects_the_raw_transcript():
    """铁律 8. The words reach the document even when the model does not."""
    def broken(text, context=None):
        raise RuntimeError("ollama is down")

    injector = FakeInjector()
    d = build(config=AppConfig(polish_enabled=True), injector=injector, polish=broken)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert injector.injected == [(0, "Translation is rewriting.")]
    d.stop()


def test_vocabulary_reaches_the_model():
    """§4.1g layer 1 — measured to fix "Thorinization" at no latency cost."""
    stt = FakeStt()
    d = build(config=AppConfig(vocabulary=["Venuti", "异化"]), stt=stt)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert "Venuti" in stt.calls[-1]["prompt"]
    d.stop()


# --- lifecycle ---------------------------------------------------------------


def test_stop_closes_the_hotkey_listener():
    d = build()
    d.start()
    listener = d.hotkey
    d.stop()

    assert listener.closed is True


def test_stop_releases_the_injection_target():
    injector = FakeInjector()
    d = build(injector=injector)
    d.start()
    d.begin_utterance()
    d.stop()

    assert injector.locked is None


def test_stop_is_idempotent():
    d = build()
    d.start()
    d.stop()
    d.stop()


def test_refuses_to_start_without_the_hotkey_permission():
    class Deaf(FakeHotkey):
        def is_available(self):
            return False, "no Accessibility permission"

        def start(self):
            raise daemon_mod.HotkeyError("no Accessibility permission")

    d = daemon_mod.DictationDaemon(
        config=AppConfig(),
        stt=FakeStt(),
        make_mic=lambda: FakeMic(),
        injector=FakeInjector(),
        hotkey_factory=lambda on_event: Deaf(on_event),
    )
    with pytest.raises(daemon_mod.HotkeyError):
        d.start()


def test_timing_is_recorded_per_dictation():
    d = build()
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert d.last_timing is not None
    assert "transcription" in d.last_timing.report()
    d.stop()


def test_timing_marks_polish_as_skipped_when_off():
    d = build()
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert "off" in d.last_timing.report()
    d.stop()


# --- the silence gate (added after a live run hallucinated) ------------------


def test_silence_is_discarded_before_it_reaches_the_model():
    """A live run with an empty room produced "you" and "Good job." — Whisper
    invents text for room tone. Injecting a fabricated sentence into the
    author's document is the worst outcome available here."""
    stt = FakeStt()
    d = build(stt=stt, speech_check=lambda audio: False)
    d.start()
    calls_after_warmup = len(stt.calls)

    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert len(stt.calls) == calls_after_warmup, "the model was never asked"
    assert len(d.scratchpad) == 0
    d.stop()


def test_silence_is_reported_in_the_timing():
    d = build(speech_check=lambda audio: False)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert "no speech" in d.last_timing.report()
    d.stop()


def test_a_broken_speech_check_lets_audio_through(monkeypatch):
    """A hallucinated sentence is bad; refusing to transcribe anything is worse."""
    stt = FakeStt()
    d = build(stt=stt, speech_check=None)  # use the real path
    monkeypatch.setattr(
        daemon_mod, "SileroVad",
        lambda: (_ for _ in ()).throw(RuntimeError("onnx session died")),
    )
    d.start()
    before = len(stt.calls)
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert len(stt.calls) > before, "a broken gate must not silence the tool"
    d.stop()


def test_the_mic_is_closed_off_the_critical_path():
    """PortAudio's close takes 130ms on this machine. Queueing first keeps that
    off the user's latency."""
    mic = FakeMic()
    slow = {"closed": False}

    def slow_stop():
        import time as _t
        _t.sleep(0.15)
        slow["closed"] = True

    mic.stop = slow_stop
    d = build(mic=mic)
    d.start()
    d.begin_utterance()

    import time as _t
    t0 = _t.perf_counter()
    d.end_utterance()
    elapsed = _t.perf_counter() - t0

    assert elapsed < 0.1, f"end_utterance blocked for {elapsed*1000:.0f}ms"
    d.wait_idle()
    d.stop()


def test_configured_language_is_passed_to_the_model():
    """Measured 2026-08-10: telling Whisper the language costs 1067ms on 8s of
    speech, letting it guess costs 1927ms. The detection pass alone is more
    than half the entire latency budget."""
    stt = FakeStt()
    d = build(config=AppConfig(dictate_language="en"), stt=stt)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert stt.calls[-1].get("language") == "en"
    d.stop()


def test_audio_is_warmed_at_startup(monkeypatch):
    """Measured 2026-08-10: the first MicSource of a process takes 713ms to
    deliver audio, every one after ~220ms. Unwarmed, that 0.5s difference is
    silently taken out of the opening of the author's first sentence.

    Patched because the dev machine's default input IS its output (AirPods), so
    the real check correctly skips the warm-up there.
    """
    monkeypatch.setattr(daemon_mod, "shared_with_output", lambda idx: None)
    mic = FakeMic()
    d = build(mic=mic)
    d.start()

    assert mic.started >= 1, "startup should have opened a throwaway stream"
    d.stop()


def test_audio_warm_up_failure_does_not_stop_the_daemon():
    def broken():
        raise RuntimeError("no audio device")

    d = daemon_mod.DictationDaemon(
        config=AppConfig(),
        stt=FakeStt(),
        make_mic=broken,
        injector=FakeInjector(),
        hotkey_factory=lambda on_event: FakeHotkey(on_event),
        speech_check=lambda a: True,
    )
    d.start()  # must not raise
    d.stop()


def test_chinese_gets_a_simplified_primer():
    """Whisper's Chinese output wanders between Simplified and Traditional and
    sometimes drops punctuation. Showing it one well-formed Simplified sentence
    steadies both — measured 2026-08-10."""
    stt = FakeStt()
    d = build(config=AppConfig(dictate_language="zh"), stt=stt)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert "简体中文" in stt.calls[-1]["prompt"]
    d.stop()


def test_english_gets_no_chinese_primer():
    stt = FakeStt()
    d = build(config=AppConfig(dictate_language="en", vocabulary=["Venuti"]), stt=stt)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert "简体中文" not in stt.calls[-1]["prompt"]
    assert "Venuti" in stt.calls[-1]["prompt"]
    d.stop()


def test_primer_and_vocabulary_combine():
    stt = FakeStt()
    d = build(config=AppConfig(dictate_language="zh", vocabulary=["semiotic", "符号"]), stt=stt)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    prompt = stt.calls[-1]["prompt"]
    assert "简体中文" in prompt and "semiotic" in prompt and "符号" in prompt
    d.stop()


def test_punctuation_width_is_normalised(monkeypatch):
    """The author's "中英文之间的标点区分不明显". Mechanical, before anything
    else sees the text, and it never touches a word."""
    stt = FakeStt(texts=["他说,I want to demonstrate，and then 他停下了."])
    d = build(stt=stt)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert d.scratchpad.entries[0].text == "他说，I want to demonstrate, and then 他停下了。"
    d.stop()


def test_warm_up_is_skipped_on_a_shared_bluetooth_device(monkeypatch):
    """Opening a headset's microphone switches it to call mode, degrading
    whatever the author is listening to. Half a second off the first sentence
    is not worth interrupting their music before they have said anything."""
    monkeypatch.setattr(daemon_mod, "shared_with_output", lambda idx: "AirPods Pro")
    mic = FakeMic()
    d = build(mic=mic)
    d.start()

    assert mic.started == 0
    d.stop()


def test_warm_up_runs_on_a_dedicated_input(monkeypatch):
    monkeypatch.setattr(daemon_mod, "shared_with_output", lambda idx: None)
    mic = FakeMic()
    d = build(mic=mic)
    d.start()

    assert mic.started >= 1
    d.stop()


# --- flushing buffered text (the call that was missing) ----------------------


class BufferingInjector(FakeInjector):
    """Behaves like the real one when focus has moved: holds instead of injecting."""

    def __init__(self, blocked=True):
        super().__init__()
        self.blocked = blocked
        self.flushed = []

    def inject(self, index, text):
        if self.blocked:
            self.pending += 1
            return InjectionResult(injected=False, buffered=True, reason="focus moved")
        return super().inject(index, text)

    def flush(self):
        if not self.pending:
            return InjectionResult(injected=False, reason="nothing pending")
        self.flushed.append(self.pending)
        self.pending = 0
        return InjectionResult(injected=True)


def test_buffered_text_is_flushed_when_the_target_returns(monkeypatch):
    """Design §4.1e promised this. Injector.flush() existed and was tested;
    nothing called it, so text buffered during a focus change sat there
    forever."""
    from backend import injection

    injector = BufferingInjector()
    # The target is frontmost again — which is the condition the worker checks.
    monkeypatch.setattr(injection, "_frontmost", lambda: Target(pid=1, name="TestApp"))

    d = build(injector=injector)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    import time as _t
    _t.sleep(0.5)  # the worker checks on each idle tick

    assert injector.flushed, "the worker should have flushed on returning"
    d.stop()


def test_shutdown_flushes_rather_than_discarding():
    """Quitting must not silently drop words the author already said."""
    injector = BufferingInjector()
    d = build(injector=injector)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()
    d.stop()

    assert injector.flushed, "pending text must go somewhere on shutdown"


class LongMic(FakeMic):
    def __init__(self, seconds):
        super().__init__(seconds=seconds)


def test_a_long_hold_is_split_at_pauses(monkeypatch):
    """65 seconds of unbroken speech came back as one run-on sentence with a
    single full stop. Whisper punctuates what it can see the shape of; handed
    one undifferentiated block it has nothing to go on."""
    stt = FakeStt(texts=["第一句。", "第二句。", "第三句。"])
    d = build(stt=stt, mic=LongMic(30.0))
    monkeypatch.setattr(
        d, "_split_at_pauses",
        lambda audio: [audio[:16000], audio[16000:32000], audio[32000:48000]],
    )
    d.start()
    before = len(stt.calls)
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert len(stt.calls) - before == 3, "one model pass per clause"
    # FakeStt cycles its texts and the warm-up consumed one, so assert the
    # pieces were joined rather than pinning an order the fake decides.
    text = d.scratchpad.entries[0].text
    assert all(piece in text for piece in ("第一句。", "第二句。", "第三句。"))
    d.stop()


def test_a_short_hold_is_not_split():
    """Below the threshold Whisper's own windowing copes, and each extra cut
    costs another model pass."""
    stt = FakeStt()
    d = build(stt=stt, mic=FakeMic(seconds=3.0))
    d.start()
    before = len(stt.calls)
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert len(stt.calls) - before == 1
    d.stop()


def test_a_failed_split_falls_back_to_one_pass(monkeypatch):
    """A broken VAD must cost punctuation quality, never the transcript."""
    stt = FakeStt()
    d = build(stt=stt, mic=LongMic(30.0))
    monkeypatch.setattr(d, "_split_at_pauses", lambda audio: [])
    d.start()
    before = len(stt.calls)
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert len(stt.calls) - before == 1
    assert d.scratchpad.entries[0].text
    d.stop()
