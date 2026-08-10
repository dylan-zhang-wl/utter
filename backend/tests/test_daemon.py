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
    id = "fake"
    display_name = "Fake STT"

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

def test_a_long_hold_is_one_model_pass():
    """Splitting was tried and measured against the author's real dictation: a
    24.2s hold split into clauses came back with *fewer* punctuation marks than
    a 16.4s one left whole, and cost five model passes instead of one. Whisper's
    Chinese punctuation is sparse at every length, so there was nothing to
    unlock. Punctuation is a language task; it belongs to polish."""
    stt = FakeStt()
    d = build(stt=stt, mic=LongMic(40.0))
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
    monkeypatch.setattr(d, "_split_evenly", lambda audio, seconds=14.0: [])
    d.start()
    before = len(stt.calls)
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert len(stt.calls) - before == 1
    assert d.scratchpad.entries[0].text
    d.stop()

def test_clause_split_uses_a_shorter_silence_than_utterance_split():
    """600ms answers "is that utterance over"; a clause boundary is a breath."""
    from backend.config import AppConfig

    assert daemon_mod.CLAUSE_SILENCE_MS < AppConfig().vad_silence_ms


def test_switching_provider_persists_the_choice(monkeypatch, tmp_path):
    """The menu's first version mutated the in-memory config and nothing else,
    so every setting reverted on restart — v1's fake Save button, rebuilt."""
    from backend import config as cfg
    from backend.providers import stt as stt_mod

    class Other(FakeStt):
        id = "sensevoice"
        display_name = "SenseVoice"

    monkeypatch.setattr(stt_mod, "get_stt_provider", lambda **kw: Other())
    saved = {}
    monkeypatch.setattr(cfg, "save", lambda c, base_dir=None: saved.update(id=c.stt_provider))

    d = build()
    d.start()
    ok, _ = d.switch_provider("sensevoice")

    assert ok is True
    assert d.stt.id == "sensevoice"
    assert saved["id"] == "sensevoice", "the choice must survive a restart"
    d.stop()


def test_switching_to_an_unavailable_provider_keeps_the_old_one(monkeypatch):
    from backend.providers import stt as stt_mod

    original = FakeStt()
    monkeypatch.setattr(stt_mod, "get_stt_provider", lambda **kw: FakeStt())
    d = build(stt=original)
    d.start()
    ok, message = d.switch_provider("nonexistent")

    assert ok is False and "不可用" in message
    assert d.stt is original, "a failed switch must not leave the daemon engineless"
    d.stop()


class SpyOverlay:
    def __init__(self):
        self.states, self.hides = [], 0

    def show(self, state="recording"):
        self.states.append(state)

    def set_state(self, state, caption=""):
        self.states.append(state)

    def feed(self, level, caption=""):
        pass

    def hide(self):
        self.hides += 1


def test_the_overlay_is_hidden_after_a_failed_transcription():
    """It returned early and left the panel stuck on 「转录中」 with the bars
    frozen — worse than no overlay, because it reports work that is not
    happening."""
    overlay = SpyOverlay()
    d = build(stt=FakeStt(fail=True), overlay=overlay)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert overlay.hides >= 1
    d.stop()


def test_the_overlay_is_hidden_after_an_empty_transcript():
    overlay = SpyOverlay()
    d = build(stt=FakeStt(texts=["   "]), overlay=overlay)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()

    assert overlay.hides >= 1
    d.stop()


def test_each_dictation_stops_the_previous_level_pump():
    """The stop event was replaced rather than set, so the old thread went on
    polling the new one — unset — and never exited. One leaked thread per
    dictation, all drawing to the same panel."""
    import threading as th

    overlay = SpyOverlay()
    d = build(overlay=overlay)
    d.start()
    before = th.active_count()
    for _ in range(5):
        d.begin_utterance()
        d.end_utterance()
        d.wait_idle()

    import time as _t
    _t.sleep(0.3)
    assert th.active_count() <= before + 1, "level pumps are piling up"
    d.stop()


def test_the_report_belongs_to_the_sentence_it_is_printed_beside():
    """`last_timing` was assigned after `on_text`, so the CLI printed the
    PREVIOUS utterance's numbers under the current utterance's text.

    Not cosmetic. A whole evening of diagnosis was done on audio durations
    paired with the wrong words, which sent the search after a truncation bug
    that the numbers, read correctly, did not support.
    """
    class EchoLength:
        """Says how long the audio it was handed is, so the text and the report
        can be checked against each other directly."""

        id = display_name = "echo"

        def transcribe(self, audio, language=None, initial_prompt=None):
            return f"{len(audio) / 16000:.1f}"

    seen = []
    d = build(
        config=AppConfig(close_sentences=False),  # else the echo gains a full stop
        stt=EchoLength(),
        on_text=lambda u: seen.append(
            (u.raw_text, d.last_timing.audio_seconds if d.last_timing else None)
        ),
    )
    d.start()
    for seconds in (1.0, 2.0, 3.0):
        d.make_mic = lambda s=seconds: FakeMic(seconds=s)
        d.begin_utterance()
        d.end_utterance()
        d.wait_idle()
    d.stop()

    assert [text for text, _ in seen] == ["1.0", "2.0", "3.0"]
    for text, reported in seen:
        assert float(text) == reported, "the report belongs to a different sentence"


def test_the_report_says_how_long_the_key_was_held():
    d = build()
    d.start()
    d.begin_utterance(at=100.0)
    d.end_utterance(at=104.5)
    d.wait_idle()
    d.stop()

    assert d.last_timing.held_seconds == pytest.approx(4.5)
    assert "按住 4.5s" in d.last_timing.report()


def test_audio_that_never_arrived_is_called_out():
    """One second of recording for a twenty-second hold is not a model problem,
    and the report must not let it read like one."""
    d = build(mic=FakeMic(seconds=1.0))
    d.start()
    d.begin_utterance(at=0.0)
    d.end_utterance(at=20.0)
    d.wait_idle()
    d.stop()

    report = d.last_timing.report()
    assert "19.0s 没录进来" in report
    assert "不是模型的问题" in report


def test_a_hold_that_matches_its_audio_is_not_flagged():
    """The microphone opens ~220ms after the key goes down, so a small gap is
    normal and must not cry wolf on every single dictation."""
    d = build(mic=FakeMic(seconds=3.0))
    d.start()
    d.begin_utterance(at=0.0)
    d.end_utterance(at=3.25)
    d.wait_idle()
    d.stop()

    assert "没录进来" not in d.last_timing.report()


def test_the_report_says_how_much_of_the_recording_was_speech():
    """Separates "the model dropped what I said" from "I held the key for
    twenty seconds and spoke for three"."""
    d = build(speech_check=None)
    d.make_mic = lambda: FakeMic(seconds=2.0)
    # Stand in for the VAD: the fake mic's DC level is genuinely not speech.
    d._speech_seconds = lambda audio: 0.4
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()
    d.stop()

    assert d.last_timing.speech_seconds == pytest.approx(0.4)
    assert "其中说话 0.4s" in d.last_timing.report()


def test_input_the_system_discarded_is_reported():
    """PortAudio's overflow count has been collected since day one and shown
    never — the silent loss this project keeps promising not to do."""
    mic = FakeMic()
    mic.overflows = 7
    d = build(mic=mic)
    d.start()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()
    d.stop()

    assert "系统丢了 7 次输入缓冲" in d.last_timing.report()


def test_a_vocabulary_that_outgrows_whispers_prompt_is_called_out(capsys):
    """Whisper truncates initial_prompt from the front at 223 tokens and says
    nothing. A terminology list the author keeps adding to WILL reach that, and
    finding out from a transcript is not acceptable."""
    config = AppConfig(vocabulary=[f"术语{i}foreignisation" for i in range(60)])
    d = build(config=config)
    d.start()
    d.stop()

    out = capsys.readouterr().out
    assert "术语表太长了" in out
    assert "从头截掉" in out


def test_the_authors_current_list_does_not_trip_the_warning():
    """30 terms is 340 characters — 133 tokens against a 223 ceiling. There is
    room to grow, and crying wolf now would train the author to ignore it."""
    config = AppConfig(vocabulary=["multimodality", "semiotic resource", "Venuti"] * 10)
    d = build(config=config)
    assert d._check_vocabulary_fits() is None


def test_the_microphone_starts_before_anything_else_in_begin_utterance():
    """Locking the target costs 82ms of AppKit and the overlay costs more, and
    every one of those milliseconds was speech the author had already begun.
    Nothing in begin_utterance depends on the microphone, so it goes first."""
    order = []

    class Watching(FakeInjector):
        def lock_target(self, target=None):
            order.append("lock")
            return super().lock_target(target)

    class WatchingMic(FakeMic):
        def start(self):
            order.append("mic")
            return super().start()

    d = build(mic=WatchingMic(), injector=Watching())
    d.start()
    order.clear()
    d.begin_utterance()
    d.end_utterance()
    d.wait_idle()
    d.stop()

    assert order[:2] == ["mic", "lock"]


def test_the_report_says_how_much_of_the_first_word_was_lost():
    mic = FakeMic()
    d = build(mic=mic)
    d.start()
    d.begin_utterance(at=100.0)
    mic.first_chunk_at = 100.31          # CoreAudio's usual ~240-310ms
    d.end_utterance(at=104.0)
    d.wait_idle()
    d.stop()

    assert "开麦 310 ms" in d.last_timing.report()


def test_an_unusually_slow_microphone_open_is_flagged():
    """~240ms is CoreAudio and unavoidable. 1.5s means something else has the
    device — a second forgotten daemon, in the case that prompted this."""
    mic = FakeMic()
    d = build(mic=mic)
    d.start()
    d.begin_utterance(at=100.0)
    mic.first_chunk_at = 101.5
    d.end_utterance(at=104.0)
    d.wait_idle()
    d.stop()

    assert "太久了" in d.last_timing.report()


def test_the_microphone_open_is_not_counted_in_the_end_to_end_total():
    """It happens while the author is still talking. Adding it to the total
    would report a wait they never had."""
    mic = FakeMic()
    d = build(mic=mic)
    d.start()
    d.begin_utterance(at=100.0)
    mic.first_chunk_at = 100.31
    d.end_utterance(at=104.0)
    d.wait_idle()
    d.stop()

    watch = d.last_timing
    assert watch.mic_open_ms == pytest.approx(310, abs=1)
    assert not any("开麦" in s.name for s in watch.stages)
    assert 310 not in [round(s.ms) for s in watch.stages if s.ms is not None]


def test_the_polish_toggle_actually_rebuilds_the_callable():
    """The menu flipped config.polish_enabled and nothing else, so switching
    polish on mid-session did nothing at all — v1's Save button again."""
    built = []

    def factory(config):
        built.append(config.polish_enabled)
        return (lambda text, context=None: text + "。") if config.polish_enabled else None

    d = build(polish_factory=factory)
    assert d.polish is None

    on, _ = d.set_polish(True)
    assert on and d.polish is not None
    assert built == [True]

    d.set_polish(False)
    assert d.polish is None


def test_asking_for_polish_with_no_key_leaves_the_menu_honest():
    """A tick next to 润色 that does not polish is worse than no tick."""
    d = build(polish_factory=lambda config: None)

    on, why = d.set_polish(True)

    assert on is False
    assert d.config.polish_enabled is False, "the menu must not show it as on"
    assert "API key" in why


def test_changing_the_level_rebuilds_too():
    seen = []
    d = build(polish_factory=lambda c: seen.append(c.polish_level) or (lambda t, context=None: t))

    d.set_polish(True, "heavy")

    assert d.config.polish_level == "heavy"
    assert seen == ["heavy"]
