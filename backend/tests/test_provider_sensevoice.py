"""SenseVoice, the candidate for Chinese and for Chinese with English in it.

Whisper's two limits here are in its weights, not its settings: one language
token per 30-second window, and sparse Chinese punctuation learned from
subtitle data. Configuration cannot reach either, which is why a second engine
is on the table at all.
"""

import sys
import types

import numpy as np
import pytest

from backend.providers import sensevoice as sv


class FakeStream:
    def __init__(self, text):
        self.result = types.SimpleNamespace(text=text)
        self.fed = []

    def accept_waveform(self, rate, samples):
        self.fed.append((rate, len(samples)))


@pytest.fixture
def fake_sherpa(monkeypatch):
    state = {"built": [], "text": "<|zh|><|NEUTRAL|><|Speech|><|withitn|>多模态语篇的符号资源。"}

    class FakeRecogniser:
        @staticmethod
        def from_sense_voice(**kwargs):
            state["built"].append(kwargs)
            return FakeRecogniser()

        def create_stream(self):
            state["stream"] = FakeStream(state["text"])
            return state["stream"]

        def decode_stream(self, stream):
            state["decoded"] = True

    module = types.ModuleType("sherpa_onnx")
    module.OfflineRecognizer = FakeRecogniser
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    monkeypatch.setattr(sv.SenseVoiceProvider, "_paths", lambda self: ("m.onnx", "t.txt"))
    return state


def audio(seconds=3):
    return np.zeros(16000 * seconds, dtype=np.float32)


def test_strips_rich_transcription_tags(fake_sherpa):
    """SenseVoice returns language, emotion and audio-event tags beside the
    words. They are metadata; the author dictates into documents."""
    assert sv.SenseVoiceProvider().transcribe(audio()) == "多模态语篇的符号资源。"


def test_itn_is_on_because_that_is_where_punctuation_comes_from(fake_sherpa):
    sv.SenseVoiceProvider().transcribe(audio())
    assert fake_sherpa["built"][0]["use_itn"] is True


def test_audio_reaches_the_model_at_16k(fake_sherpa):
    sv.SenseVoiceProvider().transcribe(audio(2))
    assert fake_sherpa["stream"].fed == [(16000, 32000)]


def test_model_is_built_once(fake_sherpa):
    provider = sv.SenseVoiceProvider()
    for _ in range(3):
        provider.transcribe(audio())
    assert len(fake_sherpa["built"]) == 1


def test_empty_audio_never_loads_the_model(fake_sherpa):
    assert sv.SenseVoiceProvider().transcribe(np.zeros(0, dtype=np.float32)) == ""
    assert fake_sherpa["built"] == []


def test_unavailable_without_sherpa(monkeypatch):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    available, reason = sv.SenseVoiceProvider().is_available()
    assert available is False
    assert "sherpa" in reason.lower()


def test_it_declares_that_it_has_no_prompt_channel():
    """Design §4.1g layer 1 — the free terminology fix that recovered
    "foreignisation" from "Thorinization" — has nowhere to go on this engine.
    That is a real cost of switching, and it should be visible rather than
    discovered later."""
    assert sv.SenseVoiceProvider().supports_prompt is False


def test_a_prompt_is_accepted_and_ignored_rather_than_crashing(fake_sherpa):
    """The protocol passes one; a provider that cannot use it must still take it."""
    assert sv.SenseVoiceProvider().transcribe(audio(), initial_prompt="Venuti")


def test_satisfies_the_protocol(fake_sherpa):
    from backend.providers.stt import SttProvider

    assert isinstance(sv.SenseVoiceProvider(), SttProvider)


def test_sherpa_is_imported_lazily():
    """铁律 6. The registry must be able to load this module on a machine
    without sherpa-onnx in order to report it unavailable."""
    import ast

    tree = ast.parse(open(sv.__file__).read())
    top = {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    top |= {n.module for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
    assert "sherpa_onnx" not in top


def test_the_official_runtime_is_not_used():
    """funasr declares 71 dependencies including torch. 铁律 5 rules it out —
    the VAD was built on raw ONNX weights precisely to avoid dragging in 476M
    for a 1.2M model, and importing funasr here would undo that."""
    source = open(sv.__file__).read()
    assert "import funasr" not in source
