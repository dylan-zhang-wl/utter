"""P1 Task 5 — the SttProvider protocol and an availability-aware registry.

铁律 6 lives here. Nothing above this layer may know that MLX exists, because
MLX only has a backend on Apple Silicon and the author owns a machine that is
not one.

Every test uses fake providers. Real ones arrive in Tasks 6 and 7.
"""

import numpy as np
import pytest

from backend.hardware import Hardware
from backend.providers import stt


class FakeProvider:
    def __init__(self, id_, available=True, reason="", text="hello"):
        self.id = id_
        self.display_name = f"Fake {id_}"
        self._available = available
        self._reason = reason
        self._text = text
        self.calls = []

    def is_available(self):
        return (self._available, self._reason)

    def transcribe(self, audio, language=None, initial_prompt=None):
        self.calls.append((len(audio), language, initial_prompt))
        return self._text


def hw(kind):
    return Hardware(kind=kind, ram_gb=16)


def test_picks_the_first_available_provider():
    down = FakeProvider("mlx", available=False, reason="not Apple Silicon")
    up = FakeProvider("faster", text="from faster")

    chosen = stt.get_stt_provider(hardware=hw("cpu"), providers=[down, up])
    assert chosen is up


def test_skips_unavailable_even_when_preferred_by_hardware():
    down = FakeProvider("mlx", available=False, reason="mlx_whisper not installed")
    up = FakeProvider("faster")

    chosen = stt.get_stt_provider(hardware=hw("apple_silicon"), providers=[down, up])
    assert chosen is up, "hardware preference must not override availability"


def test_raises_when_nothing_is_available():
    providers = [
        FakeProvider("mlx", available=False, reason="not Apple Silicon"),
        FakeProvider("faster", available=False, reason="faster_whisper not installed"),
    ]
    with pytest.raises(stt.NoProviderAvailable):
        stt.get_stt_provider(hardware=hw("cpu"), providers=providers)


def test_the_error_carries_every_reason():
    """The user sees this message and has to act on it, so it must say what to
    install rather than just that nothing worked."""
    providers = [
        FakeProvider("mlx", available=False, reason="not Apple Silicon"),
        FakeProvider("faster", available=False, reason="faster_whisper not installed"),
    ]
    with pytest.raises(stt.NoProviderAvailable) as exc:
        stt.get_stt_provider(hardware=hw("cpu"), providers=providers)

    message = str(exc.value)
    assert "not Apple Silicon" in message
    assert "faster_whisper not installed" in message


def test_apple_silicon_prefers_mlx():
    mlx = FakeProvider("mlx")
    faster = FakeProvider("faster")
    chosen = stt.get_stt_provider(hardware=hw("apple_silicon"), providers=[faster, mlx])
    assert chosen is mlx, "declaration order must not beat hardware preference"


@pytest.mark.parametrize("kind", ["cuda", "cpu"])
def test_non_apple_prefers_faster_whisper(kind):
    mlx = FakeProvider("mlx")
    faster = FakeProvider("faster")
    chosen = stt.get_stt_provider(hardware=hw(kind), providers=[mlx, faster])
    assert chosen is faster


@pytest.mark.parametrize("kind", ["apple_silicon", "cuda", "cpu"])
def test_cloud_is_always_last(kind):
    """Local-first is the product's whole premise (design §2). Cloud is a
    fallback the user opts into, never something that wins a race."""
    assert stt.preference_order(hw(kind))[-1] == "cloud"


def test_explicit_choice_wins_over_preference():
    mlx = FakeProvider("mlx")
    faster = FakeProvider("faster")
    chosen = stt.get_stt_provider(
        hardware=hw("apple_silicon"), providers=[mlx, faster], preferred="faster"
    )
    assert chosen is faster


def test_explicit_choice_that_is_unavailable_falls_back():
    """A stale config naming a provider that no longer works must not brick the
    app — fall through to whatever does work."""
    mlx = FakeProvider("mlx")
    faster = FakeProvider("faster", available=False, reason="gone")
    chosen = stt.get_stt_provider(
        hardware=hw("apple_silicon"), providers=[mlx, faster], preferred="faster"
    )
    assert chosen is mlx


def test_auto_is_not_treated_as_a_provider_id():
    """config.stt_provider defaults to "auto", which means "you decide"."""
    mlx = FakeProvider("mlx")
    chosen = stt.get_stt_provider(
        hardware=hw("apple_silicon"), providers=[mlx], preferred="auto"
    )
    assert chosen is mlx


def test_unknown_provider_id_falls_back_rather_than_raising():
    mlx = FakeProvider("mlx")
    chosen = stt.get_stt_provider(
        hardware=hw("apple_silicon"), providers=[mlx], preferred="nonexistent"
    )
    assert chosen is mlx


def test_probe_all_reports_every_provider_with_its_reason():
    """Drives `utter doctor`, which must explain what is wrong on a machine the
    author cannot log into."""
    providers = [
        FakeProvider("mlx", available=False, reason="not Apple Silicon"),
        FakeProvider("faster"),
    ]
    report = stt.probe_all(hardware=hw("cpu"), providers=providers)

    assert [r.id for r in report] == ["faster", "mlx"], "preference order"
    assert report[0].available is True
    assert report[1].available is False
    assert report[1].reason == "not Apple Silicon"


def test_probe_all_never_raises_on_a_broken_provider():
    class Exploding:
        id = "broken"
        display_name = "Exploding"

        def is_available(self):
            raise RuntimeError("driver on fire")

        def transcribe(self, audio, language=None, initial_prompt=None):
            raise AssertionError("unreachable")

    report = stt.probe_all(hardware=hw("cpu"), providers=[Exploding()])
    assert report[0].available is False
    assert "driver on fire" in report[0].reason


def test_a_broken_provider_is_skipped_not_fatal():
    class Exploding:
        id = "broken"
        display_name = "Exploding"

        def is_available(self):
            raise RuntimeError("driver on fire")

        def transcribe(self, audio, language=None, initial_prompt=None):
            raise AssertionError("unreachable")

    good = FakeProvider("faster")
    assert stt.get_stt_provider(hardware=hw("cpu"), providers=[Exploding(), good]) is good


def test_fake_provider_satisfies_the_protocol():
    assert isinstance(FakeProvider("x"), stt.SttProvider)


def test_transcribe_signature_carries_initial_prompt():
    """§4.1g layer 1: the vocabulary list is fed to the model as a prompt, which
    costs nothing and works with the LLM switched off. It has to be in the
    protocol from the start or every implementation changes later."""
    p = FakeProvider("mlx")
    p.transcribe(np.zeros(16000, dtype=np.float32), language="en", initial_prompt="Venuti")
    assert p.calls == [(16000, "en", "Venuti")]
