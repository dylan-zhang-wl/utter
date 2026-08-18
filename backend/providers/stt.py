"""The speech-to-text interface, and the registry that chooses an implementation.

The interface is a job description — "here is some audio, give me back text" —
and a provider is an applicant. The rest of the platform never learns which one
took the job, which is what lets the same code run on the author's M2 and on a
Windows machine with no MLX at all.

Selection has three inputs, in decreasing authority:

  1. availability — an unavailable provider is never chosen, whatever else says
  2. the user's explicit choice, when it is available
  3. hardware preference — MLX first on Apple Silicon, CTranslate2 elsewhere,
     cloud last everywhere, because local-first is the product (design §2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from backend.hardware import Hardware, HardwareKind
from backend.hardware import detect as detect_hardware

log = logging.getLogger(__name__)

# "auto" is config.stt_provider's default and means "you decide", not a provider.
AUTO = "auto"


@dataclass(frozen=True)
class Detailed:
    """A transcription plus what the model thought of it.

    `confidence` is Whisper's own `avg_logprob`, duration-weighted across the
    segments of one utterance. Higher is surer; real speech sits well above a
    hallucination. Recorded from the first meeting and displayed nowhere until
    the distribution on real audio is known — the only figure available today
    is -0.82 for two seconds of quiet noise that transcribed as " You".

    `no_speech` is the model's own no-speech probability. Kept for the record
    and *not* used as a hallucination signal: on that same invented " You" it
    read 8e-11, meaning the model was certain it had heard speech.
    """

    text: str
    confidence: float | None = None
    no_speech: float | None = None


def transcribe_detailed(provider, audio, language=None, initial_prompt=None) -> Detailed:
    """Ask for detail; fall back to plain text for providers without it.

    Optional rather than required, so no existing provider has to change and
    dictation is untouched.
    """
    detailed = getattr(provider, "transcribe_detailed", None)
    if detailed is not None:
        try:
            return detailed(audio, language=language, initial_prompt=initial_prompt)
        except Exception:
            log.warning("detailed transcription failed, falling back", exc_info=True)
    return Detailed(text=provider.transcribe(
        audio, language=language, initial_prompt=initial_prompt))


@runtime_checkable
class SttProvider(Protocol):
    id: str
    display_name: str

    def is_available(self) -> tuple[bool, str]:
        """(usable?, human-readable reason when not).

        The reason is shown by `utter doctor` on a machine the author cannot log
        into, so it should say what to do — "faster_whisper not installed", not
        "unavailable".
        """
        ...

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> str:
        """16 kHz mono float32 in, text out.

        `initial_prompt` carries the user's vocabulary list (design §4.1g layer
        1): biasing the model costs nothing and works with the LLM switched off,
        so it is part of the interface rather than a later addition.
        """
        ...


@dataclass(frozen=True)
class ProviderStatus:
    id: str
    display_name: str
    available: bool
    reason: str


class NoProviderAvailable(RuntimeError):
    """Nothing can transcribe. Carries why, per provider."""

    def __init__(self, reasons: dict[str, str]):
        self.reasons = reasons
        detail = "; ".join(f"{pid}: {why}" for pid, why in reasons.items())
        super().__init__(f"no speech-to-text provider is available — {detail}")


# SenseVoice sits behind Whisper for now — it is a candidate under comparison,
# not yet a default. Choose it explicitly with stt_provider="sensevoice".
_PREFERENCE: dict[HardwareKind, tuple[str, ...]] = {
    "apple_silicon": ("mlx", "faster", "sensevoice", "cloud"),
    "cuda": ("faster", "mlx", "sensevoice", "cloud"),
    "cpu": ("faster", "sensevoice", "mlx", "cloud"),
}


def preference_order(hardware: Hardware) -> tuple[str, ...]:
    return _PREFERENCE[hardware.kind]


def default_providers() -> list[SttProvider]:
    """Every provider this build knows about.

    Imported lazily and defensively: a provider module that cannot import at all
    (missing optional dependency, wrong platform) must not take down the
    registry with it.
    """
    found: list[SttProvider] = []
    for module_name, class_name in (
        ("backend.providers.mlx", "MlxWhisperProvider"),
        ("backend.providers.faster", "FasterWhisperProvider"),
        ("backend.providers.sensevoice", "SenseVoiceProvider"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            found.append(getattr(module, class_name)())
        except Exception:
            log.debug("provider %s did not load", module_name, exc_info=True)
    return found


def _ordered(providers, hardware):
    order = preference_order(hardware)
    rank = {pid: i for i, pid in enumerate(order)}
    return sorted(providers, key=lambda p: rank.get(p.id, len(order)))


def _check(provider) -> tuple[bool, str]:
    try:
        return provider.is_available()
    except Exception as exc:
        # A provider whose probe explodes is unavailable, not fatal. Its failure
        # must not deny the user a different provider that works fine.
        return False, f"probe failed: {exc}"


def probe_all(
    hardware: Hardware | None = None,
    providers: list[SttProvider] | None = None,
) -> list[ProviderStatus]:
    """Availability of every provider, in preference order. Drives `utter doctor`."""
    hardware = hardware or detect_hardware()
    providers = default_providers() if providers is None else providers

    report = []
    for provider in _ordered(providers, hardware):
        available, reason = _check(provider)
        report.append(
            ProviderStatus(
                id=provider.id,
                display_name=provider.display_name,
                available=available,
                reason=reason,
            )
        )
    return report


def get_stt_provider(
    hardware: Hardware | None = None,
    providers: list[SttProvider] | None = None,
    preferred: str | None = None,
) -> SttProvider:
    """Choose a provider, or explain why none can be chosen."""
    hardware = hardware or detect_hardware()
    providers = default_providers() if providers is None else providers
    candidates = _ordered(providers, hardware)

    if preferred and preferred != AUTO:
        # An explicit choice that is unavailable — a stale config naming a
        # provider that no longer works — falls through rather than bricking the
        # app. Note it, because silently ignoring the user's setting is its own
        # kind of surprise.
        for provider in candidates:
            if provider.id == preferred:
                available, reason = _check(provider)
                if available:
                    return provider
                log.warning(
                    "configured provider %r is unavailable (%s), falling back",
                    preferred,
                    reason,
                )
                break
        else:
            log.warning("configured provider %r is not a known provider", preferred)

    reasons: dict[str, str] = {}
    for provider in candidates:
        available, reason = _check(provider)
        if available:
            return provider
        reasons[provider.id] = reason or "unavailable"

    raise NoProviderAvailable(reasons)


def draft_provider(
    tier: str,
    hardware: Hardware | None = None,
    providers: list[SttProvider] | None = None,
    preferred: str | None = None,
) -> SttProvider | None:
    """A second, much cheaper provider, for work that is never written down.

    Listen mode's preview re-reads the sentence in progress every second. That
    is only affordable with a small model — measured on an M2, whisper-base-q4
    answers a 10-second buffer in 166ms against large-v3-turbo's 1109ms.

    Which model a tier means is the catalog's business and which provider runs
    it is the registry's, so this asks for a working provider and reopens it at
    the smaller tier rather than naming a class (铁律 6).

    Returns None when the chosen provider ships a single model and has no
    smaller tier to drop to. That is not an error: the preview is a luxury and
    nothing in the record depends on it.
    """
    import inspect

    base = get_stt_provider(hardware=hardware, providers=providers,
                            preferred=preferred)
    kind = type(base)
    if "tier" not in inspect.signature(kind.__init__).parameters:
        log.info("provider %r has no tiers, so no preview", getattr(base, "id", kind))
        return None
    smaller = kind(tier)
    available, why = _check(smaller)
    if not available:
        log.info("preview provider unavailable: %s", why)
        return None
    return smaller
