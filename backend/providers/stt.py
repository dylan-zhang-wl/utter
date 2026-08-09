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


_PREFERENCE: dict[HardwareKind, tuple[str, ...]] = {
    "apple_silicon": ("mlx", "faster", "cloud"),
    "cuda": ("faster", "mlx", "cloud"),
    "cpu": ("faster", "mlx", "cloud"),
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
