"""Model catalog: the user picks a tier, the platform picks the format.

Design §5.3. The user never sees the words "MLX", "CTranslate2" or "GGUF" — they
see "均衡（推荐）" and a size in megabytes. This module is where that translation
happens, and it is the only place that knows a repo id.

Every repo below was confirmed to exist and measured against the HuggingFace API
on 2026-08-10. The sizes are the sum of weight files, not the design's original
estimates, which were roughly half the truth for the top two tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from backend.hardware import Hardware, HardwareKind

Runtime = Literal["mlx", "ctranslate2"]

TIERS = ("high", "balanced", "light", "minimal")


class UnknownTier(ValueError):
    pass


@dataclass(frozen=True)
class ModelEntry:
    tier: str
    repo: str
    runtime: Runtime
    size_mb: int
    label: str
    blurb: str
    recommended: bool = False


# Tier presentation. Deliberately free of a "big = slow" gradient: large-v3-turbo
# carries 809M parameters but only 4 decoder layers against large-v3's 32, so the
# recommended tier is both bigger and faster than the one below it. Writing the
# blurbs as a linear tradeoff would steer users away from the right default.
_PRESENTATION = {
    "high": ("高精度", "最准，口音与专业术语都稳；需要较新的机器", False),
    "balanced": ("均衡（推荐）", "准确度接近高精度，但快得多。日常首选", True),
    "light": ("轻量", "一般机器也流畅，长词与术语偶尔出错", False),
    "minimal": ("极轻", "老机器或内存紧张时应急，准确度明显下降", False),
}

# (repo, size_mb) per tier per runtime. CUDA and CPU share the CTranslate2
# weights; they differ only in compute_type, which is the provider's business.
_MLX = {
    "high": ("mlx-community/whisper-large-v3-mlx", 3084),
    "balanced": ("mlx-community/whisper-large-v3-turbo", 1614),
    "light": ("mlx-community/whisper-medium-mlx-4bit", 512),
    "minimal": ("mlx-community/whisper-base-mlx-q4", 80),
}

_CT2 = {
    "high": ("Systran/faster-whisper-large-v3", 3087),
    # Canonical id as of 2026-08-10. faster-whisper 1.2.1's own _MODELS table
    # still says "mobiuslabsgmbh/faster-whisper-large-v3-turbo"; HuggingFace
    # redirects that to this repo after an ownership transfer (same 1.64M
    # downloads on both names). Naming the canonical id keeps a first-run
    # download off a redirect, and keeps the network test able to notice the
    # next move — a silent redirect would hide it.
    "balanced": ("dropbox-dash/faster-whisper-large-v3-turbo", 1618),
    "light": ("Systran/faster-whisper-small", 484),
    "minimal": ("Systran/faster-whisper-base", 145),
}

_BY_KIND: dict[HardwareKind, tuple[Runtime, dict]] = {
    "apple_silicon": ("mlx", _MLX),
    "cuda": ("ctranslate2", _CT2),
    "cpu": ("ctranslate2", _CT2),
}


def resolve(tier: str, hardware: Hardware) -> ModelEntry:
    """Turn a tier name plus a machine into one concrete model."""
    if tier not in TIERS:
        raise UnknownTier(f"unknown tier {tier!r}; valid tiers are {', '.join(TIERS)}")

    runtime, table = _BY_KIND[hardware.kind]
    repo, size_mb = table[tier]
    label, blurb, recommended = _PRESENTATION[tier]

    return ModelEntry(
        tier=tier,
        repo=repo,
        runtime=runtime,
        size_mb=size_mb,
        label=label,
        blurb=blurb,
        recommended=recommended,
    )


def all_tiers(hardware: Hardware) -> list[ModelEntry]:
    """Every tier for this machine, in presentation order. Drives `utter models list`."""
    return [resolve(tier, hardware) for tier in TIERS]
