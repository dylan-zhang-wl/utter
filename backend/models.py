"""Downloading and accounting for model weights.

Delegates to huggingface_hub rather than hand-rolling HTTP: it already does
resumable transfers, integrity checks, and a shared cache under
~/.cache/huggingface that other tools on the machine populate too. The author's
copy of large-v3-turbo arrived via an unrelated project, and reusing it instead
of fetching 1.6G again is the single most valuable thing this module does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from backend import catalog
from backend.hardware import Hardware
from backend.hardware import detect as detect_hardware

log = logging.getLogger(__name__)

__all__ = [
    "ensure_model",
    "is_downloaded",
    "disk_usage",
    "ModelDownloadError",
    "Progress",
    "TierUsage",
    "snapshot_download",
    "LocalEntryNotFoundError",
]


class ModelDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Progress:
    repo: str
    message: str
    done: bool = False


@dataclass(frozen=True)
class TierUsage:
    tier: str
    label: str
    repo: str
    size_mb: int
    present: bool


def _entry(tier: str, hardware: Hardware | None):
    return catalog.resolve(tier, hardware or detect_hardware())


def _cached_path(repo: str) -> Path | None:
    """Local path if the snapshot is already complete, else None.

    `local_files_only=True` is the only honest way to ask: it consults the
    cache's own completion bookkeeping rather than guessing from a directory
    listing, so a half-finished fetch does not read as present.
    """
    try:
        return Path(snapshot_download(repo, local_files_only=True))
    except (LocalEntryNotFoundError, OSError, ValueError):
        return None


def is_downloaded(tier: str, hardware: Hardware | None = None) -> bool:
    return _cached_path(_entry(tier, hardware).repo) is not None


def ensure_model(
    tier: str,
    hardware: Hardware | None = None,
    on_progress: Callable[[Progress], None] | None = None,
) -> Path:
    """Return a local path for this tier's model, fetching it if necessary."""
    entry = _entry(tier, hardware)
    report = on_progress or (lambda _p: None)

    cached = _cached_path(entry.repo)
    if cached is not None:
        report(Progress(entry.repo, "already downloaded", done=True))
        return cached

    report(Progress(entry.repo, f"downloading {entry.label} ({entry.size_mb} MB)"))
    try:
        path = Path(snapshot_download(entry.repo))
    except Exception as exc:
        # Nothing to clean up by hand: huggingface_hub keeps partial transfers
        # in .incomplete files that never register as a finished snapshot, so
        # is_downloaded() keeps saying False until a fetch actually completes.
        raise ModelDownloadError(f"could not download {entry.repo}: {exc}") from exc

    report(Progress(entry.repo, "done", done=True))
    return path


def disk_usage(hardware: Hardware | None = None) -> list[TierUsage]:
    """What each tier would cost, and which are already here. Drives `utter models list`."""
    hardware = hardware or detect_hardware()
    return [
        TierUsage(
            tier=entry.tier,
            label=entry.label,
            repo=entry.repo,
            size_mb=entry.size_mb,
            present=_cached_path(entry.repo) is not None,
        )
        for entry in catalog.all_tiers(hardware)
    ]
