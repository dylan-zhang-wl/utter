"""What machine are we on? The catalog needs this before it can pick a format.

Three kinds, because that is how many distinct runtime stories there are:

  apple_silicon  MLX on the Metal GPU
  cuda           CTranslate2 on an NVIDIA card
  cpu            CTranslate2 int8, or whisper.cpp — works everywhere

Detection never raises. A wrong answer of "cpu" costs speed; an exception costs
the user the whole application, on a machine the author cannot debug.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

log = logging.getLogger(__name__)

HardwareKind = Literal["apple_silicon", "cuda", "cpu"]


@dataclass(frozen=True)
class Hardware:
    kind: HardwareKind
    ram_gb: int  # 0 means "could not tell", never a guess

    @property
    def label(self) -> str:
        """One human-readable line for `utter doctor`."""
        names = {
            "apple_silicon": "Apple Silicon (Metal GPU)",
            "cuda": "NVIDIA GPU (CUDA)",
            "cpu": "CPU only",
        }
        name = names[self.kind]
        return f"{name}, {self.ram_gb} GB RAM" if self.ram_gb else name


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _has_nvidia() -> bool:
    """Look for nvidia-smi on PATH.

    Deliberately not `import torch` / `import pycuda` and asking them: importing
    a CUDA stack is slow, and on a machine with a half-installed driver the
    import itself can hang or abort the process. A PATH lookup cannot.
    """
    return shutil.which("nvidia-smi") is not None


def _total_ram_gb() -> int:
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return 0  # Windows, or a platform without sysconf
    return round(total / 1024**3)


@lru_cache(maxsize=1)
def detect() -> Hardware:
    """Identify the machine. Cached: the probes shell out and never change."""
    try:
        if _is_apple_silicon():
            # No Apple Silicon Mac has CUDA, so do not even ask.
            return Hardware(kind="apple_silicon", ram_gb=_total_ram_gb())
        if _has_nvidia():
            return Hardware(kind="cuda", ram_gb=_total_ram_gb())
        return Hardware(kind="cpu", ram_gb=_total_ram_gb())
    except Exception:
        log.warning("hardware detection failed, assuming CPU", exc_info=True)
        return Hardware(kind="cpu", ram_gb=0)
