"""Architecture guards for 铁律 5 and 铁律 6.

These are regression tests, not feature tests. They exist because both rules were
violated within an hour of P1 starting:

  * 铁律 5 — `uv pip install -r requirements.txt` pulled torch 2.13.0 (476M),
    because mlx-whisper falsely declares it. Fixed by requirements-overrides.txt.
  * the documented manual check (`.../bin/pip list | grep torch`) FALSE-PASSED,
    because uv venvs ship without pip. A test cannot false-pass that way.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
PROVIDERS = BACKEND / "providers"

# Concrete STT backends. Importing any of these outside backend/providers/ makes
# the module unimportable on hardware that lacks them — which is exactly what
# 铁律 6 protects against, since the author owns a non-Apple-Silicon machine.
CONCRETE_BACKENDS = {"mlx_whisper", "faster_whisper", "ctranslate2", "whisper"}

# Empty, and meant to stay that way.
#
# This held transcriber.py, v1's faster-whisper wrapper, grandfathered until
# P3 could rewire main.py. main.py and its whole chain went on 2026-08-10, so
# 铁律 6 now holds across every file with no exceptions: outside
# backend/providers/, nothing names a concrete speech backend.
#
# Adding an entry back requires a reason in the commit message. Removing one
# must never be how the suite is made to pass.
LEGACY_EXEMPT: set[str] = set()


def _source_files():
    for path in sorted(BACKEND.rglob("*.py")):
        if PROVIDERS in path.parents or "tests" in path.parts:
            continue
        yield path


def test_torch_is_not_installed():
    """铁律 5. VAD runs on silero ONNX weights via onnxruntime (~2M), not torch."""
    spec = importlib.util.find_spec("torch")
    assert spec is None, (
        f"torch is installed at {spec.origin if spec else '?'}. "
        "Install with --overrides requirements-overrides.txt; see that file for why."
    )


def test_mlx_whisper_does_not_need_torch():
    """The override rests on this claim, so assert it rather than trusting it.

    Skips off Apple Silicon, where mlx-whisper is correctly absent.
    """
    if importlib.util.find_spec("mlx_whisper") is None:
        pytest.skip("mlx_whisper not installed (expected off Apple Silicon)")

    import sys

    import mlx_whisper  # noqa: F401

    assert "torch" not in sys.modules, (
        "importing mlx_whisper pulled in torch — the premise of "
        "requirements-overrides.txt no longer holds, re-verify it"
    )


@pytest.mark.parametrize("path", list(_source_files()), ids=lambda p: p.name)
def test_no_concrete_backend_outside_providers(path):
    """铁律 6. Only backend/providers/ may name a concrete STT backend."""
    tree = ast.parse(path.read_text(), filename=str(path))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    offenders = imported & CONCRETE_BACKENDS
    if offenders and path.name in LEGACY_EXEMPT:
        pytest.xfail(f"{path.name} is grandfathered v1 code, removed in P3")

    assert not offenders, (
        f"{path.relative_to(BACKEND.parent)} imports {sorted(offenders)}. "
        "Concrete backends belong behind the provider interface (铁律 6)."
    )
