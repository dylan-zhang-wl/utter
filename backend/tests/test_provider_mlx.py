"""P1 Task 6 — MlxWhisperProvider.

The temperature test is the point of this file. mlx_whisper.transcribe defaults
`temperature` to (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) — the fallback ladder that
design §3 measured at 8.19s and 10.84s on a 3-second chunk, against 0.96s with
it pinned off. 铁律 1 exists because of those numbers, and this test exists so a
future refactor cannot quietly drop the kwarg.
"""

import sys
import types

import numpy as np
import pytest

from backend.hardware import Hardware
from backend.providers import mlx as mlx_provider


@pytest.fixture
def fake_mlx(monkeypatch):
    """Install a stub `mlx_whisper` that records how it was called."""
    calls = []

    def transcribe(audio, **kwargs):
        calls.append({"audio": audio, **kwargs})
        return {"text": "  Translation is a form of rewriting.  "}

    module = types.ModuleType("mlx_whisper")
    module.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    return calls


def on_apple(monkeypatch, yes=True):
    kind = "apple_silicon" if yes else "cpu"
    monkeypatch.setattr(
        mlx_provider, "detect_hardware", lambda: Hardware(kind=kind, ram_gb=16)
    )


def audio(seconds=3):
    return np.zeros(16000 * seconds, dtype=np.float32)


# --- 铁律 1 ------------------------------------------------------------------


def test_temperature_is_pinned_to_zero(monkeypatch, fake_mlx):
    on_apple(monkeypatch)
    mlx_provider.MlxWhisperProvider().transcribe(audio())

    assert fake_mlx[0]["temperature"] == 0.0


def test_temperature_is_a_scalar_not_the_default_ladder(monkeypatch, fake_mlx):
    """Passing the tuple would satisfy a naive `"temperature" in kwargs` check
    while restoring exactly the behaviour 铁律 1 forbids."""
    on_apple(monkeypatch)
    mlx_provider.MlxWhisperProvider().transcribe(audio())

    assert not isinstance(fake_mlx[0]["temperature"], (tuple, list))


# --- availability ------------------------------------------------------------


def test_unavailable_off_apple_silicon(monkeypatch, fake_mlx):
    on_apple(monkeypatch, yes=False)
    available, reason = mlx_provider.MlxWhisperProvider().is_available()

    assert available is False
    assert "Apple Silicon" in reason


def test_unavailable_when_the_package_is_missing(monkeypatch):
    on_apple(monkeypatch)
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)  # forces ImportError

    available, reason = mlx_provider.MlxWhisperProvider().is_available()
    assert available is False
    assert "mlx" in reason.lower()


def test_available_on_apple_silicon_with_the_package(monkeypatch, fake_mlx):
    on_apple(monkeypatch)
    available, reason = mlx_provider.MlxWhisperProvider().is_available()

    assert available is True
    assert reason == ""


def test_mlx_whisper_is_imported_lazily():
    """铁律 6. The module must import cleanly on a machine with no MLX at all,
    or the registry cannot even load it far enough to report it unavailable."""
    import ast

    tree = ast.parse(open(mlx_provider.__file__).read())
    top_level_imports = {
        alias.name
        for node in tree.body  # module scope only, not nested in functions
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "mlx_whisper" not in top_level_imports, (
        "mlx_whisper must be imported inside a method, not at module scope"
    )


# --- behaviour ---------------------------------------------------------------


def test_returns_stripped_text(monkeypatch, fake_mlx):
    on_apple(monkeypatch)
    result = mlx_provider.MlxWhisperProvider().transcribe(audio())

    assert result == "Translation is a form of rewriting."


def test_passes_language_through(monkeypatch, fake_mlx):
    on_apple(monkeypatch)
    mlx_provider.MlxWhisperProvider().transcribe(audio(), language="en")

    assert fake_mlx[0]["language"] == "en"


def test_passes_initial_prompt_through(monkeypatch, fake_mlx):
    """§4.1g layer 1 — the free half of the vocabulary feature."""
    on_apple(monkeypatch)
    mlx_provider.MlxWhisperProvider().transcribe(audio(), initial_prompt="Venuti, 异化")

    assert fake_mlx[0]["initial_prompt"] == "Venuti, 异化"


def test_omits_initial_prompt_when_empty(monkeypatch, fake_mlx):
    """An empty prompt is not the same as no prompt; passing "" can bias the
    model toward starting mid-sentence."""
    on_apple(monkeypatch)
    mlx_provider.MlxWhisperProvider().transcribe(audio(), initial_prompt="")

    assert fake_mlx[0].get("initial_prompt") is None


def test_uses_the_catalog_repo_for_its_tier(monkeypatch, fake_mlx):
    on_apple(monkeypatch)
    mlx_provider.MlxWhisperProvider(tier="balanced").transcribe(audio())

    assert fake_mlx[0]["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"


def test_tier_selects_a_different_repo(monkeypatch, fake_mlx):
    on_apple(monkeypatch)
    mlx_provider.MlxWhisperProvider(tier="minimal").transcribe(audio())

    assert fake_mlx[0]["path_or_hf_repo"] == "mlx-community/whisper-base-mlx-q4"


def test_empty_audio_returns_empty_without_calling_the_model(monkeypatch, fake_mlx):
    """VAD can hand over a zero-length buffer at a boundary. A model call on it
    costs a second and returns nothing useful."""
    on_apple(monkeypatch)
    result = mlx_provider.MlxWhisperProvider().transcribe(
        np.zeros(0, dtype=np.float32)
    )

    assert result == ""
    assert fake_mlx == []


def test_satisfies_the_protocol(monkeypatch, fake_mlx):
    from backend.providers.stt import SttProvider

    on_apple(monkeypatch)
    assert isinstance(mlx_provider.MlxWhisperProvider(), SttProvider)


def test_identity_fields():
    provider = mlx_provider.MlxWhisperProvider()
    assert provider.id == "mlx"
    assert provider.display_name
