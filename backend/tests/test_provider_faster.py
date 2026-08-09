"""P1 Task 7 — FasterWhisperProvider (CTranslate2: CPU and CUDA).

Not deferred. The author owns a non-Apple-Silicon machine and intends to share
this with colleagues on Windows, so the second backend has to exist from the
start or the provider abstraction is theatre (design §5.1, 铁律 6).

faster_whisper.transcribe defaults `temperature` to [0.0, 0.2, ... 1.0], the
same retry ladder mlx_whisper has. 铁律 1 applies identically here.
"""

import sys
import types

import numpy as np
import pytest

from backend.hardware import Hardware
from backend.providers import faster as faster_provider


class FakeSegment:
    def __init__(self, text):
        self.text = text


@pytest.fixture
def fake_fw(monkeypatch):
    """Stub `faster_whisper`, recording construction and transcription kwargs."""
    record = {"init": [], "calls": []}

    class FakeWhisperModel:
        def __init__(self, model_size_or_path, **kwargs):
            record["init"].append({"model": model_size_or_path, **kwargs})

        def transcribe(self, audio, **kwargs):
            record["calls"].append({"audio": audio, **kwargs})
            segments = iter(
                [FakeSegment(" Translation is"), FakeSegment(" a form of rewriting.")]
            )
            return segments, types.SimpleNamespace(language="en")

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return record


def on_hardware(monkeypatch, kind):
    monkeypatch.setattr(
        faster_provider, "detect_hardware", lambda: Hardware(kind=kind, ram_gb=16)
    )


def audio(seconds=3):
    return np.zeros(16000 * seconds, dtype=np.float32)


# --- 铁律 1 ------------------------------------------------------------------


def test_temperature_is_pinned_to_zero(monkeypatch, fake_fw):
    on_hardware(monkeypatch, "cpu")
    faster_provider.FasterWhisperProvider().transcribe(audio())

    assert fake_fw["calls"][0]["temperature"] == 0.0


def test_temperature_is_a_scalar_not_the_default_ladder(monkeypatch, fake_fw):
    on_hardware(monkeypatch, "cpu")
    faster_provider.FasterWhisperProvider().transcribe(audio())

    assert not isinstance(fake_fw["calls"][0]["temperature"], (tuple, list))


# --- device selection --------------------------------------------------------


def test_cuda_machine_uses_cuda_and_float16(monkeypatch, fake_fw):
    on_hardware(monkeypatch, "cuda")
    faster_provider.FasterWhisperProvider().transcribe(audio())

    init = fake_fw["init"][0]
    assert init["device"] == "cuda"
    assert init["compute_type"] == "float16"


def test_cpu_machine_uses_cpu_and_int8(monkeypatch, fake_fw):
    """int8 on CPU is v1's hard-won setting — float32 on CPU is unusably slow."""
    on_hardware(monkeypatch, "cpu")
    faster_provider.FasterWhisperProvider().transcribe(audio())

    init = fake_fw["init"][0]
    assert init["device"] == "cpu"
    assert init["compute_type"] == "int8"


def test_uses_the_catalog_repo_for_the_hardware(monkeypatch, fake_fw):
    on_hardware(monkeypatch, "cpu")
    faster_provider.FasterWhisperProvider(tier="balanced").transcribe(audio())

    assert fake_fw["init"][0]["model"] == "dropbox-dash/faster-whisper-large-v3-turbo"


def test_never_resolves_to_an_mlx_repo(monkeypatch, fake_fw):
    """铁律 6 from the other side: CTranslate2 cannot load MLX weights."""
    for kind in ("cpu", "cuda"):
        on_hardware(monkeypatch, kind)
        fake_fw["init"].clear()
        faster_provider.FasterWhisperProvider().transcribe(audio())
        assert "mlx" not in fake_fw["init"][0]["model"]


# --- availability ------------------------------------------------------------


def test_unavailable_when_the_package_is_missing(monkeypatch):
    on_hardware(monkeypatch, "cpu")
    monkeypatch.setitem(sys.modules, "faster_whisper", None)

    available, reason = faster_provider.FasterWhisperProvider().is_available()
    assert available is False
    assert "faster" in reason.lower()


def test_available_with_the_package(monkeypatch, fake_fw):
    on_hardware(monkeypatch, "cpu")
    available, reason = faster_provider.FasterWhisperProvider().is_available()

    assert available is True
    assert reason == ""


def test_available_on_apple_silicon_too(monkeypatch, fake_fw):
    """CTranslate2 runs on Apple Silicon CPUs. It loses to MLX on preference,
    not on availability — and it is the fallback when MLX is broken."""
    on_hardware(monkeypatch, "apple_silicon")
    available, _ = faster_provider.FasterWhisperProvider().is_available()

    assert available is True


def test_faster_whisper_is_imported_lazily():
    import ast

    tree = ast.parse(open(faster_provider.__file__).read())
    top_level = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "faster_whisper" not in top_level


# --- behaviour ---------------------------------------------------------------


def test_joins_segments_in_order(monkeypatch, fake_fw):
    on_hardware(monkeypatch, "cpu")
    result = faster_provider.FasterWhisperProvider().transcribe(audio())

    assert result == "Translation is a form of rewriting."


def test_model_is_loaded_once_and_reused(monkeypatch, fake_fw):
    """Reconstructing WhisperModel per utterance would reload weights from disk
    every clause — seconds of latency in a pipeline budgeted at one."""
    on_hardware(monkeypatch, "cpu")
    provider = faster_provider.FasterWhisperProvider()
    provider.transcribe(audio())
    provider.transcribe(audio())
    provider.transcribe(audio())

    assert len(fake_fw["init"]) == 1


def test_is_available_does_not_load_the_model(monkeypatch, fake_fw):
    """`utter doctor` probes every provider. Probing must not trigger a
    multi-gigabyte download."""
    on_hardware(monkeypatch, "cpu")
    faster_provider.FasterWhisperProvider().is_available()

    assert fake_fw["init"] == []


def test_passes_language_and_prompt(monkeypatch, fake_fw):
    on_hardware(monkeypatch, "cpu")
    faster_provider.FasterWhisperProvider().transcribe(
        audio(), language="en", initial_prompt="Venuti"
    )

    call = fake_fw["calls"][0]
    assert call["language"] == "en"
    assert call["initial_prompt"] == "Venuti"


def test_does_not_double_up_vad(monkeypatch, fake_fw):
    """Utter segments upstream (design §5.4). Letting faster-whisper run its own
    VAD as well would re-cut clauses the segmenter already decided on."""
    on_hardware(monkeypatch, "cpu")
    faster_provider.FasterWhisperProvider().transcribe(audio())

    assert fake_fw["calls"][0].get("vad_filter") is False


def test_empty_audio_returns_empty_without_loading_a_model(monkeypatch, fake_fw):
    on_hardware(monkeypatch, "cpu")
    result = faster_provider.FasterWhisperProvider().transcribe(
        np.zeros(0, dtype=np.float32)
    )

    assert result == ""
    assert fake_fw["init"] == []


def test_satisfies_the_protocol(monkeypatch, fake_fw):
    from backend.providers.stt import SttProvider

    on_hardware(monkeypatch, "cpu")
    assert isinstance(faster_provider.FasterWhisperProvider(), SttProvider)


def test_identity_fields():
    provider = faster_provider.FasterWhisperProvider()
    assert provider.id == "faster"
    assert provider.display_name
