"""P1 Task 12 — the CLI, and the acceptance gate for the whole phase.

The point of P1 is to prove the core works before any GUI exists. `utter doctor`
has to be readable by someone on a machine the author cannot log into, and
`utter transcribe` has to produce sensible utterances in both modes at the
latency design §3 measured.

Model-free tests here; the real-audio latency check is a separate marked test
plus a manual run recorded in docs/benchmarks/.
"""

import io

import numpy as np
import pytest
import soundfile as sf

from backend import cli
from backend.hardware import Hardware


class FakeStt:
    id = "fake"
    display_name = "Fake STT"

    def __init__(self, texts=None):
        self.texts = list(texts or ["Translation is rewriting.", "Every choice leaves a trace."])
        self.n = 0

    def is_available(self):
        return True, ""

    def transcribe(self, audio, language=None, initial_prompt=None):
        text = self.texts[self.n % len(self.texts)]
        self.n += 1
        return text


@pytest.fixture
def wav(tmp_path):
    """Two tone bursts separated by enough silence to split them."""
    sr = 16000
    t = np.arange(sr) / sr
    voiced = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    silence = np.zeros(sr, dtype=np.float32)
    path = tmp_path / "sample.wav"
    sf.write(path, np.concatenate([silence[: sr // 5], voiced, silence, voiced, silence]), sr)
    return path


@pytest.fixture
def loud_vad(monkeypatch):
    """Energy-based VAD, so the CLI tests never load the silero model."""
    import backend.vad as vad_module

    original = vad_module.VadSegmenter.__post_init__

    def patched(self):
        if self.speech_prob is None:
            self.speech_prob = lambda f: 1.0 if float(np.abs(f).mean()) > 0.01 else 0.0

    monkeypatch.setattr(vad_module.VadSegmenter, "__post_init__", patched)
    yield
    monkeypatch.setattr(vad_module.VadSegmenter, "__post_init__", original)


def run(argv, **kwargs):
    out = io.StringIO()
    code = cli.main(argv, stdout=out, **kwargs)
    return code, out.getvalue()


# --- doctor ------------------------------------------------------------------


def test_doctor_reports_hardware(monkeypatch):
    monkeypatch.setattr(
        cli, "detect_hardware", lambda: Hardware(kind="apple_silicon", ram_gb=16)
    )
    code, out = run(["doctor"])

    assert code == 0
    assert "Apple Silicon" in out
    assert "16" in out


def test_doctor_lists_providers_with_reasons(monkeypatch):
    """The reason is the whole value of the command — "unavailable" alone tells
    a remote user nothing they can act on."""
    from backend.providers.stt import ProviderStatus

    monkeypatch.setattr(
        cli,
        "probe_all",
        lambda **_: [
            ProviderStatus("mlx", "Whisper (MLX)", False, "requires Apple Silicon"),
            ProviderStatus("faster", "Whisper (CTranslate2)", True, ""),
        ],
    )
    code, out = run(["doctor"])

    assert "requires Apple Silicon" in out
    assert "faster" in out


def test_doctor_succeeds_even_when_nothing_works(monkeypatch):
    """It is the command you run *because* things are broken."""
    monkeypatch.setattr(cli, "probe_all", lambda **_: [])
    code, out = run(["doctor"])

    assert code == 0
    assert out.strip()


def test_doctor_reports_llm_providers(monkeypatch):
    code, out = run(["doctor"])
    assert "ollama" in out.lower()


def test_doctor_shows_which_models_are_downloaded(monkeypatch):
    code, out = run(["doctor"])
    assert "均衡" in out or "balanced" in out


# --- models ------------------------------------------------------------------


def test_models_lists_four_tiers():
    code, out = run(["models", "list"])

    assert code == 0
    for tier in ("高精度", "均衡", "轻量", "极轻"):
        assert tier in out


def test_models_shows_sizes():
    _, out = run(["models", "list"])
    assert "3084" in out or "3.0" in out


def test_models_marks_the_recommended_tier():
    _, out = run(["models", "list"])
    assert "推荐" in out


def test_models_shows_download_state():
    _, out = run(["models", "list"])
    assert "✓" in out or "—" in out


# --- transcribe --------------------------------------------------------------


def test_transcribe_listen_prints_utterances(wav, loud_vad):
    code, out = run(
        ["transcribe", str(wav), "--mode", "listen"],
        stt=FakeStt(),
        translate=lambda text, context=None: f"[中] {text}",
    )

    assert code == 0
    assert "Translation is rewriting." in out
    assert "[中] Translation is rewriting." in out


def test_transcribe_dictate_prints_polished_text(wav, loud_vad):
    code, out = run(
        ["transcribe", str(wav), "--mode", "dictate"],
        stt=FakeStt(),
        polish=lambda text, context=None: text.upper(),
    )

    assert code == 0
    assert "TRANSLATION IS REWRITING." in out


def test_transcribe_dictate_does_not_translate(wav, loud_vad):
    code, out = run(["transcribe", str(wav), "--mode", "dictate"], stt=FakeStt())
    assert "[中]" not in out


def test_transcribe_timing_reports_per_utterance_seconds(wav, loud_vad):
    """The acceptance gate reads these numbers."""
    code, out = run(["transcribe", str(wav), "--mode", "listen", "--timing"], stt=FakeStt())

    assert code == 0
    assert "s" in out
    assert any(char.isdigit() for char in out)


def test_transcribe_missing_file_fails_cleanly(loud_vad):
    code, out = run(["transcribe", "/nowhere/at/all.wav"], stt=FakeStt())

    assert code != 0
    assert "not found" in out.lower() or "no such" in out.lower()


def test_transcribe_rejects_wrong_sample_rate(tmp_path, loud_vad):
    """Silero and Whisper both want 16 kHz. Failing loudly beats transcribing
    something that sounds like a chipmunk."""
    path = tmp_path / "wrong.wav"
    sf.write(path, np.zeros(44100, dtype=np.float32), 44100)

    code, out = run(["transcribe", str(path)], stt=FakeStt())
    assert code != 0
    assert "16" in out


def test_transcribe_handles_stereo_by_downmixing(tmp_path, loud_vad):
    sr = 16000
    t = np.arange(sr) / sr
    mono = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.concatenate([stereo, np.zeros((sr, 2), dtype=np.float32)]), sr)

    code, out = run(["transcribe", str(path)], stt=FakeStt())
    assert code == 0


def test_transcribe_reports_when_nothing_was_heard(tmp_path, loud_vad):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(16000 * 2, dtype=np.float32), 16000)

    code, out = run(["transcribe", str(path)], stt=FakeStt())
    assert code == 0
    assert "no speech" in out.lower() or "0" in out


# --- shape -------------------------------------------------------------------


def test_no_arguments_prints_help():
    code, out = run([])
    assert "transcribe" in out
    assert "doctor" in out


def test_unknown_command_fails():
    with pytest.raises(SystemExit):
        run(["frobnicate"])
