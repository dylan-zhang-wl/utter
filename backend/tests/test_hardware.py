"""P1 Task 3 — hardware detection.

The catalog cannot choose a model format until it knows what machine it is on:
MLX needs Apple Silicon, CTranslate2 covers CPU and CUDA. Getting this wrong
does not degrade gracefully — it produces an import error at transcription time
on someone else's laptop.

Probes are patched in every logic test. One test runs the real thing and only
asserts it returns something sane, because the answer depends on the machine.
"""

import pytest

from backend import hardware


@pytest.fixture(autouse=True)
def _clear_cache():
    hardware.detect.cache_clear()
    yield
    hardware.detect.cache_clear()


def _probes(monkeypatch, *, apple=False, nvidia=False, ram=16):
    monkeypatch.setattr(hardware, "_is_apple_silicon", lambda: apple)
    monkeypatch.setattr(hardware, "_has_nvidia", lambda: nvidia)
    monkeypatch.setattr(hardware, "_total_ram_gb", lambda: ram)


def test_apple_silicon(monkeypatch):
    _probes(monkeypatch, apple=True, ram=16)
    hw = hardware.detect()
    assert hw.kind == "apple_silicon"
    assert hw.ram_gb == 16


def test_intel_mac_is_cpu(monkeypatch):
    _probes(monkeypatch, apple=False, nvidia=False)
    assert hardware.detect().kind == "cpu"


def test_nvidia_machine_is_cuda(monkeypatch):
    _probes(monkeypatch, apple=False, nvidia=True)
    assert hardware.detect().kind == "cuda"


def test_apple_silicon_wins_over_a_stray_nvidia_probe(monkeypatch):
    """No Apple Silicon Mac has CUDA. If both fire, the nvidia probe is wrong."""
    _probes(monkeypatch, apple=True, nvidia=True)
    assert hardware.detect().kind == "apple_silicon"


def test_failed_probe_falls_back_to_cpu(monkeypatch):
    """Never raise. CPU works everywhere, so it is the safe wrong answer."""
    def boom():
        raise OSError("no idea what this machine is")

    monkeypatch.setattr(hardware, "_is_apple_silicon", boom)
    monkeypatch.setattr(hardware, "_has_nvidia", boom)
    monkeypatch.setattr(hardware, "_total_ram_gb", boom)

    hw = hardware.detect()
    assert hw.kind == "cpu"
    assert hw.ram_gb == 0, "unknown RAM reports 0, not a guess"


def test_result_is_cached(monkeypatch):
    calls = []

    def counting():
        calls.append(1)
        return True

    monkeypatch.setattr(hardware, "_is_apple_silicon", counting)
    monkeypatch.setattr(hardware, "_has_nvidia", lambda: False)
    monkeypatch.setattr(hardware, "_total_ram_gb", lambda: 16)

    hardware.detect()
    hardware.detect()
    hardware.detect()
    assert len(calls) == 1, "probes shell out; they must run once per process"


def test_label_is_human_readable(monkeypatch):
    """`utter doctor` prints this, so it must read as prose, not as an enum."""
    _probes(monkeypatch, apple=True, ram=16)
    label = hardware.detect().label
    assert "Apple Silicon" in label
    assert "16" in label


def test_label_omits_ram_when_unknown(monkeypatch):
    _probes(monkeypatch, apple=False, ram=0)
    assert "0 GB" not in hardware.detect().label


def test_hardware_is_hashable(monkeypatch):
    """The catalog uses it as a dict key in places."""
    _probes(monkeypatch, apple=True)
    assert {hardware.detect(): "ok"}


def test_real_machine_reports_something_valid():
    hw = hardware.detect()
    assert hw.kind in {"apple_silicon", "cuda", "cpu"}
    assert hw.ram_gb >= 0


def test_nvidia_probe_is_false_without_the_binary(monkeypatch):
    """Probe by PATH lookup, not by importing CUDA — importing is slow and can
    itself fail on a machine with a broken driver install."""
    monkeypatch.setattr(hardware.shutil, "which", lambda _: None)
    assert hardware._has_nvidia() is False


def test_nvidia_probe_is_true_with_the_binary(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    assert hardware._has_nvidia() is True
