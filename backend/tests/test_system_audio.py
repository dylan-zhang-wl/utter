"""P3 task 7 — system audio, for online meetings.

The Core Audio parts need real hardware and are verified by hand (the results
are in the module docstring). What is tested here is the part that is pure
arithmetic and would otherwise be wrong in a way nobody notices: the conversion
from the device's 48 kHz stereo to the 16 kHz mono the pipeline expects.

That conversion matters more than it looks. PortAudio offered to do it and
returned exact zeros while reporting success, so this code is the only thing
standing between a meeting and a silent transcript.
"""

import numpy as np
import pytest

from backend.system_audio import (
    SystemAudioSource,
    _lowpass,
    _Resampler,
)
from backend.vad import SAMPLE_RATE


def tone(hz, seconds, rate, channels=2):
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    mono = np.sin(2 * np.pi * hz * t).astype(np.float32)
    return np.stack([mono] * channels, axis=1) if channels > 1 else mono


# --- the filter -----------------------------------------------------------------


def test_the_lowpass_passes_dc_untouched():
    assert _lowpass(7200, 48000).sum() == pytest.approx(1.0, abs=1e-6)


def test_the_lowpass_actually_removes_what_would_alias():
    """Anything above 8 kHz folds back into speech if it survives. A 12 kHz
    tone would land at 4 kHz — right in the middle of the voice."""
    h = _lowpass(7200, 48000)
    spectrum = np.abs(np.fft.rfft(h, 4096))
    freqs = np.fft.rfftfreq(4096, 1 / 48000)
    passband = spectrum[freqs < 3000].mean()
    at_12k = spectrum[np.argmin(np.abs(freqs - 12000))]

    assert at_12k < passband / 100, "12kHz 没被压下去，会混叠成 4kHz"


# --- the resampler --------------------------------------------------------------


def test_48k_stereo_becomes_16k_mono():
    out = _Resampler(48000)(tone(440, 1.0, 48000))

    assert out.ndim == 1, "应该是单声道"
    assert abs(len(out) - SAMPLE_RATE) < 100, f"长度不对：{len(out)}"


def test_a_speech_frequency_survives_intact():
    """440 Hz is well inside the passband; if it comes out attenuated the
    filter is eating the voice."""
    out = _Resampler(48000)(tone(440, 0.5, 48000))
    assert float(np.abs(out).max()) > 0.8


def test_the_rate_does_not_drift_over_a_long_meeting():
    """44100 does not divide 16000. Rounding each chunk independently loses a
    sample here and there, and ninety minutes of that is audible drift."""
    r = _Resampler(44100)
    total = sum(len(r(tone(440, 0.1, 44100))) for _ in range(600))   # 60 seconds

    assert abs(total - SAMPLE_RATE * 60) < SAMPLE_RATE * 0.01, \
        f"一分钟差了 {abs(total - SAMPLE_RATE * 60)} 个采样"


def test_chunk_boundaries_do_not_click():
    """Filtering each chunk independently puts a discontinuity at every
    boundary — forty a second, which a VAD hears as speech."""
    signal = tone(440, 1.0, 48000)
    whole = _Resampler(48000)(signal)

    piecewise = _Resampler(48000)
    pieces = np.concatenate([piecewise(signal[i:i + 4800])
                             for i in range(0, len(signal), 4800)])

    n = min(len(whole), len(piecewise_len := pieces))
    assert np.abs(whole[100:n - 100] - pieces[100:n - 100]).max() < 0.05, \
        "分块处理和整段处理结果不一致，说明滤波器状态没接上"


def test_mono_input_is_accepted_too():
    out = _Resampler(48000)(tone(440, 0.2, 48000, channels=1))
    assert out.ndim == 1 and len(out) > 0


def test_silence_in_silence_out():
    out = _Resampler(48000)(np.zeros((4800, 2), dtype=np.float32))
    assert float(np.abs(out).max()) == 0.0


# --- the source's contract ------------------------------------------------------


def test_it_reports_whether_this_machine_can_do_it_at_all():
    ok, why = SystemAudioSource.available()
    assert isinstance(ok, bool)
    assert ok or why, "不可用时必须说明原因"


def test_it_looks_like_a_microphone_source():
    """The pipeline must not be able to tell which one it is holding — that is
    the AudioSource slot in design §4."""
    from backend.audio_source import MicSource

    src = SystemAudioSource()
    for attribute in ("start", "stop", "chunks", "device_name",
                      "stopped_reason", "dropped", "overflows", "first_chunk_at"):
        assert hasattr(src, attribute), f"缺少 {attribute}"
        assert hasattr(MicSource(), attribute)


def test_chunks_never_blocks_when_there_is_nothing():
    """Same contract as MicSource: the caller owns the loop."""
    assert list(SystemAudioSource().chunks()) == []


def test_asking_for_an_app_that_is_not_making_sound_says_so():
    src = SystemAudioSource(pids=[999999])
    with pytest.raises(Exception) as caught:
        src.start()
    assert "没有在放声音" in str(caught.value) or "权限" in str(caught.value)
