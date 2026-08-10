"""P2a Task 1 — the microphone.

sounddevice is patched throughout. Opening a real stream triggers the macOS
microphone permission dialog, which has no place in a test run.

v1's audio_capture.py is not reused: it is built on pyaudio, which P1 did not
adopt, and its ImportError fallback means it degrades to silently doing nothing
— the exact failure mode this project refuses.
"""

import types

import numpy as np
import pytest

from backend import audio_source


class FakeStream:
    """Stands in for sounddevice.InputStream, driving the callback by hand."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    # -- test helpers --

    def deliver(self, frames: np.ndarray, status=None):
        self.kwargs["callback"](frames, len(frames), None, status or _NoStatus())


class _NoStatus:
    def __bool__(self):
        return False

    def __str__(self):
        return ""


class _Overflow:
    input_overflow = True

    def __bool__(self):
        return True

    def __str__(self):
        return "input overflow"


@pytest.fixture(autouse=True)
def fake_sd(monkeypatch):
    FakeStream.instances.clear()

    devices = [
        {"name": "Redmi 电脑音箱", "max_input_channels": 2, "max_output_channels": 2,
         "default_samplerate": 48000},
        {"name": "Speakers only", "max_input_channels": 0, "max_output_channels": 2,
         "default_samplerate": 48000},
        {"name": "WeMeet Audio Device", "max_input_channels": 2, "max_output_channels": 0,
         "default_samplerate": 48000},
    ]

    module = types.SimpleNamespace(
        InputStream=FakeStream,
        query_devices=lambda index=None: devices[index] if index is not None else devices,
        default=types.SimpleNamespace(device=(0, 1)),
        check_input_settings=lambda **kwargs: None,
        PortAudioError=RuntimeError,
    )
    monkeypatch.setattr(audio_source, "sd", module)
    return module


def mono(seconds=0.1, value=0.2):
    return np.full((int(16000 * seconds), 1), value, dtype=np.float32)


# --- device listing ----------------------------------------------------------


def test_lists_only_input_capable_devices():
    names = [d.name for d in audio_source.list_devices()]
    assert "Speakers only" not in names
    assert "Redmi 电脑音箱" in names


def test_marks_the_default_device():
    default = [d for d in audio_source.list_devices() if d.is_default]
    assert len(default) == 1
    assert default[0].name == "Redmi 电脑音箱"


def test_device_carries_its_index():
    devices = {d.name: d.index for d in audio_source.list_devices()}
    assert devices["WeMeet Audio Device"] == 2


# --- stream configuration ----------------------------------------------------


def test_requests_16k_mono_float32():
    with audio_source.MicSource():
        stream = FakeStream.instances[0]

    assert stream.kwargs["samplerate"] == 16000
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "float32"


def test_honours_a_configured_device():
    """The author's default input is a Bluetooth speaker with no built-in mic
    listed, so choosing a device has to work."""
    with audio_source.MicSource(device_index=2):
        assert FakeStream.instances[0].kwargs["device"] == 2


def test_default_device_is_left_to_the_system():
    with audio_source.MicSource():
        assert FakeStream.instances[0].kwargs["device"] is None


def test_chunk_size_is_100ms_by_default():
    with audio_source.MicSource():
        assert FakeStream.instances[0].kwargs["blocksize"] == 1600


# --- chunk delivery ----------------------------------------------------------


def test_yields_the_delivered_audio():
    with audio_source.MicSource() as source:
        FakeStream.instances[0].deliver(mono())
        chunk = next(source.chunks())

    assert chunk.shape == (1600,)
    assert chunk.dtype == np.float32


def test_downmixes_stereo():
    with audio_source.MicSource() as source:
        frames = np.stack(
            [np.full(1600, 0.4, dtype=np.float32), np.full(1600, 0.0, dtype=np.float32)],
            axis=1,
        )
        FakeStream.instances[0].deliver(frames)
        chunk = next(source.chunks())

    assert chunk.ndim == 1
    assert np.allclose(chunk, 0.2)


def test_chunks_arrive_in_order():
    with audio_source.MicSource() as source:
        for value in (0.1, 0.2, 0.3):
            FakeStream.instances[0].deliver(mono(value=value))

        got = [float(next(source.chunks())[0]) for _ in range(3)]

    assert got == pytest.approx([0.1, 0.2, 0.3])


# --- the bounded queue -------------------------------------------------------


def test_queue_is_bounded_and_drops_the_oldest():
    """An unbounded queue turns a transient stall into unbounded memory and a
    lag the user cannot see. Dropping the oldest keeps latency honest."""
    with audio_source.MicSource(max_chunks=3) as source:
        for value in (0.1, 0.2, 0.3, 0.4, 0.5):
            FakeStream.instances[0].deliver(mono(value=value))

        got = [float(next(source.chunks())[0]) for _ in range(3)]

    assert got == pytest.approx([0.3, 0.4, 0.5])


def test_drops_are_counted():
    with audio_source.MicSource(max_chunks=2) as source:
        for value in (0.1, 0.2, 0.3, 0.4):
            FakeStream.instances[0].deliver(mono(value=value))

        assert source.dropped == 2


def test_no_drops_when_keeping_up():
    with audio_source.MicSource(max_chunks=10) as source:
        FakeStream.instances[0].deliver(mono())
        assert source.dropped == 0


def test_overflow_status_is_recorded_not_raised():
    """PortAudio reporting an overflow must not kill a dictation in progress."""
    with audio_source.MicSource() as source:
        FakeStream.instances[0].deliver(mono(), status=_Overflow())
        assert source.overflows == 1


# --- lifecycle and failure ---------------------------------------------------


def test_context_manager_closes_the_stream():
    with audio_source.MicSource():
        pass
    assert FakeStream.instances[0].closed is True


def test_stream_is_closed_even_on_exception():
    with pytest.raises(ValueError):
        with audio_source.MicSource():
            raise ValueError("something else went wrong")

    assert FakeStream.instances[0].closed is True


def test_unknown_device_raises_a_named_error(monkeypatch, fake_sd):
    def refuse(**kwargs):
        raise fake_sd.PortAudioError("Invalid device")

    monkeypatch.setattr(fake_sd, "check_input_settings", refuse)

    with pytest.raises(audio_source.AudioSourceError) as exc:
        audio_source.MicSource(device_index=99).start()

    assert "99" in str(exc.value)


def test_device_lost_midstream_stops_cleanly():
    """Bluetooth drops. The daemon must see a readable reason, not a traceback
    from inside PortAudio."""
    with audio_source.MicSource() as source:
        FakeStream.instances[0].deliver(mono())
        source.fail("device disconnected")

        chunks = list(source.chunks())

    assert len(chunks) == 1, "audio already captured is still delivered"
    assert source.stopped_reason == "device disconnected"


def test_chunks_ends_rather_than_hanging_after_stop():
    source = audio_source.MicSource()
    source.start()
    FakeStream.instances[0].deliver(mono())
    source.stop()

    assert len(list(source.chunks())) == 1


def test_double_start_is_harmless():
    with audio_source.MicSource() as source:
        source.start()
    assert len(FakeStream.instances) == 1


def test_double_stop_is_harmless():
    source = audio_source.MicSource()
    source.start()
    source.stop()
    source.stop()


def test_stop_without_start_is_harmless():
    audio_source.MicSource().stop()


def test_buffer_holds_a_long_utterance():
    """The queue IS the recording in push-to-talk — nothing drains it until the
    key comes up.

    Sized twice, wrongly both times before this. 100 chunks capped every
    dictation at ten seconds; 350 capped it at 35, a number copied from the VAD
    force-cut ceiling that governs a different mode entirely, and the author hit
    it on their second real attempt. Five minutes of audio is 19 MB — memory was
    never the constraint here.
    """
    seconds = audio_source.MAX_CHUNKS * audio_source.CHUNK_SAMPLES / audio_source.SAMPLE_RATE
    assert seconds >= 300


def test_the_buffer_costs_little_memory():
    """The reason the cap can be generous. If this ever fails, the tradeoff that
    justified a five-minute buffer has changed and needs rethinking."""
    megabytes = audio_source.MAX_CHUNKS * audio_source.CHUNK_SAMPLES * 4 / 1e6
    assert megabytes < 32


def test_nothing_is_dropped_within_the_utterance_ceiling():
    with audio_source.MicSource() as source:
        for _ in range(600):  # 60 seconds at 100ms per chunk
            FakeStream.instances[0].deliver(mono())

        assert source.dropped == 0
