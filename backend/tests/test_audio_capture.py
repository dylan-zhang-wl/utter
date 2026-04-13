import numpy as np
from unittest.mock import patch, MagicMock
from backend.audio_capture import AudioCaptureManager


def test_manager_default_source():
    manager = AudioCaptureManager()
    assert manager.current_source == "microphone"


def test_manager_switch_source():
    manager = AudioCaptureManager()
    manager.switch_source("system")
    assert manager.current_source == "system"
    manager.switch_source("microphone")
    assert manager.current_source == "microphone"


def test_manager_list_devices():
    manager = AudioCaptureManager()
    devices = manager.list_devices()
    assert isinstance(devices, list)


def test_audio_callback_format():
    """Audio chunks should be 16kHz mono float32 numpy arrays."""
    manager = AudioCaptureManager()
    chunks = []
    manager.on_audio_chunk = lambda chunk: chunks.append(chunk)

    # Simulate a raw audio callback
    fake_data = np.zeros(1600, dtype=np.float32)  # 100ms at 16kHz
    manager._handle_audio_data(fake_data)

    assert len(chunks) == 1
    assert chunks[0].dtype == np.float32
    assert len(chunks[0]) == 1600
