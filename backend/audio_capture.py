import numpy as np
from typing import Callable

try:
    import pyaudio
    _PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    _PYAUDIO_AVAILABLE = False


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1600  # 100ms at 16kHz


def _get_pyaudio_format():
    """Return pyaudio.paFloat32 if pyaudio is available."""
    if _PYAUDIO_AVAILABLE:
        return pyaudio.paFloat32
    return None


FORMAT = _get_pyaudio_format()


class AudioCaptureManager:
    def __init__(self):
        self.current_source: str = "microphone"
        self.on_audio_chunk: Callable[[np.ndarray], None] | None = None
        self._stream = None
        self._pa = None
        self._running = False

    def list_devices(self) -> list[dict]:
        if not _PYAUDIO_AVAILABLE:
            return []
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                })
        pa.terminate()
        return devices

    def switch_source(self, source: str):
        was_running = self._running
        if was_running:
            self.stop()
        self.current_source = source
        if was_running:
            self.start()

    def start(self, device_index: int | None = None):
        if not _PYAUDIO_AVAILABLE:
            raise RuntimeError(
                "pyaudio is not installed. Install portaudio and pyaudio: "
                "brew install portaudio && pip install pyaudio"
            )
        self._pa = pyaudio.PyAudio()
        self._running = True
        self._stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._pyaudio_callback,
        )
        self._stream.start_stream()

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        audio = np.frombuffer(in_data, dtype=np.float32)
        self._handle_audio_data(audio)
        if _PYAUDIO_AVAILABLE:
            return (None, pyaudio.paContinue)
        return (None, 0)

    def _handle_audio_data(self, audio: np.ndarray):
        if self.on_audio_chunk:
            self.on_audio_chunk(audio.astype(np.float32))
