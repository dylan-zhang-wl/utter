# LiveScribe Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a macOS desktop app that captures audio (mic/system), transcribes speech to text via local Whisper, translates English→Chinese, and displays results in real-time.

**Architecture:** Python FastAPI backend handles audio capture, Whisper transcription, and translation. Tauri v2 + React frontend displays results via WebSocket. Backend runs as a Tauri sidecar process.

**Tech Stack:** Python 3.11+, FastAPI, faster-whisper, Silero VAD, PyAudio, googletrans, OpenAI API, Tauri v2, React, TypeScript

**Design doc:** `docs/plans/2026-04-13-livescribe-design.md`

---

## Phase 1: Python Backend Core

### Task 1: Project Scaffolding & Python Environment

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/__init__.py`
- Create: `backend/config.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_config.py`
- Create: `.gitignore`

**Step 1: Initialize git and create .gitignore**

```bash
cd "/Users/d/Desktop/华工大/cc cowork/live voice transform"
git init
```

Create `.gitignore`:
```
__pycache__/
*.pyc
.venv/
node_modules/
dist/
target/
*.egg-info/
.env
~/LiveScribe/
backend/models/
```

**Step 2: Create Python virtual environment and requirements.txt**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

`backend/requirements.txt`:
```
fastapi==0.115.*
uvicorn==0.34.*
websockets==15.*
faster-whisper==1.1.*
pyaudio==0.2.*
silero-vad==5.*
googletrans==4.0.0rc1
openai==1.*
pydantic==2.*
pydantic-settings==2.*
pytest==8.*
pytest-asyncio==0.25.*
```

```bash
pip install -r backend/requirements.txt
```

**Step 3: Write failing test for config module**

`backend/tests/test_config.py`:
```python
from backend.config import AppConfig


def test_default_config():
    config = AppConfig()
    assert config.whisper_model == "base"
    assert config.audio_source == "microphone"
    assert config.translation_engine == "google"
    assert config.display_mode == "bilingual"
    assert config.openai_api_key is None
    assert config.save_dir.endswith("LiveScribe/sessions")


def test_config_update():
    config = AppConfig()
    config.audio_source = "system"
    assert config.audio_source == "system"
```

**Step 4: Run test to verify it fails**

```bash
cd "/Users/d/Desktop/华工大/cc cowork/live voice transform"
python -m pytest backend/tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.config'`

**Step 5: Implement config module**

`backend/__init__.py`: (empty file)
`backend/tests/__init__.py`: (empty file)

`backend/config.py`:
```python
import os
from pathlib import Path
from pydantic import BaseModel


class AppConfig(BaseModel):
    whisper_model: str = "base"
    audio_source: str = "microphone"  # "microphone" or "system"
    translation_engine: str = "google"  # "google" or "openai"
    display_mode: str = "bilingual"  # "english", "bilingual", "chinese"
    openai_api_key: str | None = None
    save_dir: str = str(Path.home() / "LiveScribe" / "sessions")
    server_host: str = "127.0.0.1"
    server_port: int = 8765
```

**Step 6: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_config.py -v
```
Expected: PASS

**Step 7: Commit**

```bash
git add .gitignore backend/
git commit -m "feat: project scaffolding with config module"
```

---

### Task 2: Audio Capture — Microphone

**Files:**
- Create: `backend/audio_capture.py`
- Create: `backend/tests/test_audio_capture.py`

**Step 1: Write failing test**

`backend/tests/test_audio_capture.py`:
```python
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
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_audio_capture.py -v
```
Expected: FAIL

**Step 3: Implement audio capture module**

`backend/audio_capture.py`:
```python
import numpy as np
import pyaudio
import threading
from typing import Callable


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1600  # 100ms at 16kHz
FORMAT = pyaudio.paFloat32


class AudioCaptureManager:
    def __init__(self):
        self.current_source: str = "microphone"
        self.on_audio_chunk: Callable[[np.ndarray], None] | None = None
        self._stream = None
        self._pa: pyaudio.PyAudio | None = None
        self._running = False

    def list_devices(self) -> list[dict]:
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
        return (None, pyaudio.paContinue)

    def _handle_audio_data(self, audio: np.ndarray):
        if self.on_audio_chunk:
            self.on_audio_chunk(audio.astype(np.float32))
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_audio_capture.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add backend/audio_capture.py backend/tests/test_audio_capture.py
git commit -m "feat: audio capture module with microphone support"
```

---

### Task 3: Transcriber Module (Whisper)

**Files:**
- Create: `backend/transcriber.py`
- Create: `backend/tests/test_transcriber.py`

**Step 1: Write failing test**

`backend/tests/test_transcriber.py`:
```python
import numpy as np
from unittest.mock import patch, MagicMock
from backend.transcriber import Transcriber, TranscriptSegment


def test_transcript_segment_model():
    seg = TranscriptSegment(
        text="Hello world",
        start_time=0.0,
        end_time=1.5,
        is_partial=False,
    )
    assert seg.text == "Hello world"
    assert seg.is_partial is False


def test_transcriber_init():
    with patch("backend.transcriber.WhisperModel") as mock:
        mock.return_value = MagicMock()
        t = Transcriber(model_size="base")
        assert t.model_size == "base"
        mock.assert_called_once()


def test_transcriber_process_audio():
    """Transcriber should accept audio and return segments."""
    with patch("backend.transcriber.WhisperModel") as mock_cls:
        mock_model = MagicMock()
        # Simulate whisper output
        mock_segment = MagicMock()
        mock_segment.text = " Hello there."
        mock_segment.start = 0.0
        mock_segment.end = 1.2
        mock_model.transcribe.return_value = ([mock_segment], MagicMock(language="en"))
        mock_cls.return_value = mock_model

        t = Transcriber(model_size="base")
        audio = np.random.randn(16000).astype(np.float32)  # 1 second
        segments = t.transcribe(audio)

        assert len(segments) == 1
        assert segments[0].text == "Hello there."
        assert segments[0].start_time == 0.0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_transcriber.py -v
```
Expected: FAIL

**Step 3: Implement transcriber**

`backend/transcriber.py`:
```python
import numpy as np
from faster_whisper import WhisperModel
from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    text: str
    start_time: float
    end_time: float
    is_partial: bool = False


class Transcriber:
    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.model_size = model_size
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8",
        )

    def transcribe(self, audio: np.ndarray) -> list[TranscriptSegment]:
        segments_iter, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,
        )
        results = []
        for seg in segments_iter:
            results.append(TranscriptSegment(
                text=seg.text.strip(),
                start_time=seg.start,
                end_time=seg.end,
                is_partial=False,
            ))
        return results
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_transcriber.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add backend/transcriber.py backend/tests/test_transcriber.py
git commit -m "feat: transcriber module with faster-whisper"
```

---

### Task 4: Translator Module

**Files:**
- Create: `backend/translator.py`
- Create: `backend/tests/test_translator.py`

**Step 1: Write failing test**

`backend/tests/test_translator.py`:
```python
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from backend.translator import GoogleTranslator, OpenAITranslator, get_translator


def test_get_translator_google():
    t = get_translator("google")
    assert isinstance(t, GoogleTranslator)


def test_get_translator_openai():
    t = get_translator("openai", api_key="test-key")
    assert isinstance(t, OpenAITranslator)


@pytest.mark.asyncio
async def test_google_translate():
    with patch("backend.translator.googletrans.Translator") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.translate.return_value = MagicMock(text="你好世界")
        mock_cls.return_value = mock_instance

        t = GoogleTranslator()
        result = await t.translate("Hello world")
        assert result == "你好世界"


@pytest.mark.asyncio
async def test_openai_translate():
    with patch("backend.translator.openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="你好世界"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client

        t = OpenAITranslator(api_key="test-key")
        result = await t.translate("Hello world")
        assert result == "你好世界"


@pytest.mark.asyncio
async def test_translate_empty_string():
    t = GoogleTranslator()
    result = await t.translate("")
    assert result == ""
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_translator.py -v
```
Expected: FAIL

**Step 3: Implement translator module**

`backend/translator.py`:
```python
import asyncio
from abc import ABC, abstractmethod
import googletrans
import openai


class BaseTranslator(ABC):
    @abstractmethod
    async def translate(self, text: str, target_lang: str = "zh-cn") -> str:
        pass


class GoogleTranslator(BaseTranslator):
    def __init__(self):
        self._translator = googletrans.Translator()

    async def translate(self, text: str, target_lang: str = "zh-cn") -> str:
        if not text.strip():
            return ""
        result = await asyncio.to_thread(
            self._translator.translate, text, dest=target_lang
        )
        return result.text


class OpenAITranslator(BaseTranslator):
    def __init__(self, api_key: str):
        self._client = openai.OpenAI(api_key=api_key)

    async def translate(self, text: str, target_lang: str = "zh-cn") -> str:
        if not text.strip():
            return ""
        response = await asyncio.to_thread(
            self._client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate the following English text to Chinese. Return only the translation, nothing else."},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content


def get_translator(engine: str, api_key: str | None = None) -> BaseTranslator:
    if engine == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required")
        return OpenAITranslator(api_key=api_key)
    return GoogleTranslator()
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_translator.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add backend/translator.py backend/tests/test_translator.py
git commit -m "feat: translator module with Google and OpenAI engines"
```

---

### Task 5: Session Management & Export

**Files:**
- Create: `backend/session.py`
- Create: `backend/tests/test_session.py`

**Step 1: Write failing test**

`backend/tests/test_session.py`:
```python
import json
import tempfile
from pathlib import Path
from backend.session import Session
from backend.transcriber import TranscriptSegment


def _make_segment(text, zh, start, end):
    return TranscriptSegment(text=text, start_time=start, end_time=end)


def test_session_creation():
    s = Session(audio_source="microphone")
    assert s.audio_source == "microphone"
    assert len(s.entries) == 0


def test_session_add_entry():
    s = Session(audio_source="microphone")
    s.add_entry("Hello world", "你好世界", 0.0, 1.5)
    assert len(s.entries) == 1
    assert s.entries[0]["text"] == "Hello world"
    assert s.entries[0]["translation"] == "你好世界"


def test_session_save_and_load(tmp_path):
    s = Session(audio_source="microphone", save_dir=str(tmp_path))
    s.add_entry("Hello", "你好", 0.0, 1.0)
    s.add_entry("World", "世界", 1.0, 2.0)
    filepath = s.save()

    assert Path(filepath).exists()
    with open(filepath) as f:
        data = json.load(f)
    assert len(data["entries"]) == 2


def test_export_txt(tmp_path):
    s = Session(audio_source="microphone", save_dir=str(tmp_path))
    s.add_entry("Hello", "你好", 0.0, 1.0)
    s.add_entry("World", "世界", 1.0, 2.0)
    content = s.export("txt")
    assert "Hello" in content
    assert "你好" in content


def test_export_srt(tmp_path):
    s = Session(audio_source="microphone", save_dir=str(tmp_path))
    s.add_entry("Hello", "你好", 0.0, 1.0)
    content = s.export("srt")
    assert "00:00:00,000" in content
    assert "Hello" in content


def test_export_markdown(tmp_path):
    s = Session(audio_source="microphone", save_dir=str(tmp_path))
    s.add_entry("Hello", "你好", 0.0, 1.0)
    content = s.export("markdown")
    assert "# " in content  # has a header
    assert "Hello" in content
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_session.py -v
```
Expected: FAIL

**Step 3: Implement session module**

`backend/session.py`:
```python
import json
from datetime import datetime
from pathlib import Path


class Session:
    def __init__(self, audio_source: str, save_dir: str = ""):
        self.audio_source = audio_source
        self.save_dir = save_dir or str(Path.home() / "LiveScribe" / "sessions")
        self.entries: list[dict] = []
        self.started_at = datetime.now().isoformat()

    def add_entry(self, text: str, translation: str, start_time: float, end_time: float):
        self.entries.append({
            "text": text,
            "translation": translation,
            "start_time": start_time,
            "end_time": end_time,
            "timestamp": datetime.now().isoformat(),
        })

    def save(self) -> str:
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = str(Path(self.save_dir) / filename)
        data = {
            "started_at": self.started_at,
            "audio_source": self.audio_source,
            "entries": self.entries,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath

    def export(self, fmt: str) -> str:
        if fmt == "txt":
            return self._export_txt()
        elif fmt == "srt":
            return self._export_srt()
        elif fmt == "markdown":
            return self._export_markdown()
        raise ValueError(f"Unknown format: {fmt}")

    def _export_txt(self) -> str:
        lines = []
        for e in self.entries:
            lines.append(e["text"])
            if e.get("translation"):
                lines.append(e["translation"])
            lines.append("")
        return "\n".join(lines)

    def _export_srt(self) -> str:
        lines = []
        for i, e in enumerate(self.entries, 1):
            start = self._format_srt_time(e["start_time"])
            end = self._format_srt_time(e["end_time"])
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(e["text"])
            if e.get("translation"):
                lines.append(e["translation"])
            lines.append("")
        return "\n".join(lines)

    def _export_markdown(self) -> str:
        lines = [f"# LiveScribe Session — {self.started_at}", ""]
        lines.append(f"**Audio source:** {self.audio_source}\n")
        for e in self.entries:
            ts = self._format_time(e["start_time"])
            lines.append(f"**[{ts}]** {e['text']}")
            if e.get("translation"):
                lines.append(f"> {e['translation']}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _format_time(seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_session.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add backend/session.py backend/tests/test_session.py
git commit -m "feat: session management with save and multi-format export"
```

---

### Task 6: FastAPI WebSocket Server

**Files:**
- Create: `backend/main.py`
- Create: `backend/tests/test_main.py`

**Step 1: Write failing test**

`backend/tests/test_main.py`:
```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.main import app


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_config_get():
    client = TestClient(app)
    resp = client.get("/config")
    assert resp.status_code == 200
    assert "whisper_model" in resp.json()


def test_config_update():
    client = TestClient(app)
    resp = client.patch("/config", json={"display_mode": "english"})
    assert resp.status_code == 200
    assert resp.json()["display_mode"] == "english"


def test_devices_list():
    with patch("backend.main.audio_manager") as mock:
        mock.list_devices.return_value = [{"index": 0, "name": "MacBook Mic", "channels": 1}]
        client = TestClient(app)
        resp = client.get("/devices")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


def test_sessions_list(tmp_path):
    import json
    session_file = tmp_path / "session_20260413_100000.json"
    session_file.write_text(json.dumps({
        "started_at": "2026-04-13T10:00:00",
        "audio_source": "microphone",
        "entries": []
    }))

    with patch("backend.main.config") as mock_config:
        mock_config.save_dir = str(tmp_path)
        client = TestClient(app)
        resp = client.get("/sessions")
        assert resp.status_code == 200
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_main.py -v
```
Expected: FAIL

**Step 3: Implement FastAPI server**

`backend/main.py`:
```python
import asyncio
import json
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.config import AppConfig
from backend.audio_capture import AudioCaptureManager
from backend.transcriber import Transcriber, TranscriptSegment
from backend.translator import get_translator
from backend.session import Session


config = AppConfig()
audio_manager = AudioCaptureManager()
transcriber: Transcriber | None = None
current_session: Session | None = None
connected_clients: set[WebSocket] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcriber
    transcriber = Transcriber(model_size=config.whisper_model)
    yield
    audio_manager.stop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def get_config():
    return config.model_dump()


@app.patch("/config")
def update_config(updates: dict):
    for key, value in updates.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config.model_dump()


@app.get("/devices")
def list_devices():
    return audio_manager.list_devices()


@app.get("/sessions")
def list_sessions():
    save_path = Path(config.save_dir)
    if not save_path.exists():
        return []
    sessions = []
    for f in sorted(save_path.glob("session_*.json"), reverse=True):
        with open(f) as fh:
            data = json.load(fh)
            data["filename"] = f.name
            sessions.append(data)
    return sessions


@app.get("/sessions/{filename}/export")
def export_session(filename: str, fmt: str = "txt"):
    filepath = Path(config.save_dir) / filename
    if not filepath.exists():
        return {"error": "Session not found"}
    with open(filepath) as f:
        data = json.load(f)
    session = Session(audio_source=data["audio_source"])
    session.entries = data["entries"]
    session.started_at = data["started_at"]
    return {"content": session.export(fmt), "format": fmt}


async def broadcast(message: dict):
    for ws in connected_clients.copy():
        try:
            await ws.send_json(message)
        except Exception:
            connected_clients.discard(ws)


# Audio buffer for accumulating chunks before transcription
_audio_buffer: list[np.ndarray] = []
_buffer_lock = asyncio.Lock()
BUFFER_DURATION_SEC = 3  # transcribe every N seconds


async def process_audio_loop():
    global current_session
    while True:
        await asyncio.sleep(BUFFER_DURATION_SEC)
        async with _buffer_lock:
            if not _audio_buffer:
                continue
            audio = np.concatenate(_audio_buffer)
            _audio_buffer.clear()

        if transcriber is None:
            continue

        segments = await asyncio.to_thread(transcriber.transcribe, audio)

        for seg in segments:
            translation = ""
            if config.display_mode in ("bilingual", "chinese"):
                try:
                    translator = get_translator(
                        config.translation_engine,
                        api_key=config.openai_api_key,
                    )
                    translation = await translator.translate(seg.text)
                except Exception:
                    translation = "[translation error]"

            if current_session:
                current_session.add_entry(seg.text, translation, seg.start_time, seg.end_time)

            await broadcast({
                "type": "transcript",
                "text": seg.text,
                "translation": translation,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
            })


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global current_session
    await ws.accept()
    connected_clients.add(ws)
    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")

            if action == "start":
                current_session = Session(
                    audio_source=config.audio_source,
                    save_dir=config.save_dir,
                )

                def on_chunk(chunk):
                    _audio_buffer.append(chunk)

                audio_manager.on_audio_chunk = on_chunk
                audio_manager.switch_source(config.audio_source)
                audio_manager.start(device_index=data.get("device_index"))

                asyncio.create_task(process_audio_loop())
                await ws.send_json({"type": "status", "status": "recording"})

            elif action == "stop":
                audio_manager.stop()
                if current_session and current_session.entries:
                    filepath = current_session.save()
                    await ws.send_json({
                        "type": "status",
                        "status": "stopped",
                        "saved_to": filepath,
                    })
                else:
                    await ws.send_json({"type": "status", "status": "stopped"})
                current_session = None

            elif action == "switch_source":
                config.audio_source = data.get("source", "microphone")
                audio_manager.switch_source(config.audio_source)
                await ws.send_json({
                    "type": "status",
                    "status": "source_switched",
                    "source": config.audio_source,
                })

    except WebSocketDisconnect:
        connected_clients.discard(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.server_host, port=config.server_port)
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_main.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add backend/main.py backend/tests/test_main.py
git commit -m "feat: FastAPI server with WebSocket, REST endpoints, and audio pipeline"
```

---

## Phase 2: Tauri + React Frontend

### Task 7: Tauri + React Project Scaffolding

**Files:**
- Create: `frontend/` (entire Tauri + React project via CLI)

**Step 1: Install Tauri CLI prerequisites**

Check Rust is installed:
```bash
rustc --version || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Step 2: Create Tauri + React project**

```bash
cd "/Users/d/Desktop/华工大/cc cowork/live voice transform"
npm create tauri-app@latest frontend -- --template react-ts
```

When prompted: use default options, package manager = npm.

**Step 3: Install frontend dependencies**

```bash
cd frontend
npm install
npm install zustand         # lightweight state management
```

**Step 4: Verify the scaffolded app builds**

```bash
npm run tauri dev
```
Expected: A Tauri window opens with the default React template. Close it.

**Step 5: Commit**

```bash
cd "/Users/d/Desktop/华工大/cc cowork/live voice transform"
git add frontend/
git commit -m "feat: scaffold Tauri v2 + React frontend"
```

---

### Task 8: WebSocket Client Hook

**Files:**
- Create: `frontend/src/hooks/useWebSocket.ts`
- Create: `frontend/src/store/appStore.ts`

**Step 1: Create app state store**

`frontend/src/store/appStore.ts`:
```typescript
import { create } from "zustand";

export interface TranscriptEntry {
  text: string;
  translation: string;
  startTime: number;
  endTime: number;
  timestamp: number;
}

interface AppState {
  // Connection
  connected: boolean;
  setConnected: (v: boolean) => void;

  // Recording
  recording: boolean;
  setRecording: (v: boolean) => void;

  // Transcripts
  entries: TranscriptEntry[];
  addEntry: (entry: TranscriptEntry) => void;
  clearEntries: () => void;

  // Settings
  audioSource: "microphone" | "system";
  setAudioSource: (s: "microphone" | "system") => void;
  translationEngine: "google" | "openai";
  setTranslationEngine: (e: "google" | "openai") => void;
  displayMode: "english" | "bilingual" | "chinese";
  setDisplayMode: (m: "english" | "bilingual" | "chinese") => void;
}

export const useAppStore = create<AppState>((set) => ({
  connected: false,
  setConnected: (v) => set({ connected: v }),

  recording: false,
  setRecording: (v) => set({ recording: v }),

  entries: [],
  addEntry: (entry) => set((s) => ({ entries: [...s.entries, entry] })),
  clearEntries: () => set({ entries: [] }),

  audioSource: "microphone",
  setAudioSource: (audioSource) => set({ audioSource }),
  translationEngine: "google",
  setTranslationEngine: (translationEngine) => set({ translationEngine }),
  displayMode: "bilingual",
  setDisplayMode: (displayMode) => set({ displayMode }),
}));
```

**Step 2: Create WebSocket hook**

`frontend/src/hooks/useWebSocket.ts`:
```typescript
import { useEffect, useRef, useCallback } from "react";
import { useAppStore } from "../store/appStore";

const WS_URL = "ws://127.0.0.1:8765/ws";

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const { setConnected, setRecording, addEntry } = useAppStore();

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000); // auto-reconnect
      };
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "transcript") {
          addEntry({
            text: data.text,
            translation: data.translation,
            startTime: data.start_time,
            endTime: data.end_time,
            timestamp: Date.now(),
          });
        } else if (data.type === "status") {
          if (data.status === "recording") setRecording(true);
          if (data.status === "stopped") setRecording(false);
        }
      };

      wsRef.current = ws;
    };

    connect();
    return () => wsRef.current?.close();
  }, []);

  const send = useCallback((action: string, extra?: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action, ...extra }));
    }
  }, []);

  return { send };
}
```

**Step 3: Commit**

```bash
cd "/Users/d/Desktop/华工大/cc cowork/live voice transform"
git add frontend/src/hooks/ frontend/src/store/
git commit -m "feat: WebSocket hook and Zustand state store"
```

---

### Task 9: Transcript Display Component

**Files:**
- Create: `frontend/src/components/TranscriptView.tsx`
- Create: `frontend/src/components/TranscriptView.css`

**Step 1: Implement TranscriptView**

`frontend/src/components/TranscriptView.tsx`:
```tsx
import { useEffect, useRef } from "react";
import { useAppStore } from "../store/appStore";
import "./TranscriptView.css";

export function TranscriptView() {
  const { entries, displayMode } = useAppStore();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries]);

  return (
    <div className="transcript-view">
      {entries.length === 0 && (
        <div className="transcript-empty">
          Press Start to begin transcription...
        </div>
      )}
      {entries.map((entry, i) => (
        <div key={i} className="transcript-entry">
          {displayMode !== "chinese" && (
            <p className="transcript-text">{entry.text}</p>
          )}
          {displayMode !== "english" && entry.translation && (
            <p className="transcript-translation">{entry.translation}</p>
          )}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
```

`frontend/src/components/TranscriptView.css`:
```css
.transcript-view {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.transcript-empty {
  color: #888;
  text-align: center;
  margin-top: 40%;
  font-size: 15px;
}

.transcript-entry {
  margin-bottom: 12px;
  padding: 10px 14px;
  border-radius: 8px;
  background: #f8f9fa;
  animation: fadeIn 0.3s ease;
}

.transcript-text {
  margin: 0;
  font-size: 15px;
  color: #1a1a1a;
  line-height: 1.5;
}

.transcript-translation {
  margin: 4px 0 0;
  font-size: 14px;
  color: #4a6fa5;
  line-height: 1.5;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}
```

**Step 2: Commit**

```bash
git add frontend/src/components/
git commit -m "feat: TranscriptView component with auto-scroll"
```

---

### Task 10: Control Bar Component

**Files:**
- Create: `frontend/src/components/ControlBar.tsx`
- Create: `frontend/src/components/ControlBar.css`

**Step 1: Implement ControlBar**

`frontend/src/components/ControlBar.tsx`:
```tsx
import { useAppStore } from "../store/appStore";
import { useWebSocket } from "../hooks/useWebSocket";
import "./ControlBar.css";

export function ControlBar() {
  const {
    recording, connected,
    audioSource, setAudioSource,
    translationEngine, setTranslationEngine,
    displayMode, setDisplayMode,
  } = useAppStore();
  const { send } = useWebSocket();

  const toggleRecording = () => {
    if (recording) {
      send("stop");
    } else {
      send("start");
    }
  };

  const switchSource = (source: "microphone" | "system") => {
    setAudioSource(source);
    send("switch_source", { source });
  };

  return (
    <div className="control-bar">
      <button
        className={`btn-record ${recording ? "recording" : ""}`}
        onClick={toggleRecording}
        disabled={!connected}
      >
        {recording ? "⏹ Stop" : "● Start"}
      </button>

      <div className="control-group">
        <label>Source:</label>
        <select
          value={audioSource}
          onChange={(e) => switchSource(e.target.value as "microphone" | "system")}
        >
          <option value="microphone">Microphone</option>
          <option value="system">System Audio</option>
        </select>
      </div>

      <div className="control-group">
        <label>Translate:</label>
        <select
          value={translationEngine}
          onChange={(e) => setTranslationEngine(e.target.value as "google" | "openai")}
        >
          <option value="google">Google</option>
          <option value="openai">OpenAI</option>
        </select>
      </div>

      <div className="control-group">
        <label>Display:</label>
        <select
          value={displayMode}
          onChange={(e) => setDisplayMode(e.target.value as "english" | "bilingual" | "chinese")}
        >
          <option value="bilingual">EN + 中文</option>
          <option value="english">English</option>
          <option value="chinese">中文</option>
        </select>
      </div>

      <div className={`status-dot ${connected ? "connected" : "disconnected"}`} />
    </div>
  );
}
```

`frontend/src/components/ControlBar.css`:
```css
.control-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 16px;
  background: #ffffff;
  border-top: 1px solid #e8e8e8;
  font-size: 13px;
}

.btn-record {
  padding: 6px 16px;
  border: none;
  border-radius: 6px;
  background: #4a6fa5;
  color: white;
  font-size: 13px;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-record:hover { background: #3a5f95; }
.btn-record.recording { background: #e74c3c; }
.btn-record.recording:hover { background: #c0392b; }
.btn-record:disabled { opacity: 0.5; cursor: not-allowed; }

.control-group {
  display: flex;
  align-items: center;
  gap: 4px;
}

.control-group label {
  color: #666;
  font-size: 12px;
}

.control-group select {
  padding: 4px 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 12px;
  background: white;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-left: auto;
}

.status-dot.connected { background: #2ecc71; }
.status-dot.disconnected { background: #e74c3c; }
```

**Step 2: Commit**

```bash
git add frontend/src/components/ControlBar.*
git commit -m "feat: ControlBar with recording, source, translation, display controls"
```

---

### Task 11: Main App Layout & Styling

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.css`

**Step 1: Rewrite App.tsx**

Replace the entire content of `frontend/src/App.tsx`:
```tsx
import { TranscriptView } from "./components/TranscriptView";
import { ControlBar } from "./components/ControlBar";
import "./App.css";

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>LiveScribe</h1>
      </header>
      <TranscriptView />
      <ControlBar />
    </div>
  );
}

export default App;
```

Replace the entire content of `frontend/src/App.css`:
```css
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #ffffff;
}

.app-header {
  padding: 10px 20px;
  background: #f8f9fa;
  border-bottom: 1px solid #e8e8e8;
}

.app-header h1 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: #333;
}
```

**Step 2: Verify frontend builds**

```bash
cd frontend && npm run build
```
Expected: Build succeeds with no errors.

**Step 3: Commit**

```bash
cd "/Users/d/Desktop/华工大/cc cowork/live voice transform"
git add frontend/src/App.tsx frontend/src/App.css
git commit -m "feat: main app layout with header, transcript view, and control bar"
```

---

## Phase 3: Integration

### Task 12: Settings Page

**Files:**
- Create: `frontend/src/components/Settings.tsx`
- Create: `frontend/src/components/Settings.css`
- Modify: `frontend/src/App.tsx` (add settings toggle)

**Step 1: Implement Settings component**

`frontend/src/components/Settings.tsx`:
```tsx
import { useState, useEffect } from "react";
import "./Settings.css";

interface SettingsProps {
  onClose: () => void;
}

export function Settings({ onClose }: SettingsProps) {
  const [apiKey, setApiKey] = useState("");
  const [whisperModel, setWhisperModel] = useState("base");
  const [savePath, setSavePath] = useState("");

  useEffect(() => {
    fetch("http://127.0.0.1:8765/config")
      .then((r) => r.json())
      .then((data) => {
        setApiKey(data.openai_api_key || "");
        setWhisperModel(data.whisper_model);
        setSavePath(data.save_dir);
      })
      .catch(() => {});
  }, []);

  const save = () => {
    fetch("http://127.0.0.1:8765/config", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        openai_api_key: apiKey || null,
        whisper_model: whisperModel,
        save_dir: savePath,
      }),
    }).then(() => onClose());
  };

  return (
    <div className="settings-overlay">
      <div className="settings-panel">
        <h2>Settings</h2>

        <div className="setting-row">
          <label>Whisper Model</label>
          <select value={whisperModel} onChange={(e) => setWhisperModel(e.target.value)}>
            <option value="tiny">tiny (fastest)</option>
            <option value="base">base (recommended)</option>
            <option value="small">small (more accurate)</option>
            <option value="medium">medium (slower)</option>
          </select>
        </div>

        <div className="setting-row">
          <label>OpenAI API Key</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-..."
          />
        </div>

        <div className="setting-row">
          <label>Save Directory</label>
          <input
            type="text"
            value={savePath}
            onChange={(e) => setSavePath(e.target.value)}
          />
        </div>

        <div className="setting-actions">
          <button className="btn-cancel" onClick={onClose}>Cancel</button>
          <button className="btn-save" onClick={save}>Save</button>
        </div>
      </div>
    </div>
  );
}
```

`frontend/src/components/Settings.css`:
```css
.settings-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.settings-panel {
  background: white;
  border-radius: 12px;
  padding: 24px;
  width: 400px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
}

.settings-panel h2 {
  margin: 0 0 20px;
  font-size: 18px;
}

.setting-row {
  margin-bottom: 16px;
}

.setting-row label {
  display: block;
  font-size: 13px;
  color: #666;
  margin-bottom: 4px;
}

.setting-row input,
.setting-row select {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  box-sizing: border-box;
}

.setting-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 20px;
}

.btn-cancel {
  padding: 6px 16px;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: white;
  cursor: pointer;
}

.btn-save {
  padding: 6px 16px;
  border: none;
  border-radius: 6px;
  background: #4a6fa5;
  color: white;
  cursor: pointer;
}
```

**Step 2: Add settings button to App.tsx**

Update `frontend/src/App.tsx`:
```tsx
import { useState } from "react";
import { TranscriptView } from "./components/TranscriptView";
import { ControlBar } from "./components/ControlBar";
import { Settings } from "./components/Settings";
import "./App.css";

function App() {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <div className="app">
      <header className="app-header">
        <h1>LiveScribe</h1>
        <button className="btn-settings" onClick={() => setShowSettings(true)}>
          ⚙
        </button>
      </header>
      <TranscriptView />
      <ControlBar />
      {showSettings && <Settings onClose={() => setShowSettings(false)} />}
    </div>
  );
}

export default App;
```

Add to `frontend/src/App.css`:
```css
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.btn-settings {
  background: none;
  border: none;
  font-size: 18px;
  cursor: pointer;
  color: #666;
  padding: 4px 8px;
  border-radius: 4px;
}

.btn-settings:hover {
  background: #eee;
}
```

**Step 3: Commit**

```bash
git add frontend/src/components/Settings.* frontend/src/App.*
git commit -m "feat: settings page with model, API key, and save path config"
```

---

### Task 13: Tauri Sidecar Configuration

**Files:**
- Modify: `frontend/src-tauri/tauri.conf.json`
- Modify: `frontend/src-tauri/capabilities/default.json`

**Step 1: Configure Tauri to spawn Python backend as sidecar**

In `frontend/src-tauri/tauri.conf.json`, add to the `bundle` section:
```json
{
  "bundle": {
    "externalBin": ["../backend/livescribe-server"]
  }
}
```

And add shell plugin to `plugins`:
```json
{
  "plugins": {
    "shell": {
      "open": true,
      "scope": [
        {
          "name": "livescribe-server",
          "cmd": "../backend/livescribe-server",
          "args": true
        }
      ]
    }
  }
}
```

**Step 2: Add Tauri sidecar launch code**

Install the Tauri shell plugin:
```bash
cd frontend
npm install @tauri-apps/plugin-shell
cd src-tauri
cargo add tauri-plugin-shell
```

Create `frontend/src/sidecar.ts`:
```typescript
import { Command } from "@tauri-apps/plugin-shell";

let serverProcess: Awaited<ReturnType<Command["spawn"]>> | null = null;

export async function startBackend() {
  const command = Command.sidecar("../backend/livescribe-server");

  command.on("error", (error) => {
    console.error("Backend error:", error);
  });

  command.stdout.on("data", (data) => {
    console.log("Backend:", data);
  });

  serverProcess = await command.spawn();
}

export async function stopBackend() {
  if (serverProcess) {
    await serverProcess.kill();
    serverProcess = null;
  }
}
```

**Step 3: Create PyInstaller build script for the Python backend**

Create `backend/build.sh`:
```bash
#!/bin/bash
cd "$(dirname "$0")"
pip install pyinstaller
pyinstaller --onefile --name livescribe-server main.py
cp dist/livescribe-server .
```

**Step 4: Commit**

```bash
cd "/Users/d/Desktop/华工大/cc cowork/live voice transform"
git add frontend/src-tauri/ frontend/src/sidecar.ts backend/build.sh
git commit -m "feat: Tauri sidecar config for Python backend"
```

---

### Task 14: Tauri Menu Bar (System Tray) Integration

**Files:**
- Modify: `frontend/src-tauri/src/lib.rs`

**Step 1: Add system tray to Tauri**

```bash
cd frontend/src-tauri
cargo add tauri-plugin-shell
```

Modify `frontend/src-tauri/src/lib.rs`:
```rust
use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager,
};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let show = MenuItem::with_id(app, "show", "Show LiveScribe", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &quit])?;

            TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .menu(&menu)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                })
                .build(app)?;

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**Step 2: Commit**

```bash
cd "/Users/d/Desktop/华工大/cc cowork/live voice transform"
git add frontend/src-tauri/
git commit -m "feat: system tray with show/quit menu for background running"
```

---

### Task 15: End-to-End Integration Test

**Step 1: Start Python backend manually**

```bash
cd "/Users/d/Desktop/华工大/cc cowork/live voice transform"
source .venv/bin/activate
python -m backend.main
```
Expected: Server starts on `127.0.0.1:8765`

**Step 2: In another terminal, start Tauri dev**

```bash
cd "/Users/d/Desktop/华工大/cc cowork/live voice transform/frontend"
npm run tauri dev
```
Expected: Tauri window opens, green dot shows connected

**Step 3: Test flow**

1. Click "Start" — verify recording begins (button turns red)
2. Speak into microphone — verify text appears in the transcript view
3. Switch display mode — verify layout changes
4. Click "Stop" — verify session is saved
5. Check `~/LiveScribe/sessions/` for saved JSON file
6. Close window — verify app stays in menu bar tray
7. Click tray icon — verify window reopens

**Step 4: Run all backend tests**

```bash
cd "/Users/d/Desktop/华工大/cc cowork/live voice transform"
python -m pytest backend/tests/ -v
```
Expected: All tests pass

**Step 5: Commit any fixes from integration testing**

```bash
git add -A
git commit -m "fix: integration test fixes"
```

---

## Summary

| Phase | Tasks | What it delivers |
|-------|-------|-----------------|
| Phase 1 (Tasks 1-6) | Python backend | Audio capture, Whisper transcription, translation, session save, WebSocket server |
| Phase 2 (Tasks 7-11) | Tauri frontend | React UI with transcript view, controls, settings, state management |
| Phase 3 (Tasks 12-15) | Integration | Settings page, sidecar bundling, system tray, end-to-end testing |
