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
_process_task: asyncio.Task | None = None


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
BUFFER_DURATION_SEC = 8  # transcribe every N seconds


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

                global _process_task
                if _process_task is None or _process_task.done():
                    _process_task = asyncio.create_task(process_audio_loop())
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
