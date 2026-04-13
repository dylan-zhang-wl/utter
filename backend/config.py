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
