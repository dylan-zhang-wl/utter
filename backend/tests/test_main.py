import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Patch WhisperModel so importing backend.main (and lifespan) does not download the model
_whisper_patcher = patch("backend.transcriber.WhisperModel", MagicMock())
_whisper_patcher.start()

from backend.main import app  # noqa: E402


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
