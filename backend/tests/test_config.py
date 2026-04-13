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
