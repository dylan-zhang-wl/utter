"""P1 Task 1 — config that actually persists.

v1's `PATCH /config` only mutated an in-memory pydantic object, so the Settings
page's Save button was an illusion: every setting reverted on restart. These
tests pin the fix.

Every test passes explicit directories. Nothing here may touch the real
~/Utter or ~/LiveScribe — the latter holds the author's three v1 sessions.
"""

import json
import stat

import pytest

from backend import config as cfg


def test_defaults_match_the_design():
    c = cfg.AppConfig()
    assert c.stt_provider == "auto"
    assert c.model_tier == "balanced"
    # Gemini, not Ollama, since 2026-08-10. A 16GB machine already holding
    # Whisper has no room for a local language model, and polish is one short
    # request per utterance — well inside a free tier.
    assert c.llm_provider == "gemini"
    assert c.vad_silence_ms == 600
    assert c.vad_sensitivity == 0.5
    assert c.max_utterance_sec == 30


def test_input_device_defaults_to_the_system_default():
    assert cfg.AppConfig().input_device is None


def test_input_device_round_trips(tmp_path):
    cfg.save(cfg.AppConfig(input_device=3), base_dir=tmp_path)
    assert cfg.load(base_dir=tmp_path).input_device == 3


def test_polish_defaults_encode_the_rules():
    """铁律 10 and the P2a decision, locked in as defaults rather than prose."""
    c = cfg.AppConfig()
    assert c.polish_enabled is False, "P2a ships with polish off (design §8)"
    assert c.polish_level == "light", "铁律 10: the conservative level is the default"
    assert c.vocabulary == []


def test_save_then_load_round_trips(tmp_path):
    c = cfg.AppConfig(model_tier="high", vad_silence_ms=450)
    cfg.save(c, base_dir=tmp_path)

    loaded = cfg.load(base_dir=tmp_path)
    assert loaded.model_tier == "high"
    assert loaded.vad_silence_ms == 450


def test_load_without_a_file_returns_defaults(tmp_path):
    loaded = cfg.load(base_dir=tmp_path / "nothing-here")
    assert loaded.model_tier == "balanced"


def test_load_survives_a_corrupt_file(tmp_path):
    """A bad config file must not stop the app from starting."""
    path = cfg.config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json")

    assert cfg.load(base_dir=tmp_path).model_tier == "balanced"


def test_written_file_is_0600(tmp_path):
    cfg.save(cfg.AppConfig(), base_dir=tmp_path)
    mode = stat.S_IMODE(cfg.config_path(tmp_path).stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {mode:o}"


def test_api_key_never_reaches_the_file(tmp_path):
    """铁律 4. Keys live in the Keychain; the JSON must not know about them."""
    c = cfg.AppConfig(openai_api_key="sk-should-never-be-written")
    cfg.save(c, base_dir=tmp_path)

    raw = cfg.config_path(tmp_path).read_text()
    assert "sk-should-never-be-written" not in raw
    assert "openai_api_key" not in json.loads(raw)


def test_api_key_is_also_absent_from_model_dump():
    """The same exclusion protects GET /config, which returns model_dump()."""
    c = cfg.AppConfig(openai_api_key="sk-leak")
    assert "openai_api_key" not in c.model_dump()
    assert c.openai_api_key == "sk-leak", "attribute access must still work"


def test_unknown_keys_in_the_file_are_ignored(tmp_path):
    """Forward compatibility: a newer build's config must not crash an older one."""
    path = cfg.config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model_tier": "light", "invented_later": 1}))

    assert cfg.load(base_dir=tmp_path).model_tier == "light"


# --- migration from the v1 directory ----------------------------------------


def _legacy_tree(root):
    sessions = root / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "session_20260414_233134.json").write_text('{"id": "one"}')
    return sessions


def test_migration_copies_v1_sessions(tmp_path):
    old, new = tmp_path / "LiveScribe", tmp_path / "Utter"
    _legacy_tree(old)

    assert cfg.migrate_legacy(old_dir=old, new_dir=new) is True
    assert (new / "sessions" / "session_20260414_233134.json").exists()


def test_migration_preserves_the_original(tmp_path):
    """Copy, never move. This is the author's only copy of the v1 sessions."""
    old, new = tmp_path / "LiveScribe", tmp_path / "Utter"
    _legacy_tree(old)

    cfg.migrate_legacy(old_dir=old, new_dir=new)
    assert (old / "sessions" / "session_20260414_233134.json").exists()


def test_migration_does_not_run_when_target_exists(tmp_path):
    """Never overwrite live data with a stale v1 copy."""
    old, new = tmp_path / "LiveScribe", tmp_path / "Utter"
    _legacy_tree(old)
    (new / "sessions").mkdir(parents=True)
    (new / "sessions" / "current.json").write_text('{"id": "newer"}')

    assert cfg.migrate_legacy(old_dir=old, new_dir=new) is False
    assert not (new / "sessions" / "session_20260414_233134.json").exists()
    assert (new / "sessions" / "current.json").read_text() == '{"id": "newer"}'


def test_migration_is_a_noop_without_a_legacy_dir(tmp_path):
    assert cfg.migrate_legacy(
        old_dir=tmp_path / "absent", new_dir=tmp_path / "Utter"
    ) is False


def test_load_triggers_migration(tmp_path):
    old, new = tmp_path / "LiveScribe", tmp_path / "Utter"
    _legacy_tree(old)

    cfg.load(base_dir=new, legacy_dir=old)
    assert (new / "sessions" / "session_20260414_233134.json").exists()


def test_save_dir_defaults_under_the_new_home():
    assert cfg.AppConfig().save_dir.endswith("Utter/sessions")


@pytest.mark.parametrize("tier", ["high", "balanced", "light", "minimal"])
def test_every_tier_is_accepted(tier):
    assert cfg.AppConfig(model_tier=tier).model_tier == tier


def test_nonsense_tier_is_rejected():
    with pytest.raises(ValueError):
        cfg.AppConfig(model_tier="enormous")


# --- a bad read must not become a bad write -------------------------------------


def test_an_unreadable_config_is_not_silently_replaced(tmp_path):
    """`load` never fails, by design — a user who cannot launch the app cannot
    use it to repair the setting that broke it. But that means it returns
    defaults, and a caller that then saves turns a read problem into permanent
    data loss. That is how 30 vocabulary terms disappeared."""
    path = tmp_path / "config.json"
    path.write_text("{ this is not json")

    assert cfg.load(base_dir=tmp_path).vocabulary == [], "load still yields something usable"
    assert cfg.load_or_none(base_dir=tmp_path) is None, "but a writer is told not to"


def test_a_missing_config_is_fine_to_write_over(tmp_path):
    assert cfg.load_or_none(base_dir=tmp_path) is not None


def test_a_readable_config_round_trips(tmp_path):
    cfg.save(cfg.AppConfig(vocabulary=["foreignisation"]), base_dir=tmp_path)
    loaded = cfg.load_or_none(base_dir=tmp_path)
    assert loaded is not None and loaded.vocabulary == ["foreignisation"]


def test_dropped_fields_are_reported(tmp_path, caplog):
    """Losing a setting is survivable. Losing it without a word is not."""
    import json
    import logging

    (tmp_path / "config.json").write_text(json.dumps({
        "vocabulary": ["Venuti"],
        "polish_level": "nuclear",       # not a valid level
    }))

    with caplog.at_level(logging.WARNING):
        loaded = cfg.load(base_dir=tmp_path)

    assert loaded.vocabulary == ["Venuti"], "the good fields survive"
    assert loaded.polish_level == "light", "the bad one falls back to the safe value"
    assert any("polish_level" in r.getMessage() for r in caplog.records), \
        "the drop has to be reported, not just survived"
