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
    assert c.llm_provider == "ollama"
    assert c.vad_silence_ms == 600
    assert c.vad_sensitivity == 0.5
    assert c.max_utterance_sec == 30


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
