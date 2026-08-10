"""Application configuration, persisted to ~/Utter/config.json.

v1 kept configuration in a module-level pydantic object that `PATCH /config`
mutated in memory. Nothing was ever written to disk, so the Settings page's Save
button did nothing that survived a restart. This module is the fix.

Two rules are enforced structurally rather than by convention:

  * 铁律 4 — `openai_api_key` is `exclude=True`, so it is absent from both the
    JSON file and `model_dump()` (which GET /config returns over HTTP).
    Keys belong in the Keychain; see backend/secrets.py.
  * 铁律 10 — `polish_level` defaults to "light" and `polish_enabled` to False.
    An LLM quietly rewriting an academic argument into something fluent but
    different is this project's worst failure mode, so the safe setting is the
    one you get without asking.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_DIR = Path.home() / "Utter"
LEGACY_DIR = Path.home() / "LiveScribe"  # v1's directory, pre-rename

ModelTier = Literal["high", "balanced", "light", "minimal"]
PolishLevel = Literal["light", "medium", "heavy"]


class AppConfig(BaseModel):
    # --- v3 core ---
    stt_provider: str = "auto"  # "auto" lets the registry pick by hardware
    model_tier: ModelTier = "balanced"
    llm_provider: str = "ollama"

    # --- audio input ---
    # None means "whatever macOS calls the default". Not a safe assumption on
    # this machine: the default has been seen as a Bluetooth speaker and as
    # AirPods, and no built-in microphone appears in the list at all. Bluetooth
    # inputs often negotiate a call-quality profile, which degrades
    # transcription in a way that reads as a bad model rather than a bad mic.
    input_device: int | None = None

    # --- segmentation (design §5.4) ---
    vad_silence_ms: int = 600
    vad_sensitivity: float = 0.5
    max_utterance_sec: int = 30

    # --- dictation (P2a) ---
    # Measured 2026-08-10 on 8s of speech: language="en" 1067ms, "zh" 1057ms,
    # None 1927ms. Whisper runs a separate detection pass when it is not told,
    # and that pass alone costs more than half the entire latency budget.
    # None stays the default anyway, because the author writes in both languages
    # and a wrong guess is worse than a slow answer — but `utter dictate` says
    # so at startup, since 860ms is a lot to pay silently for something one
    # config line fixes.
    dictate_language: str | None = None

    # Two keys, not one key with a mode switch. Holding to insert a phrase and
    # toggling on to dictate a paragraph are different gestures used at
    # different moments, and binding them separately means each can dodge
    # whatever else on this machine already claims a key.
    #
    # No default is safe on every machine — the author found right Option taken
    # by WeChat and a left-Option double tap taken by Claude, and macOS exposes
    # no way to enumerate what other apps have grabbed (see `utter keys`).
    # Either may be set to None to disable that gesture entirely.
    hotkey_push: str | None = "<alt_r>"
    hotkey_toggle: str | None = None

    # Double tap rather than single. A single press on a bare modifier is a trap:
    # brushing the key silently starts recording everything said next. macOS uses
    # a double tap for its own dictation shortcut for the same reason.
    hotkey_toggle_double_tap: bool = True
    dictate_target: Literal["cursor", "scratchpad"] = "cursor"

    # --- dictation post-processing (design §4.1g, §4.1h) ---
    polish_enabled: bool = False
    polish_level: PolishLevel = "light"
    vocabulary: list[str] = Field(default_factory=list)

    # --- carried over from v1; main.py still reads these until P3 ---
    whisper_model: str = "small"
    audio_source: str = "microphone"  # "microphone" | "system"
    translation_engine: str = "google"  # "google" | "openai"
    display_mode: str = "bilingual"  # "english" | "bilingual" | "chinese"
    save_dir: str = str(DEFAULT_DIR / "sessions")
    server_host: str = "127.0.0.1"
    server_port: int = 8765

    # Transitional: reachable as an attribute so v1's main.py keeps working, but
    # excluded from every serialisation path. Task 2 moves the real storage into
    # the Keychain and this field goes away with main.py's rewrite in P3.
    openai_api_key: str | None = Field(default=None, exclude=True)


def config_path(base_dir: Path | None = None) -> Path:
    return (base_dir or DEFAULT_DIR) / "config.json"


def migrate_legacy(*, old_dir: Path, new_dir: Path) -> bool:
    """Copy v1's ~/LiveScribe to ~/Utter. Returns whether anything was copied.

    Copies rather than moves. The design called this a 搬迁, but the author's
    three v1 sessions are the only copy in existence and they total three
    kilobytes — there is nothing to gain by removing the original, and a botched
    move loses data that cannot be regenerated. The old directory is left for
    the author to delete by hand.

    Refuses to run when the target already exists, so a stale v1 tree can never
    overwrite live data.
    """
    if not old_dir.is_dir() or new_dir.exists():
        return False

    shutil.copytree(old_dir, new_dir)
    return True


def load(base_dir: Path | None = None, legacy_dir: Path | None = None) -> AppConfig:
    """Read the config file, falling back to defaults for anything unusable.

    Never raises. A missing, unreadable, or corrupt file all yield defaults —
    the app must always start, because a user who cannot launch the app also
    cannot use it to repair the setting that broke it.
    """
    base_dir = base_dir or DEFAULT_DIR
    migrate_legacy(old_dir=legacy_dir or LEGACY_DIR, new_dir=base_dir)

    path = config_path(base_dir)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return AppConfig()

    if not isinstance(data, dict):
        return AppConfig()

    try:
        return AppConfig(**data)
    except ValueError:
        # A single bad value (hand-edited tier, say) should not cost every other
        # setting. Keep what validates, drop what does not.
        return _load_field_by_field(data)


def _load_field_by_field(data: dict) -> AppConfig:
    usable = {}
    for key, value in data.items():
        if key not in AppConfig.model_fields:
            continue
        try:
            AppConfig(**{key: value})
        except ValueError:
            continue
        usable[key] = value
    return AppConfig(**usable)


def save(config: AppConfig, base_dir: Path | None = None) -> Path:
    """Write the config atomically with 0600 permissions.

    Atomic because a crash mid-write would otherwise leave a truncated file that
    the next launch has to fall back out of. 0600 because this file sits in the
    home directory and, while it must never hold a key, it does hold the user's
    vocabulary list.
    """
    base_dir = base_dir or DEFAULT_DIR
    base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = config_path(base_dir)

    fd, tmp = tempfile.mkstemp(dir=base_dir, prefix=".config-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(config.model_dump(), fh, indent=2, ensure_ascii=False)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    return path
