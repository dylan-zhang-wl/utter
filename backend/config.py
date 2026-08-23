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
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

DEFAULT_DIR = Path.home() / "Utter"
LEGACY_DIR = Path.home() / "LiveScribe"  # v1's directory, pre-rename

ModelTier = Literal["high", "balanced", "light", "minimal"]
PolishLevel = Literal["light", "medium", "heavy"]


class AppConfig(BaseModel):
    # --- v3 core ---
    stt_provider: str = "auto"  # "auto" lets the registry pick by hardware
    model_tier: ModelTier = "balanced"
    # Gemini rather than Ollama. The author chose a hosted free tier over a
    # local model on 2026-08-10 — a 16GB machine already holding Whisper in
    # memory has no room for a language model as well, and polish is one short
    # request per utterance, which the free quota covers.
    llm_provider: str = "gemini"

    # Overrides the provider's own choice. None means "ask the key what exists
    # and take the best" — a hardcoded id is how a working build starts
    # returning 404 six months after it shipped, which is exactly what happened
    # to gpt-4o-mini in this file. `utter polish --benchmark` fills this in.
    llm_model: str | None = None
    #: OpenAI-compatible endpoint to talk to instead of api.openai.com —
    #: DeepSeek, 小米 MiMo, Groq, 硅基流动 all speak the same protocol. The
    #: provider class always supported this; until 2026-08-23 nothing wired it
    #: from config, so the setting people needed did not exist. Key still comes
    #: from the Keychain under openai_api_key regardless of whose key it is.
    llm_base_url: str | None = None

    # Vertex AI, for when there is Google Cloud credit but no API key. Not a
    # secret: a project id is an identifier, and the credential it is used with
    # lives in gcloud's own store, never here (铁律 4).
    vertex_project: str | None = None
    vertex_location: str = "global"
    vertex_model: str = "google/gemini-2.5-flash-lite"

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
    # Default is "zh" after measuring the author's own mixed-language dictation.
    # Auto-detect does not merely cost time: Whisper carries ONE language token
    # per 30s window, so a Chinese question after a long English passage came
    # back TRANSLATED into English rather than transcribed. Pinning "zh" keeps
    # Chinese as Chinese, leaves embedded English terms intact (verified on 13s
    # of English including "Venuti" and "foreignisation"), and halves latency.
    #
    # This is the right default for THIS author, not universally — an
    # English-only user should set "en".
    dictate_language: str | None = "zh"

    # Two keys, not one key with a mode switch. Holding to insert a phrase and
    # toggling on to dictate a paragraph are different gestures used at
    # different moments, and binding them separately means each can dodge
    # whatever else on this machine already claims a key.
    #
    # No default is safe on every machine — the author found right Option taken
    # by WeChat and a left-Option double tap taken by Claude, and macOS exposes
    # no way to enumerate what other apps have grabbed (see `utter keys`).
    # Either may be set to None to disable that gesture entirely.
    #
    # The names are portable — pynput reports Key.alt_l and Key.ctrl_l on
    # Windows and Linux too — but the *choice* is not. On Windows a bare tap of
    # left Alt moves focus to the menu bar in most applications, so a Windows
    # user should pick something else. Worth remembering before sharing this
    # with a colleague.
    # Chosen by the author on their own keyboard, 2026-08-10, after `utter keys`:
    # right Option is WeChat's push-to-talk and right Command / F13 were awkward
    # to reach. Both of these sit under the left hand, leaving the right free.
    hotkey_push: str | None = "<alt_l>"
    hotkey_toggle: str | None = "<ctrl_l>"

    # Double tap rather than single. A single press on a bare modifier is a trap:
    # brushing the key silently starts recording everything said next. macOS uses
    # a double tap for its own dictation shortcut for the same reason.
    hotkey_toggle_double_tap: bool = True

    # 边说边出字, on the toggle gesture only (P2b, design §4.1a).
    #
    # Bounded by 铁律 9: injected text is never revised, so there is no live
    # caption being corrected — each clause is transcribed once its pause has
    # arrived and lands final. Push-to-talk is unaffected: the finger is
    # already the segmenter there.
    #
    # Off by default, at the user's request after living with it. Streaming
    # trades punctuation quality for immediacy: a clause is closed by the
    # silence around it rather than by Whisper reading the whole sentence, so
    # the text arrives sooner and reads slightly rougher. That is the right
    # trade for some work and the wrong one for dictating an argument, and the
    # safer default is the one that does not surprise anybody. One switch away.
    stream_while_speaking: bool = False

    # 听记的翻译档位（P3 §2.2）。"literal" 忠实、"fluent" 联系上下文补通顺、
    # "explain" 再加一句必要背景。默认取中间那档：与润色不同，译文是**并排**
    # 摆在原文旁边的，改错了抬眼就能看见，所以这里允许解释性翻译。
    translate_level: str = "fluent"

    # 听记的分段。**不能沿用 vad_silence_ms=600 / max_utterance_sec=30**，
    # 那是给"一段话说完了吗"用的。2026-08-16 拿一场真实的英文演讲实测：
    # 讲者根本不停 600ms，于是每一条都撞到 30 秒上限——
    #
    #   10 条记录，每条 380–495 字，间隔 26–30 秒
    #
    # 也就是说**每 30 秒才出一次字，再加 3 秒转录**。作者的原话是
    # "哪怕延迟 2 秒都没做到"，完全正确。
    #
    # 500ms 问的是"这句说完了吗"，12 秒封顶保证不会等太久。代价是每段音频
    # 的固定开销摊得更薄（实测短音频约 375ms/音频秒 vs 长音频 100ms），
    # 占空比升到三四成——余量仍然够，而且翻译本来就是并行的。
    listen_silence_ms: int = 500
    listen_max_seconds: int = 8
    # And a floor, added after the first fix overshot. 500ms alone cut on every
    # breath between phrases: 25 pieces, median 23 characters, half of them not
    # sentences at all. A pause only ends an utterance once there are a few
    # seconds of speech behind it, so 「My father / from equity states / in
    # Southwest ...」 stays one sentence.
    #
    # The two numbers together bound the latency: nothing waits longer than
    # listen_max_seconds, and nothing arrives in fragments shorter than this.
    # 1.5, not 4. The floor existed because sentences were being cut from the
    # audio, so a chunk had to be long enough to *be* a sentence. Sentences are
    # now cut from the text and a short chunk's tail is simply joined to the
    # next one — which makes the floor pure waiting. Measured 2026-08-17:
    # Whisper costs ~1.1s whether the chunk is 10s or 18s, and the LLM ~0.9s,
    # so the models were never the delay. Waiting for the sentence to finish
    # being spoken was, and four seconds of that was self-inflicted.
    listen_min_seconds: float = 1.5

    #: The grey tail — a small model re-reading the sentence in progress so the
    #: screen is not blank while somebody talks. Never archived, never
    #: translated; see backend/preview.py for why it has to be a second model
    #: rather than a faster tick of the first one.
    #: The language being *listened to*, which has nothing to do with the one
    #: you dictate in. Listen mode read `dictate_language` until 2026-08-20,
    #: so a user who dictates in Chinese was telling Whisper that an English
    #: lecture was Chinese. The big model mostly shrugged that off; the small
    #: preview model did not, and put 「它有色彩的…」 under the English.
    listen_language: str | None = "en"

    listen_preview: bool = True
    listen_preview_tick: float = 1.0
    listen_preview_tier: str = "minimal"

    # How many sentences travel in one translation request, and how long the
    # queue may hold a lone one.
    #
    # This started at 3 and 8s, which was right when entries arrived every two
    # seconds: batching bought back the round trips. Making the segments whole
    # sentences broke that arithmetic — at 5-12 seconds an entry, waiting for
    # three of them is 15 to 36 seconds before any Chinese appears. Fixing the
    # transcription latency had quietly tripled the translation latency.
    #
    # One at a time, and a short wait. It costs more requests per meeting and
    # the previous two entries still go along as context, so nothing is lost
    # but the batching.
    # Two, not one. With whole sentences arriving every few seconds, a pair
    # gives the model something to make cohere — a pronoun in the second
    # sentence can see its referent in the first — while costing at most one
    # sentence of lag. Translating each in isolation was fast and read like a
    # list of unrelated fragments.
    translate_batch: int = 1
    translate_wait_seconds: float = 1.0

    # 结束时自动生成纪要。关掉的话记录照写，只是不跑那几次模型往返。
    summarise_at_end: bool = True

    # --- 听记的外观，全部可调 ---
    listen_font_size: float = 14.0
    #: 高对比：译文用主文字色而不是次要色。默认关——次要色让原文和译文
    #: 一眼分得开——但那是个偏好，不该由我替使用者决定。
    listen_high_contrast: bool = False

    # Where to cut, measured on the author's own dictation 2026-08-11. The
    # numbers matter more than they look:
    #
    #   600ms (listen mode's value)  ->  2 segments in 47s, both force-cuts
    #   400ms                        -> 12 segments of 3.3-5.4s   <- sentences
    #   250ms                        -> 25 segments, many under a second
    #
    # 600 asks "is this utterance over"; a fluent speaker never pauses that
    # long mid-paragraph, so nothing appeared until the 30-second ceiling fired.
    # 400 asks "did a clause just end", which is the question streaming needs.
    stream_silence_ms: int = 400
    # And a ceiling, so a speaker who never pauses still sees text. Ten seconds
    # rather than listen mode's thirty: three sentences behind is already too
    # far to feel live.
    stream_max_seconds: int = 10
    # Whisper does not close a sentence when the audio stops mid-breath, which
    # is every push-to-talk release. One utterance in seven came back with any
    # end punctuation at all. Off if you dictate one sentence across several
    # presses — a full stop mid-thought is worse than none.
    close_sentences: bool = True

    # ⌘V only posts a keystroke; the application reads the pasteboard on its own
    # event loop some milliseconds later, and restoring before it does means it
    # pastes the *old* clipboard. VoiceInk exposes this as a setting rather than
    # a constant, which is the right call — how long an application takes varies,
    # and 250ms is a guess that happened to work here.
    paste_settle_ms: int = 250

    # Last resort when clipboard paste is silently refused. Off by default:
    # measured on this machine, typing 125 characters delivered only 113 of
    # them. Losing a dozen characters without knowing which is worse than a
    # visible failure — but it beats nothing at all in an app that will not
    # accept a paste (铁律 12 forbids this as the *default*, not as a fallback).
    type_out_fallback: bool = False
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
    # default_factory, not a plain default: a plain default is evaluated
    # when this module is imported, which bakes the real home directory
    # into the class and makes the path unpatchable afterwards. That is
    # how the test suite came to write 1,299 session files into the
    # author's live archive.
    save_dir: str = Field(default_factory=lambda: str(DEFAULT_DIR / "sessions"))
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
    """Keep what validates, drop what does not — and say what was dropped.

    Dropping silently is how the author's 30-term vocabulary disappeared: a
    degraded load returns defaults, something saves, and the defaults are now
    the file. Losing a setting is survivable; losing it without a word is the
    thing this project does not do.
    """
    usable, dropped = {}, []
    for key, value in data.items():
        if key not in AppConfig.model_fields:
            dropped.append(f"{key}（不认识这个设置）")
            continue
        try:
            AppConfig(**{key: value})
        except ValueError:
            dropped.append(key)
            continue
        usable[key] = value
    if dropped:
        log.warning("config: 丢掉了看不懂的字段 %s —— 这些设置会回到默认值", "、".join(dropped))
    return AppConfig(**usable)


def load_or_none(base_dir: Path | None = None) -> AppConfig | None:
    """Like `load`, but None when the file exists and could not be read.

    `load` never fails, by design: a user who cannot launch the app cannot use
    it to repair the setting that broke it. But "never fails" means it hands
    back defaults, and a caller that then *writes* turns a temporary read
    problem into permanent data loss. Anything about to save should ask this
    one instead.
    """
    path = config_path(base_dir)
    if not path.exists():
        return AppConfig()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        log.warning("config: 读不了 %s —— 这次不覆盖它", path)
        return None
    if not isinstance(data, dict):
        return None
    try:
        return AppConfig(**data)
    except ValueError:
        return _load_field_by_field(data)


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
