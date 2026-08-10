"""`utter` on the command line — P1's proof that the core works with no GUI.

Three commands:

    utter doctor                  what this machine is, and what works on it
    utter models list             the four tiers, their size, what is downloaded
    utter transcribe FILE         run a file through the real pipeline

`doctor` is the one that matters for distribution: it is what a colleague on a
Windows laptop runs when nothing happens, so every line has to be actionable on
a machine the author cannot log into.

File input rather than microphone. It keeps the whole thing testable, and live
capture belongs to P2 where the hotkey defines the segment.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from backend import catalog, models
from backend.config import load as load_config
from backend.hardware import detect as detect_hardware
from backend.pipeline import dictate_pipeline, listen_pipeline
from backend.providers.stt import NoProviderAvailable, get_stt_provider, probe_all
from backend.vad import SAMPLE_RATE, VadSegmenter

log = logging.getLogger(__name__)

CHUNK = 1600  # 100 ms, the size a microphone callback would deliver


def _load_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"expected {SAMPLE_RATE} Hz, got {sample_rate}. Convert first:\n"
            f"  ffmpeg -i {path.name} -ar 16000 -ac 1 out.wav"
        )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix; VAD and Whisper both want mono
    return audio.astype(np.float32)


def _chunks(audio: np.ndarray):
    for i in range(0, len(audio), CHUNK):
        yield audio[i : i + CHUNK]


# --- doctor ------------------------------------------------------------------


def cmd_doctor(_args, out) -> int:
    hardware = detect_hardware()
    config = load_config()

    print("Hardware", file=out)
    print(f"  {hardware.label}", file=out)

    print("\nSpeech-to-text providers", file=out)
    statuses = probe_all(hardware=hardware)
    if not statuses:
        print("  none found", file=out)
    for status in statuses:
        mark = "✓" if status.available else "✗"
        detail = f"  — {status.reason}" if status.reason else ""
        print(f"  [{mark}] {status.id:<16} {status.display_name}{detail}", file=out)

    print("\nLanguage-model providers (translation and polish)", file=out)
    for provider in _llm_providers():
        available, reason = provider.is_available()
        mark = "✓" if available else "✗"
        detail = f"  — {reason}" if reason else ""
        print(f"  [{mark}] {provider.id:<16} {provider.display_name}{detail}", file=out)

    print("\nDictation prerequisites", file=out)
    for label, (ok, reason) in _dictation_checks(config).items():
        mark = "✓" if ok else "✗"
        detail = f"  — {reason}" if reason else ""
        print(f"  [{mark}] {label}{detail}", file=out)

    print("\nModels", file=out)
    for usage in models.disk_usage(hardware=hardware):
        mark = "✓" if usage.present else "—"
        print(f"  [{mark}] {usage.label:<12} {usage.size_mb:>5} MB  {usage.repo}", file=out)

    print(f"\nConfig\n  tier={config.model_tier}  provider={config.stt_provider}  "
          f"polish={'on' if config.polish_enabled else 'off'} ({config.polish_level})",
          file=out)
    return 0


def _dictation_checks(config) -> dict[str, tuple[bool, str]]:
    """What P2a needs that P1 did not. Each line has to be actionable.

    Both of these fail silently in their natural state — a missing Accessibility
    grant makes the hotkey deaf without an error, and the wrong input device
    just sounds bad — so `doctor` is where they become visible.
    """
    checks: dict[str, tuple[bool, str]] = {}

    try:
        from backend.hotkey import HotkeyListener

        checks["辅助功能权限 (hotkey + injection)"] = HotkeyListener(
            on_event=lambda _e: None
        ).is_available()
    except Exception as exc:  # pragma: no cover - defensive
        checks["辅助功能权限 (hotkey + injection)"] = (False, str(exc))

    try:
        from backend import audio_source

        devices = audio_source.list_devices()
        if config.input_device is not None:
            chosen = next((d for d in devices if d.index == config.input_device), None)
            label = chosen.name if chosen else f"设备 {config.input_device} 不存在"
            checks["麦克风"] = (chosen is not None, label)
        else:
            default = next((d for d in devices if d.is_default), None)
            checks["麦克风"] = (
                default is not None,
                f"{default.name}（系统默认）" if default else "找不到任何输入设备",
            )

        shared = audio_source.shared_with_output(config.input_device)
        if shared:
            checks["输入输出不共用设备"] = (
                False,
                f"{shared} 同时是麦克风和扬声器。听写时 macOS 会把它切到通话模式，"
                "造成音画不同步、音量变化、转录质量下降。建议插一个 USB 麦克风，"
                "或在 config.json 里把 input_device 设成别的设备（`utter doctor` 上方有编号）",
            )
    except Exception as exc:  # pragma: no cover - defensive
        checks["麦克风"] = (False, str(exc))

    return checks


def build_polish(config):
    """Turn config into the callable the daemon expects, or None.

    P2a Task 8. Everything under it — the prompts, the levels, the length-ratio
    guard, the fall-back-to-raw contract — has existed and been tested since
    P1. Nothing ever constructed the callable, so `polish_enabled: true` did
    precisely nothing. A setting that silently does nothing is the same failure
    as v1's Save button.

    Returns None when polish is off or unreachable, and the daemon treats None
    as "off" — never as an error. 铁律 8: a missing key costs the polish, never
    the words.
    """
    if not config.polish_enabled:
        return None

    provider = _llm_provider_named(config.llm_provider)
    if provider is None:
        log.warning("polish is on but %r is unknown", config.llm_provider)
        return None

    available, reason = provider.is_available()
    if not available:
        # Loud, because the author switched this on and expects it to happen.
        print(f"\n⚠ 润色开着，但 {provider.display_name} 用不了：{reason}\n"
              "  这次会照常听写，只是不润色。\n", flush=True)
        return None

    from backend.providers.llm import safe_polish

    def polish(text: str, context: str | None = None) -> str:
        # safe_polish returns (text, polished?) and never raises; the daemon
        # only wants the text, and treats "unchanged" as not polished.
        result, _ = safe_polish(
            provider,
            text,
            level=config.polish_level,
            context=context,
            vocabulary=list(config.vocabulary),
        )
        return result

    return polish


def _llm_provider_named(name: str):
    for provider in _llm_providers():
        if provider.id == name:
            return provider
    return None


def _llm_providers():
    """Instantiated defensively — a broken optional dependency should degrade
    one line of `doctor`, not the whole command."""
    found = []
    for module_name, class_name in (
        ("backend.providers.gemini", "GeminiProvider"),
        ("backend.providers.ollama", "OllamaProvider"),
        ("backend.providers.openai_compat", "OpenAICompatProvider"),
        ("backend.providers.free_translate", "FreeTranslateProvider"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            found.append(getattr(module, class_name)())
        except Exception:  # pragma: no cover - defensive
            pass
    return found


# --- models ------------------------------------------------------------------


def cmd_models(args, out) -> int:
    if args.models_command != "list":
        print(f"unknown models subcommand {args.models_command!r}", file=out)
        return 2

    hardware = detect_hardware()
    print(f"Tiers for {hardware.label}\n", file=out)
    print(f"  {'':<2} {'档位':<12} {'体积':>8}  {'说明'}", file=out)
    for entry, usage in zip(catalog.all_tiers(hardware), models.disk_usage(hardware=hardware)):
        mark = "✓" if usage.present else "—"
        print(f"  {mark:<2} {entry.label:<12} {entry.size_mb:>5} MB  {entry.blurb}", file=out)
    print("\n  ✓ = already downloaded", file=out)
    return 0


# --- transcribe --------------------------------------------------------------


def cmd_transcribe(args, out, *, stt=None, translate=None, polish=None) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"file not found: {path}", file=out)
        return 1

    try:
        audio = _load_audio(path)
    except ValueError as exc:
        print(str(exc), file=out)
        return 1

    config = load_config()

    if stt is None:
        try:
            stt = get_stt_provider(preferred=config.stt_provider)
        except NoProviderAvailable as exc:
            print(str(exc), file=out)
            print("\nRun `utter doctor` for the full picture.", file=out)
            return 1

    timings: list[float] = []
    started = {"at": 0.0}

    class TimedStt:
        """Wraps the provider so --timing measures the model alone, not the
        VAD or the printing around it."""

        def transcribe(self, chunk, language=None, initial_prompt=None):
            started["at"] = time.perf_counter()
            result = stt.transcribe(chunk, language=language, initial_prompt=initial_prompt)
            timings.append(time.perf_counter() - started["at"])
            return result

    def emit(utterance):
        stamp = f"[{utterance.start_sec:6.1f}s → {utterance.end_sec:6.1f}s]"
        timing = f"  ({timings[utterance.index]:.2f}s)" if args.timing and timings else ""
        forced = "  [forced cut]" if utterance.forced else ""
        print(f"{stamp}{timing}{forced}", file=out)
        print(f"  {utterance.text}", file=out)
        if utterance.translation:
            print(f"  {utterance.translation}", file=out)
        if utterance.polished and utterance.raw_text != utterance.text:
            print(f"  (raw: {utterance.raw_text})", file=out)
        print(file=out)

    factory = listen_pipeline if args.mode == "listen" else dictate_pipeline
    kwargs = {
        "stt": TimedStt(),
        "sink": emit,
        "vocabulary": list(config.vocabulary),
    }
    if args.mode == "listen" and translate is not None:
        kwargs["translate"] = translate
    if args.mode == "dictate" and polish is not None:
        kwargs["polish"] = polish
    if args.language:
        kwargs["language"] = args.language

    pipeline = factory(**kwargs)
    segmenter = VadSegmenter(
        vad_silence_ms=config.vad_silence_ms,
        vad_sensitivity=config.vad_sensitivity,
        max_utterance_sec=config.max_utterance_sec,
    )

    wall_start = time.perf_counter()
    pipeline.run_stream(_chunks(audio), segmenter)
    wall = time.perf_counter() - wall_start

    count = pipeline._index
    if count == 0:
        print("no speech found", file=out)
        return 0

    if args.timing and timings:
        duration = len(audio) / SAMPLE_RATE
        print(
            f"{count} utterances from {duration:.1f}s of audio in {wall:.1f}s\n"
            f"  per utterance: min {min(timings):.2f}s  "
            f"median {sorted(timings)[len(timings) // 2]:.2f}s  "
            f"max {max(timings):.2f}s\n"
            f"  duty cycle: {sum(timings) / duration * 100:.0f}% "
            f"(>100% means the pipeline falls behind a live speaker)",
            file=out,
        )
    return 0


# --- entry point -------------------------------------------------------------


def _build_stamp() -> str:
    """Which build is actually running.

    Several rounds of debugging were spent on symptoms that had already been
    fixed, because the daemon is resident and the author was still running the
    process from before the fix. Neither of us could tell by looking. Now the
    banner says.
    """
    import subprocess
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "log", "-1", "--format=%h %cd", "--date=format:%m-%d %H:%M"],
            capture_output=True, text=True, timeout=3,
        )
        return f"(build {out.stdout.strip()})" if out.returncode == 0 else ""
    except Exception:
        return ""


#: Which Keychain entry each provider reads, so `utter key` can name them
#: rather than making the author find the string in the source.
KEY_NAMES = {
    "google_api_key": "Gemini — https://aistudio.google.com/apikey",
    "openai_api_key": "OpenAI / DeepSeek / Groq 等 OpenAI 兼容端点",
}


def cmd_key(args, out) -> int:
    """Store an API key in the Keychain, without it ever touching a file.

    铁律 4. The alternative on offer was `security add-generic-password`, which
    works but requires getting the service name exactly right by hand — and a
    key stored under the wrong service reads back as "no key stored", which is
    a miserable thing to debug.

    getpass, so the key is not echoed and does not enter shell history. Nothing
    here logs the value, and the only confirmation printed is its length.
    """
    import getpass

    from backend import secrets

    if args.name == "list" or not args.name:
        print("可以存的 key：\n", file=out)
        for name, what in KEY_NAMES.items():
            stored = "✓ 已存" if secrets.get_secret(name) else "— 没存"
            print(f"  {stored}  {name:<16} {what}", file=out)
        print(f"\n存一个：utter key google_api_key\n删掉：utter key google_api_key --forget", file=out)
        return 0

    if args.forget:
        secrets.delete_secret(args.name)
        print(f"已删除 {args.name}。", file=out)
        return 0

    print(f"粘贴 {args.name}（不会显示，也不会进 shell 历史），回车确认：", file=out)
    try:
        value = getpass.getpass("").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n取消了，什么都没改。", file=out)
        return 1
    if not value:
        print("空的，什么都没改。", file=out)
        return 1

    secrets.set_secret(args.name, value)
    stored = secrets.get_secret(args.name)
    if stored != value:
        print("⚠ 写进钥匙串后读回来对不上，没存成。", file=out)
        return 1
    # Length only. Printing any part of a key is how keys end up in screenshots.
    print(f"✓ 已存进钥匙串（{len(value)} 个字符）。用 `utter doctor` 确认。", file=out)
    return 0


def cmd_polish(args, out) -> int:
    """Run the polish prompt against the real model, on text you type.

    P2a Task 8 says the prompt must be *verified* against a real model, and
    verifying it through dictation would mean the author speaking a paragraph
    every time a wording changes. This takes the text on the command line, so
    the same input can be run against every level and every model.

    It prints both versions, because 铁律 10's whole defence is that the
    original stays visible.
    """
    from backend.providers.llm import polish_prompt, safe_polish

    config = load_config()
    name = args.provider or config.llm_provider
    provider = _llm_provider_named(name)
    if provider is None:
        print(f"没有叫 {name!r} 的 LLM provider。`utter doctor` 列了有哪些。", file=out)
        return 1

    available, reason = provider.is_available()
    if not available:
        print(f"{provider.display_name} 用不了：{reason}", file=out)
        return 1

    text = args.text or sys.stdin.read()
    if not text.strip():
        print("没有输入。用法：utter polish '要处理的文字'", file=out)
        return 1

    levels = [args.level] if args.level else ["light", "medium", "heavy"]
    print(f"模型  {getattr(provider, 'model', provider.display_name)}\n", file=out)
    print(f"原文  {text.strip()}\n", file=out)

    for level in levels:
        started = time.perf_counter()
        result, changed = safe_polish(
            provider, text.strip(), level=level,
            vocabulary=list(config.vocabulary),
        )
        ms = (time.perf_counter() - started) * 1000
        mark = "" if changed else "   ⚠ 没有采用（降级回原文）"
        print(f"[{level:<6}] {ms:5.0f} ms{mark}\n  {result}\n", file=out)

    if args.show_prompt:
        print(f"\n--- system prompt ({levels[0]}) ---\n{polish_prompt(levels[0])}", file=out)
    return 0


def cmd_dictate(args, out, *, stt=None, polish=None) -> int:
    """Run the resident dictation daemon until interrupted."""
    from backend.daemon import DictationDaemon
    from backend.hotkey import HotkeyError

    config = load_config()
    if args.target:
        config.dictate_target = args.target
    if args.mode:
        config.hotkey_mode = args.mode

    if stt is None:
        try:
            stt = get_stt_provider(preferred=config.stt_provider)
        except NoProviderAvailable as exc:
            print(str(exc), file=out)
            return 1

    if polish is None:
        polish = build_polish(config)

    # Before anything expensive. A second instance steals half the hotkey
    # presses and half the microphone, and looks from the outside like a broken
    # model — which is where two evenings went.
    from backend.config import DEFAULT_DIR
    from backend.instance_lock import InstanceLock

    lock = InstanceLock(DEFAULT_DIR / "dictate.pid")
    owner = lock.acquire()
    if owner is not None:
        print(f"\n{owner.message()}\n", file=out)
        return 1

    def show(utterance):
        print(f"» {utterance.text}", file=out)
        if args.timing and daemon.last_timing is not None:
            print(daemon.last_timing.report(), file=out)

    overlay = None
    if not args.no_ui:
        from backend.overlay import Overlay

        overlay = Overlay()

    daemon = DictationDaemon(
        config=config,
        stt=stt,
        polish=polish,
        polish_factory=build_polish,  # so the menu's toggle actually does something
        on_text=show,
        overlay=overlay,
    )

    where = "光标处" if config.dictate_target == "cursor" else "暂存区（不注入）"
    toggle_label = (
        f"{config.hotkey_toggle}  "
        f"({'双击开、再双击停' if config.hotkey_toggle_double_tap else '按一次开、再按一次停'})"
        if config.hotkey_toggle
        else "未设置（`utter keys` 可找一个空闲键）"
    )
    print(
        f"Utter 听写已就绪   {_build_stamp()}\n"
        f"  按住说 {config.hotkey_push or '未设置'}\n"
        f"  长口述 {toggle_label}\n"
        f"  输出   {where}\n"
        f"  模型   {getattr(stt, 'display_name', stt)}\n"
        f"  润色   {'开（' + config.polish_level + '）' if config.polish_enabled else '关'}\n"
        f"  界面   {'浮窗 + 菜单栏' if not args.no_ui else '无（--no-ui）'}\n"
        f"  语言   {config.dictate_language or '自动检测'}"
        f"{'   ⚠ 自动检测每句多花约 0.9 秒；在 ~/Utter/config.json 里设 dictate_language 可省下' if not config.dictate_language else ''}\n"
        f"\n预热模型中…",
        file=out,
    )

    try:
        daemon.start()
    except HotkeyError as exc:
        print(f"\n{exc}", file=out)
        lock.release()
        return 1

    print("就绪。按住热键说话，Ctrl-C 退出。\n", file=out)
    try:
        if args.no_ui:
            daemon.run_forever()
        else:
            _run_with_ui(daemon)
    except KeyboardInterrupt:
        pass
    finally:
        daemon.stop()
        lock.release()

    if len(daemon.scratchpad):
        print(f"\n本次共 {len(daemon.scratchpad)} 段，已存档到 {daemon.scratchpad.archive.path}", file=out)
    return 0


def _prune_sessions(files, days: int, out) -> int:
    """Delete archives older than `days`. Says exactly what went.

    VoiceInk keeps transcripts until you delete them and offers auto-delete
    after 24 hours or 7 days; Superwhisper only added a retention setting in
    2026 and still has no bulk delete, so its users write cron jobs. Both store
    the audio as well, which is what actually fills a disk.

    Utter never writes audio (铁律 3), so this is housekeeping rather than a
    space problem — which is why it is a command the author runs, not a policy
    that deletes their words behind their back.
    """
    import time

    cutoff = time.time() - days * 86400
    doomed = [p for p in files if p.stat().st_mtime < cutoff]
    if not doomed:
        print(f"没有超过 {days} 天的存档，{len(files)} 个都留着。", file=out)
        return 0

    freed = sum(p.stat().st_size for p in doomed)
    segments = sum(1 for p in doomed for _ in p.open())
    for path in doomed:
        try:
            path.unlink()
        except OSError as exc:  # pragma: no cover - defensive
            print(f"  删不掉 {path.name}：{exc}", file=out)
    print(
        f"删掉了 {len(doomed)} 个存档（{segments} 段、{freed/1024:.0f} KB），"
        f"剩下 {len(files) - len(doomed)} 个。",
        file=out,
    )
    return 0


def cmd_sessions(args, out) -> int:
    """Read back what was dictated. The archive existed from day one but there
    was no way to look at it without knowing the file layout."""
    import json

    from backend.config import DEFAULT_DIR

    folder = DEFAULT_DIR / "sessions"
    files = sorted(folder.glob("session_*.jsonl"), reverse=True)
    if not files:
        print(f"还没有存档（会写到 {folder}）", file=out)
        return 0

    if getattr(args, "prune", None) is not None:
        return _prune_sessions(files, args.prune, out)

    if args.list:
        for path in files[: args.limit]:
            lines = sum(1 for _ in path.open())
            print(f"  {path.stem[8:]}  {lines} 段  {path}", file=out)
        total = sum(p.stat().st_size for p in files)
        segments = sum(1 for p in files for _ in p.open())
        # Printed because the author asked whether this grows without bound.
        # It does grow — but 铁律 3 keeps audio out of it, and text is three
        # orders of magnitude cheaper than the recordings other dictation apps
        # keep. The number is the argument.
        print(
            f"\n  共 {len(files)} 个存档、{segments} 段、{total/1024:.0f} KB"
            f"（平均每段 {total/max(segments,1):.0f} 字节，只有文字，没有录音）\n"
            f"  清理旧的：utter sessions --prune 30",
            file=out,
        )
        return 0

    path = files[0]
    print(f"{path}\n", file=out)
    for line in path.open():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a crash mid-write costs one line, not the file
        print(f"[{record['index']}] {record['text']}", file=out)
        if record.get("polished") and record["raw_text"] != record["text"]:
            # 铁律 10's audit trail is only useful if it can be read.
            print(f"    原文: {record['raw_text']}", file=out)
    return 0


def cmd_mics(args, out) -> int:
    """Open every input device and see whether sound actually arrives.

    `list_devices()` only asks CoreAudio what exists, and on this machine that
    answer is misleading: a speaker, two virtual recorders and a meeting app all
    advertise input channels, all open without error, and all deliver pure
    silence. Recommending one of them from the listing alone — which is exactly
    what happened on 2026-08-10 — left the author with a dictation tool that
    heard nothing.
    """
    import time

    import numpy as np

    from backend import audio_source

    devices = audio_source.list_devices()
    if not devices:
        print("找不到任何输入设备。", file=out)
        return 1

    print(f"逐个真开 {args.seconds:.1f} 秒，说点话再看结果：\n", file=out)
    working = []
    for device in devices:
        try:
            source = audio_source.MicSource(device_index=device.index).start()
            time.sleep(args.seconds)
            chunks = list(source.chunks())
            source.stop()
        except Exception as exc:
            print(f"  [{device.index}] {device.name:<34} ❌ 打不开：{str(exc)[:50]}", file=out)
            continue

        if not chunks:
            print(f"  [{device.index}] {device.name:<34} ❌ 开了但没有数据", file=out)
            continue

        peak = float(np.abs(np.concatenate(chunks)).max())
        if peak < 1e-4:
            print(f"  [{device.index}] {device.name:<34} ⚠ 全静音（不是真麦克风，或没收到声音）", file=out)
        else:
            mark = "  ←系统默认" if device.is_default else ""
            print(f"  [{device.index}] {device.name:<34} ✅ 峰值 {peak:.3f}{mark}", file=out)
            working.append(device)

    if working:
        print(f"\n可用。在 ~/Utter/config.json 里设 \"input_device\": {working[0].index}，"
              f"或留 null 跟随系统默认。", file=out)
    else:
        print("\n⚠ 没有一个设备录到声音。Mac mini 没有内置麦克风——"
              "确认蓝牙耳机已连接，或插一个 USB 麦克风。", file=out)
    return 0 if working else 1


def _run_with_ui(daemon) -> None:  # pragma: no cover - interactive
    """Hand the main thread to AppKit and let the daemon work underneath it.

    AppKit insists on the main thread, and so does its run loop, so the shell
    cannot be a side car to the sleep loop — it has to *be* the loop. The
    daemon's own work already lives on background threads, so nothing moves.

    NSApplicationActivationPolicyAccessory keeps Utter out of the Dock and out
    of ⌘Tab: it is a menu-bar tool, and a Dock icon for something you never
    switch to is clutter.
    """
    import signal

    import AppKit

    from backend.menubar import MenuBar

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    menu = MenuBar(daemon, on_quit=daemon.stop)
    menu.install()

    # Ctrl-C would otherwise be swallowed by the AppKit run loop.
    signal.signal(signal.SIGINT, lambda *_: app.terminate_(None))
    AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
        0.3, True, lambda _t: None
    )

    app.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="utter", description="Utter — local-first speech")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="what this machine is and what works on it")

    models_parser = sub.add_parser("models", help="model tiers and downloads")
    models_parser.add_argument("models_command", choices=["list"], nargs="?", default="list")

    transcribe = sub.add_parser("transcribe", help="run an audio file through the pipeline")
    transcribe.add_argument("file")
    transcribe.add_argument("--mode", choices=["listen", "dictate"], default="listen")
    transcribe.add_argument("--language", default=None)
    transcribe.add_argument("--timing", action="store_true", help="report per-utterance latency")

    mics = sub.add_parser("mics", help="open every input device and see which actually hears")
    mics.add_argument("--seconds", type=float, default=1.5)

    sessions = sub.add_parser("sessions", help="read back what was dictated")
    sessions.add_argument("--list", action="store_true", help="list sessions instead of printing the latest")
    sessions.add_argument("--limit", type=int, default=10)
    sessions.add_argument(
        "--prune",
        type=int,
        metavar="DAYS",
        help="delete archives older than DAYS days (no default — you name the number)",
    )

    key = sub.add_parser("key", help="store an API key in the Keychain (铁律 4)")
    key.add_argument("name", nargs="?", help="e.g. google_api_key; omit to list")
    key.add_argument("--forget", action="store_true", help="delete it instead")

    polish_cmd = sub.add_parser(
        "polish", help="run the polish prompt on text, to check it before trusting it"
    )
    polish_cmd.add_argument("text", nargs="?", help="text to polish (or pipe it in)")
    polish_cmd.add_argument("--level", choices=["light", "medium", "heavy"], default=None,
                            help="one level; omit to compare all three")
    polish_cmd.add_argument("--provider", default=None, help="override config.llm_provider")
    polish_cmd.add_argument("--show-prompt", action="store_true")

    keys = sub.add_parser("keys", help="find a hotkey nothing else has claimed")
    keys.add_argument("--seconds", type=float, default=60.0)

    dictate = sub.add_parser("dictate", help="run the resident dictation daemon")
    dictate.add_argument("--target", choices=["cursor", "scratchpad"], default=None)
    dictate.add_argument("--mode", choices=["push", "toggle"], default=None)
    dictate.add_argument("--no-ui", action="store_true",
                         help="不显示浮窗和菜单栏，纯终端运行")
    dictate.add_argument("--timing", action="store_true", help="print a latency breakdown per utterance")

    return parser


def main(argv=None, stdout=None, **overrides) -> int:
    out = stdout or sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command is None:
        parser.print_help(out)
        return 0
    if args.command == "doctor":
        return cmd_doctor(args, out)
    if args.command == "models":
        return cmd_models(args, out)
    if args.command == "transcribe":
        return cmd_transcribe(args, out, **overrides)
    if args.command == "mics":
        return cmd_mics(args, out)
    if args.command == "sessions":
        return cmd_sessions(args, out)
    if args.command == "key":
        return cmd_key(args, out)
    if args.command == "polish":
        return cmd_polish(args, out)
    if args.command == "keys":
        from backend import keyprobe

        return keyprobe.run(out, seconds=args.seconds)
    if args.command == "dictate":
        return cmd_dictate(args, out, **overrides)

    parser.print_help(out)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
