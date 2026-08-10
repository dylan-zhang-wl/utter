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
    except Exception as exc:  # pragma: no cover - defensive
        checks["麦克风"] = (False, str(exc))

    return checks


def _llm_providers():
    """Instantiated defensively — a broken optional dependency should degrade
    one line of `doctor`, not the whole command."""
    found = []
    for module_name, class_name in (
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

    parser.print_help(out)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
