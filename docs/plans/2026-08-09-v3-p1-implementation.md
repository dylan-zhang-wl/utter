# Utter v3 — P1 Implementation Plan (Shared Core)

**Date:** 2026-08-09
**Design source:** `2026-08-09-v3-design.md` (authoritative — read §4, §5 before starting)
**Scope:** Backend only. No GUI work. Ends with a CLI that proves the pipeline end to end.
**REQUIRED SUB-SKILL:** Use `superpowers:executing-plans`.
**REQUIRED SUB-SKILL:** Use `superpowers:test-driven-development` for every task.

---

## Preflight

Read before Task 1:

1. `CLAUDE.md` at repo root — the seven 铁律 are binding. Two that bite hardest here:
   - Never call `mlx_whisper` outside a provider. MLX is Apple-Silicon-only and the author owns non-Apple-Silicon hardware.
   - Never add `torch`. VAD uses silero ONNX weights on `onnxruntime`.
2. `docs/plans/2026-08-09-v3-design.md` §3 — the measured numbers. Task 12 verifies against them.
3. Current branch is `v3`. All P1 work lands there.

**Environment (do this first, it is not a task):**

```bash
uv venv ~/.venvs/utter
~/.venvs/utter/bin/python -m pip --version   # sanity
```

Per global rules the venv lives outside the repo (this repo is inside iCloud). Never create `.venv/` in the project.

**Existing code:** v1's `backend/` modules stay in place during P1. `transcriber.py` is superseded by the provider layer but is not deleted until P3 rewires `main.py`. Do not edit `main.py` in P1.

**What P1 does NOT do:** no WebSocket changes, no frontend, no hotkeys, no cursor injection, no packaging. Those are P2–P4.

---

## Task 1 — Config that actually persists

v1's `PATCH /config` only mutated an in-memory pydantic object; the Settings page's Save button was an illusion. Fix the foundation first.

**Test first** — `backend/tests/test_config.py`:
- `save()` then `load()` round-trips a non-default value.
- `load()` on a missing file returns defaults, does not raise.
- The written file is mode `0600`.
- `openai_api_key` is **never** present in the written JSON.

Expected failure: `AttributeError: 'AppConfig' object has no attribute 'save'`

**Implement** — `backend/config.py`:
- Path `~/Utter/config.json`, created with `0600`.
- New fields: `stt_provider`, `model_tier`, `llm_provider`, `vad_silence_ms=600`, `vad_sensitivity=0.5`, `max_utterance_sec=30`.
- Secrets go through Task 2's keyring, never into the JSON.
- On load, if `~/LiveScribe/` exists and `~/Utter/` does not, migrate the directory (design §6).

**Commit:** `feat(config): persist to ~/Utter/config.json with 0600 and key exclusion`

---

## Task 2 — Secrets in Keychain

**Test first** — `backend/tests/test_secrets.py`, with `keyring` backend patched to an in-memory stub:
- `set_secret`/`get_secret` round-trip.
- `get_secret` for an unset name returns `None`, does not raise.
- `delete_secret` on an unset name is a no-op.

Expected failure: `ModuleNotFoundError: No module named 'backend.secrets'`

**Implement** — `backend/secrets.py`: thin wrapper over `keyring` with service name `com.dylan.utter`. Never log a value, not even truncated.

**Commit:** `feat(secrets): store API keys in macOS Keychain`

---

## Task 3 — Hardware detection

The model catalog needs to know what machine it is on before it can pick a format.

**Test first** — `backend/tests/test_hardware.py`, with `platform`/`torch`-free probes patched:
- Apple Silicon mac → `Hardware(kind="apple_silicon", ram_gb=16)`.
- Intel mac → `kind="cpu"`.
- Linux/Windows with NVIDIA present → `kind="cuda"`.
- Unknown/failed probe → `kind="cpu"` (safe default, never raises).

Expected failure: `ModuleNotFoundError: No module named 'backend.hardware'`

**Implement** — `backend/hardware.py`: `detect() -> Hardware`. Use `platform.machine()`/`platform.system()` for the Apple Silicon test; probe NVIDIA by checking `nvidia-smi` on PATH (cheap, no CUDA import). Cache the result.

**Commit:** `feat(hardware): detect apple_silicon / cuda / cpu for model selection`

---

## Task 4 — Model catalog: tiers to concrete models

Design §5.3. Users pick a tier; the catalog resolves it against hardware.

**Test first** — `backend/tests/test_catalog.py`:
- `resolve("balanced", Hardware("apple_silicon"))` → repo id `mlx-community/whisper-large-v3-turbo`, runtime `mlx`.
- The same tier on `Hardware("cpu")` → a non-MLX runtime.
- Every tier resolves on every hardware kind (no hole in the matrix — loop over the product).
- Each entry carries `label`, `blurb`, `size_mb` for the UI.
- An unknown tier raises `UnknownTier`.

Expected failure: `ModuleNotFoundError: No module named 'backend.catalog'`

**Implement** — `backend/catalog.py`: tiers `high` / `balanced` / `light` / `minimal`, mapped per design §5.3. Blurbs are the plain-language strings from the design — do not invent a "big = slow" gradient, `large-v3-turbo` is deliberately big *and* fast.

**Commit:** `feat(catalog): tier-to-model resolution per hardware kind`

---

## Task 5 — SttProvider protocol and registry

**Test first** — `backend/tests/test_stt_registry.py`, with two fake providers:
- A provider reporting `is_available() -> (False, reason)` is skipped by the registry.
- `get_stt_provider()` returns the first available provider in preference order.
- When none is available, it raises `NoProviderAvailable` carrying every reason.

Expected failure: `ModuleNotFoundError: No module named 'backend.providers'`

**Implement** — `backend/providers/__init__.py` + `backend/providers/stt.py`:

```python
class SttProvider(Protocol):
    id: str
    display_name: str
    def is_available(self) -> tuple[bool, str]: ...
    def transcribe(self, audio: np.ndarray, language: str | None) -> str: ...
```

Registry orders by hardware: `apple_silicon` → mlx first; `cuda`/`cpu` → faster-whisper first; cloud last always.

**Commit:** `feat(providers): SttProvider protocol with availability-aware registry`

---

## Task 6 — MlxWhisperProvider

**Test first** — `backend/tests/test_provider_mlx.py`, with `mlx_whisper` patched:
- `is_available()` is `False` with a readable reason on non-Apple-Silicon (patch `hardware.detect`).
- `transcribe` passes **`temperature=0.0`** to `mlx_whisper.transcribe` — assert on the kwarg. This is 铁律 1; the test exists to stop it regressing.
- `transcribe` returns the stripped `text` field.

Expected failure: `ModuleNotFoundError: No module named 'backend.providers.mlx'`

**Implement** — `backend/providers/mlx.py`. Import `mlx_whisper` lazily inside the method so the module imports fine on machines without it.

**Commit:** `feat(providers): MlxWhisperProvider with mandatory temperature=0.0`

---

## Task 7 — FasterWhisperProvider

Not deferred — the author owns non-Apple-Silicon hardware (design §9).

**Test first** — `backend/tests/test_provider_faster.py`, with `faster_whisper` patched:
- `is_available()` is `False` with a readable reason when the package is missing.
- `transcribe` passes `temperature=0.0` and joins segment texts in order.
- Device is `cuda` when hardware says cuda, else `cpu`.

Expected failure: `ModuleNotFoundError: No module named 'backend.providers.faster'`

**Implement** — `backend/providers/faster.py`. Reuse v1's parameter experience (`int8` on CPU). Lazy import as in Task 6.

**Commit:** `feat(providers): FasterWhisperProvider for cuda and cpu machines`

---

## Task 8 — Model downloader

**Test first** — `backend/tests/test_models.py`, with `huggingface_hub.snapshot_download` patched:
- `ensure_model(tier)` resolves through the catalog and returns a local path.
- An already-present model does **not** re-download — assert the patched call count is 0. (`large-v3-turbo` is already in `~/.cache/huggingface`; re-downloading 1.5G would be a real-world bug.)
- Progress callbacks fire.
- A failed download raises `ModelDownloadError` and leaves no partial directory registered.

Expected failure: `ModuleNotFoundError: No module named 'backend.models'`

**Implement** — `backend/models.py`. Delegate to `huggingface_hub`; do not hand-roll HTTP.

**Commit:** `feat(models): tier-aware downloader that reuses the local cache`

---

## Task 9 — VAD segmenter on ONNX

Design §5.4. **No torch.**

**Test first** — `backend/tests/test_vad.py`, using a synthetic fixture (tone burst / silence / tone burst, generated in the test — no binary fixture needed):
- Emits `speech_start` then `speech_end` at the expected sample offsets (±100 ms).
- Silence shorter than `vad_silence_ms` does **not** split an utterance.
- An utterance exceeding `max_utterance_sec` is force-finalized (design §5's error handling).
- Feeding pure silence emits nothing.

Expected failure: `ModuleNotFoundError: No module named 'backend.vad'`

**Implement** — `backend/vad.py`: `VadSegmenter` accepting 16 kHz float32 chunks, emitting `speech_start` / `speech_end` only. **No `partial` event** — design §3(b) killed it. Pure and synchronous: no asyncio, so it stays testable.

Add `onnxruntime` to requirements. Do **not** add `silero-vad` (it pulls torch).

**Commit:** `feat(vad): silero ONNX segmenter, utterance boundaries only`

---

## Task 10 — LLM providers

**Test first** — `backend/tests/test_llm.py`, with HTTP patched:
- `OllamaProvider.is_available()` is `False` with a readable reason when the daemon is unreachable.
- `OpenAICompatProvider` reads its key from `secrets`, never from config JSON.
- `translate()` and `polish()` each send their own prompt; assert they differ.
- A provider error surfaces as `LlmError`, and the caller can still finalize an entry without a result (design §5's error handling: never lose the transcript because translation failed).

Expected failure: `ModuleNotFoundError: No module named 'backend.providers.llm'`

**Implement** — `backend/providers/llm.py` + `ollama.py` + `openai_compat.py`. Keep v1's `deep_translator` path as `FreeTranslateProvider`, marked fallback-only — v1's lesson is in design §5.2.

**Commit:** `feat(providers): Ollama, OpenAI-compatible, and fallback translation`

---

## Task 11 — Pipeline assembly

The five slots of design §4, wired but mode-agnostic.

**Test first** — `backend/tests/test_pipeline.py`, with fake providers and a scripted VAD:
- A `listen`-mode pipeline calls `translate` on each finalized utterance and not on silence.
- A `dictate`-mode pipeline calls `polish`, never `translate`.
- Utterances are emitted in order with monotonic timestamps.
- An STT failure on one utterance is skipped without killing the pipeline.

Expected failure: `ModuleNotFoundError: No module named 'backend.pipeline'`

**Implement** — `backend/pipeline.py`: `Pipeline(source, segmenter, stt, post, sink)` plus two factory functions `listen_pipeline()` / `dictate_pipeline()`. The pipeline must not import mlx, faster_whisper, or any concrete provider.

**Commit:** `feat(pipeline): mode-agnostic five-stage pipeline with listen/dictate presets`

---

## Task 12 — CLI and latency verification

The point of P1: prove the core works before any GUI exists.

**Test first** — `backend/tests/test_cli.py`:
- `utter transcribe <file> --mode listen` prints utterances with translations.
- `--mode dictate` prints polished text.
- `utter models list` prints the four tiers with sizes.
- `utter doctor` prints detected hardware and each provider's availability with reasons.

Expected failure: `ModuleNotFoundError: No module named 'backend.cli'`

**Implement** — `backend/cli.py` with those four subcommands. File input rather than mic keeps it testable; mic comes in P2.

**Then run the real verification** (manual, not a unit test):

```bash
~/.venvs/utter/bin/python -m backend.cli transcribe docs/benchmarks/bench60.wav --mode listen --timing
```

**Acceptance:** per-utterance transcription time must land near the design §3 figures (~1.0–1.2 s for 2–15 s utterances on this M2). If it is materially slower, stop and diagnose before P2 — something in the wiring is wrong, because the raw model was measured at these numbers.

Record the result in `docs/benchmarks/` as a second data point alongside the raw-model numbers.

**Commit:** `feat(cli): transcribe/models/doctor commands and pipeline latency check`

---

## Done criteria

- `utter doctor` correctly reports hardware and provider availability on this M2.
- `utter transcribe` produces sensible utterances from the benchmark file in both modes.
- Measured per-utterance latency matches design §3.
- No `torch` in the dependency tree — verify with `~/.venvs/utter/bin/pip list | grep -i torch` returning nothing.
- No concrete provider imported outside `backend/providers/`.

Then P2: dictation mode — global hotkey, push-to-talk segmentation, cursor injection.
