# Utter v3 — P2a Implementation Plan (Minimal Usable Dictation)

**Date:** 2026-08-10
**Design source:** `2026-08-09-v3-design.md` §4.1 — read it before starting, it is the spec
**Predecessor:** P1 complete and accepted, see `2026-08-10-p1-build-log.md`
**Scope:** Headless. No GUI, no Tauri, no overlay. Ends with a resident daemon the author uses to write.
**REQUIRED SUB-SKILL:** `superpowers:test-driven-development` for every task.

---

## What P2a is for

P1 built an engine and proved it runs at 1.18s per utterance. Nobody can dictate
with it. P2a is the smallest thing that makes it usable while writing a paper,
and nothing more.

The ordering below was agreed with the author on 2026-08-10 and has one
non-obvious property: **timing instrumentation lands before injection**, so
every later step is measurable rather than a matter of impression.

---

## Preflight

1. `CLAUDE.md` 铁律 8–13 are binding here. This is the phase they were written
   for. The one most likely to be violated by well-meaning code is **铁律 9 —
   injected text is never rewritten.**
2. Design §4.1 is the spec, in particular (c) three fallback lines, (e) window
   locking, (f) scratchpad mode.
3. Branch `v3`.

### Architectural decision recorded here

**The Python daemon owns everything functional. The future Tauri window is a
viewport and nothing else.**

Hotkey, capture, transcription, injection and archiving all live in the daemon.
When P2b adds the overlay it talks to the daemon over WebSocket and displays
what it is told.

The reason is v1's failure mode. In v1 the GUI was mandatory, the sidecar was
broken, and the whole tool therefore needed two terminals and went unused. If
the daemon is self-sufficient, a broken Tauri build costs the overlay and
nothing else — dictation keeps working. That inverts the dependency that killed
v1.

### Known facts about this machine, checked 2026-08-10

- **Default input is "Redmi 电脑音箱", a Bluetooth/USB speaker. No built-in
  MacBook microphone appears in the device list at all.** Bluetooth microphones
  often negotiate a call-quality profile, which will degrade transcription in a
  way that looks like a bad model. Device choice must therefore be configurable
  and `utter doctor` must print which device is actually in use.
- 16 kHz mono float32 is supported natively on the default device — no
  resampling stage needed.
- v1's `backend/audio_capture.py` uses `pyaudio`, which is **not** in
  requirements.txt (P1 chose `sounddevice`). That file is dead code for our
  purposes; write a new source rather than reviving it.
- `pynput` costs ~30 MB, over half of it PyObjCTest shipped inside pyobjc-core.
  Acceptable against a 653 MB environment, but it is a new dependency and needs
  recording in requirements.txt.

### The permission that will bite

macOS requires **Accessibility** permission for both global hotkey capture and
synthetic keystrokes. Without it, `pynput`'s listener receives nothing **and
raises nothing** — the silent failure this project's rules exist to prevent.

So Task 2 must detect it explicitly via `AXIsProcessTrusted()` and refuse to
start with a clear message. A dictation tool that appears to run and quietly
never hears the hotkey is worse than one that fails at launch.

Microphone permission triggers its own dialog on first stream open. Expected,
one-off, and the author is present.

---

## Task 1 — Microphone into the pipeline

**Test first** — `backend/tests/test_audio_source.py`, with `sounddevice` patched:
- yields 16 kHz mono float32 chunks of the configured size.
- stereo input is downmixed to mono.
- a non-default `device_index` from config is honoured.
- `list_devices()` returns input-capable devices only, marking the default.
- a device that disappears mid-stream (Bluetooth drops) surfaces as a clean
  stop with a readable reason, not a traceback from inside PortAudio.
- opening a device that does not exist raises `AudioSourceError` naming it.
- the source is a context manager and always closes the stream, including on
  exception.

Expected failure: `ModuleNotFoundError: No module named 'backend.audio_source'`

**Implement** — `backend/audio_source.py`: `MicSource` wrapping
`sounddevice.InputStream` with a callback feeding a bounded `queue.Queue`.

Bounded deliberately: if the consumer stalls, drop the oldest chunk and count
it. An unbounded queue turns a transient stall into unbounded memory and a
growing lag the user cannot see. The drop count belongs in the timing report.

Add `sounddevice` device selection to config (`input_device: int | None = None`,
None meaning system default).

**Commit:** `feat(audio): 16kHz mono microphone source with bounded buffering`

---

## Task 2 — Global hotkey, both segmentation modes

**Test first** — `backend/tests/test_hotkey.py`, with `pynput` patched:
- push-to-talk: key down starts, key up stops, and the two events carry
  timestamps.
- toggle: first press starts, second press stops; a third starts again.
- the combination is configurable and a malformed one raises at construction,
  not at first press.
- an unavailable Accessibility permission is reported by `is_available()` with
  an actionable reason and does **not** start a listener.
- releasing a key that was never pressed is ignored rather than emitting a stop.
- the listener is stopped on `close()`, including after an exception.

Expected failure: `ModuleNotFoundError: No module named 'backend.hotkey'`

**Implement** — `backend/hotkey.py`. `pynput.keyboard.Listener` with explicit
modifier tracking; `GlobalHotKeys` is not enough because it only fires on
activation and push-to-talk needs the release too.

Probe Accessibility with `AXIsProcessTrusted()` from
`ApplicationServices` (arrives with pyobjc, which pynput already requires).

Add to config: `hotkey` (default to be chosen in Task 6 after checking for
conflicts), `hotkey_mode` (`"push"` | `"toggle"`).

Extend `utter doctor` with an Accessibility line.

**Commit:** `feat(hotkey): global push-to-talk and toggle with permission probe`

---

## Task 3 — Stage timing

Before injection, so every later step is measurable. "It feels slow" cannot be
fixed; "injection took 800ms" can.

**Test first** — `backend/tests/test_timing.py`:
- records each stage and reports a breakdown in order.
- a stage that is skipped (polish off) is absent, not zero.
- the report includes the total and the dropped-chunk count from Task 1.
- timings are monotonic-clock based, not wall clock.
- a stage that raises still closes its span.

Expected failure: `ModuleNotFoundError: No module named 'backend.timing'`

**Implement** — `backend/timing.py`: a small `Stopwatch` with `span(name)`
context manager, using `time.perf_counter`.

Target shape, printed after each dictation when `--timing` is on:

```
hotkey release -> buffer closed      8 ms
transcription                     1180 ms
polish                             (off)
clipboard + paste                   45 ms
─────────────────────────────────────────
total                             1233 ms
```

**Acceptance targets** (design conversation, 2026-08-10): ≤1.5 s without polish,
≤3 s with. Above 3 s the user starts doubting whether it worked at all.

**Commit:** `feat(timing): per-stage latency breakdown for the dictation path`

---

## Task 4 — Scratchpad mode

Design §4.1f. No window binding, no injection: text accumulates and the author
decides where it goes. Cheapest thing that is genuinely useful, and it works
before injection exists.

**Test first** — `backend/tests/test_scratchpad.py`:
- utterances accumulate in order.
- `take()` returns everything and empties the buffer.
- `peek()` does not empty it.
- both raw and processed text are retained per entry (铁律 10).
- entries are appended to the session archive as they arrive, not at the end —
  a crash must not cost ten minutes of dictation (铁律 8, design §4.1c line 2).
- the archive file is written under `~/Utter/sessions/` with 0600.

Expected failure: `ModuleNotFoundError: No module named 'backend.scratchpad'`

**Implement** — `backend/scratchpad.py`, plus session archiving shared with
listen mode later.

**Commit:** `feat(scratchpad): unbound dictation buffer with incremental archiving`

---

## Task 5 — Cursor injection

**Test first** — `backend/tests/test_injection.py`, with pynput and the
pasteboard patched:
- long text goes via clipboard paste, never per-character typing (铁律 12).
- the user's prior clipboard is restored afterwards, including a non-text
  flavour such as an image — restoring only `public.utf8-plain-text` silently
  destroys whatever they had copied.
- the target application is captured at dictation start and injection
  re-activates it first (design §4.1e).
- when focus has moved elsewhere, text is **buffered, not injected**, and the
  pending count is reported.
- returning to the target flushes the buffer.
- a target that has quit falls back to leaving the text on the clipboard plus a
  notification, and never raises (design §4.1c line 3).
- injection failure never loses text — the entry stays in the scratchpad.
- nothing already injected is ever re-injected or edited (铁律 9): a test feeds
  the same utterance index twice and asserts one injection.

Expected failure: `ModuleNotFoundError: No module named 'backend.injection'`

**Implement** — `backend/injection.py`. `NSPasteboard` via pyobjc for
save/restore across all flavours; `pynput` for the ⌘V keystroke.

**Explicitly do not implement** `CGEventPostToPid` or direct AX value setting.
Design §4.1e rejected both: they fail randomly on Electron, browsers and
terminals, and random failure is worse than clear failure because the user
cannot tell which segment was lost.

**Commit:** `feat(injection): clipboard paste with target locking and buffering`

---

## Task 6 — The daemon

**Test first** — `backend/tests/test_daemon.py`, with every edge faked:
- start, dictate once, stop — the full chain runs in order.
- the model is warmed at startup, so the first real utterance does not pay the
  2.76 s load measured in P1.
- a second hotkey press during transcription is queued, not dropped.
- SIGINT shuts down cleanly, flushing the scratchpad and restoring the clipboard.
- polish stays off unless `config.polish_enabled`.

Expected failure: `ModuleNotFoundError: No module named 'backend.daemon'`

**Implement** — `backend/daemon.py` and a `utter dictate` CLI subcommand.
Resident by design: a per-use launch pays the model load every time.

Choose the default hotkey here, after checking it against macOS system
shortcuts and the author's common applications. Record the reasoning.

**Commit:** `feat(daemon): resident dictation daemon with startup warm-up`

---

## Task 7 — Application compatibility table

Manual, and required before P2a is called done. Injection behaves differently
per application and the author must know which of their tools work before
building a workflow on one that does not.

Test each with a short dictation, recording success and the injection time from
Task 3:

| Application | Kind |
|---|---|
| Notes / Pages | native |
| Microsoft Word | native, complex |
| Obsidian | Electron |
| VS Code | Electron |
| Safari, Chrome text fields | web view |
| Terminal | special |
| Mail | native |

Record in `docs/benchmarks/2026-08-XX-injection-compatibility.md`, with the date
and macOS version. Note any application that needs the clipboard fallback.

**Commit:** `docs(benchmarks): injection compatibility across target applications`

---

## Task 8 — Wire polish, and verify the prompt against a real model

**Blocked on the author.** Needs either Ollama installed and running, or an API
key in the Keychain. Ask at this point, not before — every earlier task runs
without an LLM.

The wiring itself is small: `Pipeline` already accepts a `polish` callable and
P1's providers are written and unit-tested. What has never happened is a single
real call.

**Test first:**
- polish off by default; enabling it in config wires the configured provider.
- a provider that is configured but unavailable falls back to raw text and warns
  (铁律 8), and the daemon still starts.
- the timing report gains a polish stage.

**Then the verification that actually matters** — take three real dictated
passages from the author, run each at all three levels, and diff against the raw
transcript. Check specifically:
- was any wording changed? (铁律 10 — this is a failure, not a preference)
- was anything added or removed beyond filler?
- did "medium" handle a genuine self-correction sensibly?

Record verbatim in `docs/benchmarks/`. If the model rewords under a prompt that
forbids rewording, the prompt is not adequate and the default stays off until it
is.

**Commit:** `feat(dictate): optional polish wired behind config, prompt verified`

---

## Done criteria

- The author dictates a paragraph into a real document with one hotkey press.
- Measured end-to-end latency ≤1.5 s without polish.
- The first utterance after daemon start is not slower than the rest.
- The clipboard is intact after every dictation, including a copied image.
- No text is lost when focus moves mid-dictation.
- The compatibility table exists and names any application that does not work.

Then P2b: toggle-mode long-form, paragraph batching, the overlay, and the
sidecar fix.
