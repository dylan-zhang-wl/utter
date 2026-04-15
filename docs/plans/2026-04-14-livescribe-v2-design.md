# LiveScribe v2 Design

**Date:** 2026-04-14
**Builds on:** `2026-04-13-livescribe-design.md`

## Motivation

v1 is functional end-to-end but feels rough in use:

1. **Transcripts arrive in small fragments** — fixed 3/8-second audio buffer cuts across natural pauses, producing many short, often half-finished entries.
2. **Segments occasionally duplicate** — an artifact of React StrictMode opening two WebSocket connections in dev; patched but worth hardening.
3. **Accuracy is mediocre for non-native speakers** — `base` model is small; upgrading to `small` helps meaningfully on M2.
4. **No way to fix Whisper errors on the fly** — transcript is read-only.
5. **No in-session controls** — missing pause/resume, clear, rename, manual save, export, and history.

v2 reshapes the audio pipeline around voice activity detection (VAD), adds inline editing, and fills in the missing session management UX.

## Decisions

These were settled during brainstorming:

| Decision | Chosen |
|---|---|
| Editable fields | Both English transcript and Chinese translation |
| Session model | Lecture-scoped: one session per Start/Stop cycle, with Pause/Resume in between |
| Streaming display | Live partial transcription while speaking; finalize + translate on pause |

## Architecture

### Audio pipeline (new)

```
mic (100ms chunks, 16kHz float32)
    │
    ▼
VadSegmenter  ─────┐ emits events:
                   │   speech_start
                   │   partial   (every ~1s while speaking)
                   │   speech_end (after 600ms silence)
    │
    ▼
process_audio_loop
    │
    ├─ on "partial" → transcriber.transcribe_partial(audio)  (fast, beam=1)
    │                 → broadcast {type: "partial", entry_id, text}
    │
    └─ on "speech_end" → transcriber.transcribe(audio)  (accurate, beam=5)
                         → translator.translate(text)
                         → session.add_entry(...)
                         → broadcast {type: "final", entry_id, text, translation}
```

**Key tunables (in config, exposed in Settings page):**

- `vad_silence_ms`: 600 (how long silence before we finalize)
- `vad_sensitivity`: 0.5 (silero-vad threshold)
- `partial_interval_ms`: 1000 (how often to re-transcribe mid-utterance)

### Editable entries

- Every entry has `id` (UUID), `text`, `translation`, `edited: {text: bool, translation: bool}`.
- Edits propagate frontend → WS `{action: "edit", id, field, value}` → `session.update_entry` → broadcast to other clients.
- Editing English does **not** auto-retranslate. A ↻ button next to the translation lets the user opt in.
- The live partial entry is **not editable**. Editing becomes available only after `final`.

### Session model

A session is one continuous "lecture" from Start to Stop. Pause/Resume does not break session continuity.

```
start  → StartDialog lets user name the session (default: "YYYY-MM-DD HH:MM")
         session = Session(name, started_at, entries=[])
         audio capture begins; VAD loop begins

pause  → audio capture stops, session remains active, entries preserved

resume → audio capture resumes, feeding VAD loop

clear  → confirm dialog → session.entries = []; saved file (if exists) truncated

rename → updates session.name; reflected in filename on next save

edit   → session.update_entry(id, field, value)

stop   → final save; session closes; UI returns to idle
```

**Auto-save:** every 30 seconds, plus 500ms debounce after each edit, plus immediately on Pause/Stop. File is `~/LiveScribe/sessions/{id}.json` during the session; on Stop, renamed to `{slug(name)}_{YYYY-MM-DD}.json`.

### Frontend state shape

```ts
type Entry = {
  id: string;
  text: string;
  translation: string;
  startTime: number;
  endTime: number;
  edited: { text: boolean; translation: boolean };
};

type AppState = {
  connected: boolean;
  recording: boolean;
  paused: boolean;
  sessionName: string;
  sessionStartedAt: number | null;
  elapsedSec: number;               // derived from started_at and paused intervals
  entries: Entry[];                 // finalized entries
  partialEntry: { id: string; text: string } | null;  // streaming in-progress
  audioSource, translationEngine, displayMode;       // as v1
};
```

## Component changes

### Backend (new/changed)

| File | Change |
|---|---|
| `backend/vad.py` **(new)** | `VadSegmenter` class wraps silero-vad; accepts 16kHz float32 chunks; emits `speech_start` / `partial` / `speech_end` events. Pure — no asyncio, testable with fixture audio. |
| `backend/transcriber.py` | Add `transcribe_partial(audio)` — beam_size=1, no VAD filter, faster. Keep `transcribe` at beam_size=5 for finals. |
| `backend/session.py` | Entries get UUID `id`; add `update_entry(id, field, value)`, `rename(name)`, `clear()`; add debounced `auto_save()`. |
| `backend/main.py` | Replace 3/8-second buffer loop with VAD-driven loop. New WS actions: `edit`, `pause`, `resume`, `clear`, `rename`. New WS message types: `partial`, `final` (replaces generic `transcript`). |
| `backend/config.py` | Add `vad_silence_ms`, `vad_sensitivity`, `partial_interval_ms`. |

### Frontend (new/changed)

| File | Change |
|---|---|
| `store/appStore.ts` | Add `paused`, `sessionName`, `sessionStartedAt`, `elapsedSec`, `partialEntry`. Add `updateEntry`, `deleteEntry`, `setPartialEntry`, `finalizePartial`, `setSessionName`. |
| `hooks/useWebSocket.ts` | Handle `partial` → `setPartialEntry`; `final` → `finalizePartial` + `addEntry`. Wrap new actions: `edit`, `pause`, `resume`, `clear`, `rename`. |
| `components/TranscriptEntry.tsx` **(new)** | Single-entry card. Hover shows pencil. Double-click or pencil → inline `<textarea>`. Enter/blur saves; Esc cancels. Edited dot indicator. ↻ retranslate button. |
| `components/TranscriptView.tsx` | Simplified to map over entries + render `partialEntry` as greyed trailing row. |
| `components/ControlBar.tsx` | Replace single Start/Stop with Start → (Pause \| Stop). Add Clear (with confirm) and Export dropdown. |
| `components/SessionHeader.tsx` **(new)** | Editable session title; live elapsed timer. |
| `components/StartDialog.tsx` **(new)** | Modal on Start: name input with default, source/engine quick-set. |
| `components/History.tsx` **(new)** | Right-side drawer toggled from header. Lists saved sessions; click opens read-only view with export/delete. |

## Error handling

- **VAD false negatives (never detects silence)**: hard cap on utterance length (e.g., 30s) — force finalize.
- **Transcription fails mid-partial**: silently skip that partial tick; keep accumulating audio.
- **Translation fails**: entry is finalized with empty translation and a retry button appears (same ↻ button).
- **Auto-save fails (disk full, permission)**: toast notification, session continues in memory.
- **WS disconnect during session**: session state lives on backend; frontend reconnects and re-fetches current entries via a new `/sessions/current` endpoint.

## Testing approach

- `vad.py`: unit test with pre-recorded WAV fixtures — speech_start/end events fire at expected timestamps (±100ms).
- `transcriber.partial`: confirm it returns something plausible for a 1-second clip and is faster than `transcribe`.
- `session.update_entry/rename/clear`: unit tests on in-memory Session.
- Backend WS: integration test with a FastAPI `TestClient` WebSocket — start → fake audio → assert partial + final messages.
- Frontend: focus on `useWebSocket` message handling + store updates (jest + RTL).
- Manual end-to-end: a 2-minute spoken lecture, verify streaming, edit, pause/resume, export.

## Out of scope (YAGNI)

- Cloud sync, user accounts
- Multi-session tabs
- Noise suppression / audio effects (macOS system-level handles this adequately)
- Speaker diarization
- Multi-language source (only EN → ZH for now)
- Mobile / Windows support

## Rollout

One branch, one PR, implement per the accompanying implementation plan. No feature-flagging — v2 is a clean replacement for v1's pipeline. v1 sessions on disk are JSON-compatible (v2 just adds `id` field, which can be backfilled on load).
