# LiveScribe v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rework LiveScribe from a fixed-buffer transcription prototype into a pause-driven streaming tool with editable entries, proper session management, and export.

**Architecture:** Replace the 3/8-second audio buffer with a silero-vad driven segmenter that emits `speech_start` / `partial` / `speech_end` events. Add entry UUIDs so edits have a target. Introduce a lecture-scoped Session with pause/resume/rename/clear/auto-save. Frontend gains inline editing, a streaming "partial" row, a session header with timer, a start dialog, and a history drawer.

**Tech Stack:** Python 3.11 · FastAPI · silero-vad · faster-whisper · deep-translator · React 19 · TypeScript · Zustand · Tauri v2.

**Companion design doc:** `docs/plans/2026-04-14-livescribe-v2-design.md`

---

## Preflight

Before starting Task 1, make sure the earlier hot-fix commit from 2026-04-14 night (increased buffer to 8s, whisper model bumped to small, WS reconnect hardened, tauri.conf.json externalBin removed) is either merged or reverted. Task 3 will re-remove the fixed buffer entirely, so that change becomes moot. Keep the whisper model at `small` and the useWebSocket cancelled-flag fix — both stay.

Confirm:
```bash
cd "/Users/d/Desktop/华工大/cc cowork/live voice transform"
git status        # should be clean
git log --oneline -5
```

---

## Task 1: Add VAD config fields

**Files:**
- Modify: `backend/config.py`
- Test: `backend/tests/test_config.py`

**Step 1: Write the failing test**

Append to `backend/tests/test_config.py`:

```python
def test_config_has_vad_defaults():
    c = AppConfig()
    assert c.vad_silence_ms == 600
    assert c.vad_sensitivity == 0.5
    assert c.partial_interval_ms == 1000
```

**Step 2: Run test — expect fail**

```bash
cd "/Users/d/Desktop/华工大/cc cowork/live voice transform"
source .venv/bin/activate
pytest backend/tests/test_config.py::test_config_has_vad_defaults -v
```

Expected: FAIL (AttributeError).

**Step 3: Add the fields**

In `backend/config.py`, inside `AppConfig`:

```python
    vad_silence_ms: int = 600
    vad_sensitivity: float = 0.5
    partial_interval_ms: int = 1000
```

**Step 4: Re-run — expect pass**

**Step 5: Commit**

```bash
git add backend/config.py backend/tests/test_config.py
git commit -m "feat(config): add VAD + partial interval tunables"
```

---

## Task 2: VAD segmenter module

**Files:**
- Create: `backend/vad.py`
- Test: `backend/tests/test_vad.py`

The segmenter is a pure synchronous class. It accepts 16kHz float32 chunks (any size) and calls a callback with events. No asyncio here — the caller wires it into the asyncio loop.

**Step 1: Write failing tests**

Create `backend/tests/test_vad.py`:

```python
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from backend.vad import VadSegmenter


def _silence(seconds: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sr * seconds), dtype=np.float32)


def _fake_speech(seconds: float, sr: int = 16000) -> np.ndarray:
    # low-freq sine at -6dB, enough to trip any VAD threshold
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


class FakeVadModel:
    """Deterministic stand-in for silero-vad: returns 0.9 for non-silent, 0.0 for silent."""
    def __call__(self, chunk, sr):
        import torch
        rms = float(np.sqrt(np.mean(chunk.numpy() ** 2)))
        return torch.tensor(0.9 if rms > 0.01 else 0.0)
    def reset_states(self):
        pass


def test_emits_speech_start_then_end():
    events = []
    seg = VadSegmenter(
        on_event=lambda e: events.append(e),
        model=FakeVadModel(),
        silence_ms=300,
        sensitivity=0.5,
        partial_interval_ms=10_000,  # suppress partials for this test
    )
    seg.feed(_silence(0.2))
    seg.feed(_fake_speech(1.0))
    seg.feed(_silence(0.5))

    kinds = [e["event"] for e in events]
    assert "speech_start" in kinds
    assert "speech_end" in kinds
    assert kinds.index("speech_start") < kinds.index("speech_end")


def test_emits_partial_during_speech():
    events = []
    seg = VadSegmenter(
        on_event=lambda e: events.append(e),
        model=FakeVadModel(),
        silence_ms=300,
        sensitivity=0.5,
        partial_interval_ms=500,
    )
    seg.feed(_fake_speech(1.6))

    partials = [e for e in events if e["event"] == "partial"]
    assert len(partials) >= 2
    # partial payload must contain accumulated audio
    assert partials[0]["audio"].dtype == np.float32
    assert len(partials[0]["audio"]) > 0


def test_silence_without_speech_emits_nothing():
    events = []
    seg = VadSegmenter(
        on_event=lambda e: events.append(e),
        model=FakeVadModel(),
        silence_ms=300,
        sensitivity=0.5,
        partial_interval_ms=500,
    )
    seg.feed(_silence(2.0))
    assert events == []


def test_speech_end_carries_full_utterance():
    events = []
    seg = VadSegmenter(
        on_event=lambda e: events.append(e),
        model=FakeVadModel(),
        silence_ms=300,
        sensitivity=0.5,
        partial_interval_ms=10_000,
    )
    seg.feed(_fake_speech(1.0))
    seg.feed(_silence(0.5))

    end = next(e for e in events if e["event"] == "speech_end")
    # full utterance is ~1s @ 16kHz ≈ 16000 samples
    assert 14_000 <= len(end["audio"]) <= 18_000
```

**Step 2: Run tests — expect fail** (`ModuleNotFoundError: backend.vad`).

**Step 3: Implement `backend/vad.py`**

```python
"""Voice-activity-driven segmenter using silero-vad.

The segmenter consumes audio chunks and emits three kinds of events to a
callback: speech_start (no audio), partial (accumulating audio every
partial_interval_ms while speaking), speech_end (full utterance audio).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # silero-vad expects 512-sample frames at 16kHz
MAX_UTTERANCE_SEC = 30


@dataclass
class _State:
    in_speech: bool = False
    silence_samples: int = 0
    samples_since_partial: int = 0
    buffer: list = None  # list[np.ndarray]


def _load_silero():
    import torch
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
        onnx=False,
    )
    return model


class VadSegmenter:
    def __init__(
        self,
        on_event: Callable[[dict], None],
        model: Any = None,
        silence_ms: int = 600,
        sensitivity: float = 0.5,
        partial_interval_ms: int = 1000,
    ):
        self._on_event = on_event
        self._model = model if model is not None else _load_silero()
        self._silence_samples_threshold = int(SAMPLE_RATE * silence_ms / 1000)
        self._partial_samples_threshold = int(SAMPLE_RATE * partial_interval_ms / 1000)
        self._max_utterance_samples = SAMPLE_RATE * MAX_UTTERANCE_SEC
        self._sensitivity = sensitivity
        self._state = _State(buffer=[])
        self._residual = np.zeros(0, dtype=np.float32)

    def feed(self, chunk: np.ndarray) -> None:
        """Feed a chunk of 16kHz float32 audio. Any size; will be split into frames."""
        import torch
        audio = np.concatenate([self._residual, chunk.astype(np.float32, copy=False)])
        n_frames = len(audio) // FRAME_SAMPLES
        consumed = n_frames * FRAME_SAMPLES
        self._residual = audio[consumed:]
        frames = audio[:consumed].reshape(n_frames, FRAME_SAMPLES)

        for frame in frames:
            score = float(self._model(torch.from_numpy(frame), SAMPLE_RATE))
            is_speech = score >= self._sensitivity
            self._handle_frame(frame, is_speech)

    def _handle_frame(self, frame: np.ndarray, is_speech: bool) -> None:
        s = self._state
        if is_speech:
            if not s.in_speech:
                s.in_speech = True
                s.buffer = []
                s.samples_since_partial = 0
                self._on_event({"event": "speech_start"})
            s.buffer.append(frame)
            s.silence_samples = 0
            s.samples_since_partial += FRAME_SAMPLES

            total = sum(len(f) for f in s.buffer)
            if s.samples_since_partial >= self._partial_samples_threshold:
                s.samples_since_partial = 0
                self._on_event({
                    "event": "partial",
                    "audio": np.concatenate(s.buffer),
                })
            if total >= self._max_utterance_samples:
                self._finalize()
        else:
            if s.in_speech:
                s.buffer.append(frame)
                s.silence_samples += FRAME_SAMPLES
                if s.silence_samples >= self._silence_samples_threshold:
                    self._finalize()

    def _finalize(self) -> None:
        s = self._state
        if not s.buffer:
            s.in_speech = False
            return
        audio = np.concatenate(s.buffer)
        self._on_event({"event": "speech_end", "audio": audio})
        self._state = _State(buffer=[])
```

**Step 4: Re-run tests — expect pass**

```bash
pytest backend/tests/test_vad.py -v
```

**Step 5: Commit**

```bash
git add backend/vad.py backend/tests/test_vad.py
git commit -m "feat(vad): silero-vad-driven segmenter emitting start/partial/end events"
```

---

## Task 3: Transcriber partial mode

**Files:**
- Modify: `backend/transcriber.py`
- Test: `backend/tests/test_transcriber.py`

**Step 1: Add failing test**

Append to `backend/tests/test_transcriber.py`:

```python
def test_transcribe_partial_uses_beam_size_1():
    from unittest.mock import MagicMock, patch
    with patch("backend.transcriber.WhisperModel") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = ([], MagicMock())
        mock_cls.return_value = mock_instance

        from backend.transcriber import Transcriber
        t = Transcriber(model_size="base")
        import numpy as np
        t.transcribe_partial(np.zeros(16000, dtype=np.float32))

        _, kwargs = mock_instance.transcribe.call_args
        assert kwargs.get("beam_size") == 1
```

**Step 2: Run — expect fail**

**Step 3: Add method**

In `backend/transcriber.py`, add:

```python
    def transcribe_partial(self, audio: np.ndarray) -> str:
        """Fast, lower-accuracy transcription for streaming partials."""
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segments)
```

**Step 4: Re-run — expect pass**

**Step 5: Commit**

```bash
git add backend/transcriber.py backend/tests/test_transcriber.py
git commit -m "feat(transcriber): add transcribe_partial for streaming updates"
```

---

## Task 4: Session: id, update_entry, rename, clear, auto_save

**Files:**
- Modify: `backend/session.py`
- Test: `backend/tests/test_session.py`

**Step 1: Add failing tests**

Append to `backend/tests/test_session.py`:

```python
def test_add_entry_returns_id_and_sets_it():
    s = Session(audio_source="microphone", save_dir="/tmp/ls-test")
    eid = s.add_entry("Hello", "你好", 0.0, 1.0)
    assert isinstance(eid, str) and len(eid) > 0
    assert s.entries[0]["id"] == eid
    assert s.entries[0]["edited"] == {"text": False, "translation": False}


def test_update_entry_text_flags_edited():
    s = Session(audio_source="microphone", save_dir="/tmp/ls-test")
    eid = s.add_entry("Hello", "你好", 0.0, 1.0)
    s.update_entry(eid, "text", "Hi there")
    assert s.entries[0]["text"] == "Hi there"
    assert s.entries[0]["edited"]["text"] is True
    assert s.entries[0]["edited"]["translation"] is False


def test_rename_updates_name():
    s = Session(audio_source="microphone", save_dir="/tmp/ls-test")
    s.rename("Econ Lec 5")
    assert s.name == "Econ Lec 5"


def test_clear_empties_entries():
    s = Session(audio_source="microphone", save_dir="/tmp/ls-test")
    s.add_entry("a", "b", 0, 1)
    s.clear()
    assert s.entries == []


def test_save_uses_slugified_name(tmp_path):
    s = Session(audio_source="microphone", save_dir=str(tmp_path), name="Econ Lec 5")
    s.add_entry("a", "b", 0, 1)
    path = s.save(finalize=True)
    assert "econ-lec-5" in path.lower()
    assert path.endswith(".json")
```

**Step 2: Run — expect fail**

**Step 3: Rewrite `backend/session.py`**

Read the existing file first to preserve unrelated structure, then apply:

- add `import uuid`, `import re`
- `Session.__init__` accepts `name: str | None = None` → default `datetime.now().strftime("%Y-%m-%d %H:%M")`
- `self.id = str(uuid.uuid4())`
- `add_entry(...)` generates an id, appends `{"id", "text", "translation", "start_time", "end_time", "edited": {"text": False, "translation": False}}`, returns the id
- `update_entry(self, id_: str, field: str, value: str)` — locate by id, set field, set `edited[field] = True`
- `rename(self, name: str)`
- `clear(self)` — entries = []
- helper `_slugify(s)` — lowercase, non-alphanum → `-`, trim
- `save(finalize=False)` — during session use `{id}.json`; if `finalize=True`, write under `{slug(name)}_{YYYY-MM-DD}.json` and remove the temp file

**Step 4: Re-run — expect pass**

**Step 5: Commit**

```bash
git add backend/session.py backend/tests/test_session.py
git commit -m "feat(session): entry ids, editing, rename, clear, slug filename on finalize"
```

---

## Task 5: Backend VAD pipeline + new WS actions

**Files:**
- Modify: `backend/main.py`
- Test: `backend/tests/test_main_ws.py` (new)

**Step 1: Add integration test**

Create `backend/tests/test_main_ws.py`:

```python
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from backend.main import app


def test_edit_action_updates_session_and_broadcasts():
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"action": "start"})
            msg = ws.receive_json()
            assert msg["type"] == "status" and msg["status"] == "recording"

            # inject a fake final entry
            from backend.main import current_session
            eid = current_session.add_entry("hello", "你好", 0, 1)

            ws.send_json({"action": "edit", "id": eid, "field": "text", "value": "Hello!"})
            update = ws.receive_json()
            assert update["type"] == "entry_updated"
            assert update["id"] == eid
            assert update["text"] == "Hello!"


def test_pause_resume_toggles_audio_only():
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"action": "start"})
            ws.receive_json()
            ws.send_json({"action": "pause"})
            assert ws.receive_json()["status"] == "paused"
            ws.send_json({"action": "resume"})
            assert ws.receive_json()["status"] == "recording"
```

**Step 2: Run — expect fail**

**Step 3: Rewrite `process_audio_loop` and WS handler**

Replace the old buffered loop with a VAD-driven one. Sketch:

```python
from backend.vad import VadSegmenter

_vad: VadSegmenter | None = None
_current_partial_id: str | None = None
_event_queue: asyncio.Queue = asyncio.Queue()

async def process_vad_events():
    global _current_partial_id
    while True:
        evt = await _event_queue.get()
        if evt["event"] == "speech_start":
            _current_partial_id = str(uuid.uuid4())
        elif evt["event"] == "partial":
            if transcriber is None: continue
            text = await asyncio.to_thread(transcriber.transcribe_partial, evt["audio"])
            await broadcast({"type": "partial", "id": _current_partial_id, "text": text})
        elif evt["event"] == "speech_end":
            segs = await asyncio.to_thread(transcriber.transcribe, evt["audio"])
            for seg in segs:
                translation = ""
                if config.display_mode in ("bilingual", "chinese"):
                    try:
                        translation = await get_translator(
                            config.translation_engine,
                            api_key=config.openai_api_key,
                        ).translate(seg.text)
                    except Exception:
                        translation = "[translation error]"
                eid = _current_partial_id or str(uuid.uuid4())
                if current_session:
                    current_session.entries.append({
                        "id": eid,
                        "text": seg.text,
                        "translation": translation,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "edited": {"text": False, "translation": False},
                    })
                await broadcast({
                    "type": "final",
                    "id": eid,
                    "text": seg.text,
                    "translation": translation,
                })
            _current_partial_id = None
```

WS handler gains:

- `start`: unchanged except also constructs `VadSegmenter(on_event=lambda e: _event_queue.put_nowait(e), silence_ms=config.vad_silence_ms, sensitivity=config.vad_sensitivity, partial_interval_ms=config.partial_interval_ms)`; audio on_chunk pushes into `_vad.feed(chunk)` (wrap in `asyncio.to_thread` — VAD is sync and may be slow).
- `pause`: `audio_manager.stop()`; broadcast `{"type": "status", "status": "paused"}`
- `resume`: `audio_manager.start()`; broadcast `{"type": "status", "status": "recording"}`
- `clear`: `current_session.clear()`; broadcast `{"type": "cleared"}`
- `rename`: `current_session.rename(name)`; broadcast `{"type": "renamed", "name": name}`
- `edit`: `current_session.update_entry(id, field, value)`; broadcast `{"type": "entry_updated", "id": id, "field": field, "text": ..., "translation": ...}`
- `retranslate`: translate entry text again, call `update_entry(id, "translation", new_val)`, broadcast

**Step 4: Re-run tests — expect pass**

**Step 5: Commit**

```bash
git add backend/main.py backend/tests/test_main_ws.py
git commit -m "feat(main): VAD pipeline + edit/pause/resume/clear/rename WS actions"
```

---

## Task 6: Add `/sessions/current` endpoint + auto-save

**Files:**
- Modify: `backend/main.py`, `backend/session.py`
- Test: `backend/tests/test_main_ws.py`

**Step 1: Add failing test**

```python
def test_sessions_current_returns_active_session_snapshot():
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"action": "start"})
            ws.receive_json()
            resp = client.get("/sessions/current")
            assert resp.status_code == 200
            data = resp.json()
            assert "id" in data and "entries" in data and "name" in data
```

**Step 2 onward**: add `GET /sessions/current` returning `current_session.to_dict()`; add `Session.to_dict()`; hook auto-save via `asyncio.create_task` scheduled every 30s while session is active.

**Commit:**

```bash
git commit -m "feat: /sessions/current endpoint and 30s auto-save"
```

---

## Task 7: Frontend store for v2

**Files:**
- Modify: `frontend/src/store/appStore.ts`

Add to the store:

```ts
paused: boolean;
sessionName: string;
sessionStartedAt: number | null;
partialEntry: { id: string; text: string } | null;

setPaused: (v: boolean) => void;
setSessionName: (name: string) => void;
startSession: (name: string) => void;     // sets name, startedAt, clears entries
endSession: () => void;                    // resets everything
setPartialEntry: (p: { id: string; text: string } | null) => void;
finalizePartial: (entry: Entry) => void;   // clears partial, adds to entries
updateEntry: (id: string, field: "text" | "translation", value: string) => void;
deleteEntry: (id: string) => void;
```

Entry type now has `id: string` and `edited: { text: boolean; translation: boolean }`.

**Tests:** skip for store — tested indirectly through useWebSocket tests in next task.

**Commit:**

```bash
git commit -m "feat(store): session metadata, partial entry, entry edits"
```

---

## Task 8: Frontend useWebSocket — new message types and actions

**Files:**
- Modify: `frontend/src/hooks/useWebSocket.ts`

Handle incoming:

- `partial` → `setPartialEntry({id, text})`
- `final` → `finalizePartial({id, text, translation, ...})`
- `entry_updated` → `updateEntry(id, field, value)`
- `status: paused` → `setPaused(true)` ; `recording` → `setPaused(false); setRecording(true)`
- `cleared` → reset entries
- `renamed` → `setSessionName(name)`

Expose helpers: `edit`, `pause`, `resume`, `clear`, `rename`, `retranslate`, plus existing `send(action, extra)`.

**Commit:**

```bash
git commit -m "feat(ws): partial/final/edit messages and action helpers"
```

---

## Task 9: TranscriptEntry component with inline edit

**Files:**
- Create: `frontend/src/components/TranscriptEntry.tsx`
- Modify: `frontend/src/components/TranscriptView.tsx`

TranscriptEntry responsibilities:
- Render `entry.text` and `entry.translation` per `displayMode`
- Hover → small pencil button per field
- Double-click text OR click pencil → `<textarea autoFocus>` replaces the text
- Enter (no shift) or blur → emit `onEdit(id, field, value)`; Esc → revert
- Show a small dot (blue for text, amber for translation) if `edited[field]` true
- When translation edited or missing, show ↻ button → `onRetranslate(id)`

TranscriptView passes `entries.map(e => <TranscriptEntry key={e.id} />)` and, below, renders `partialEntry` as a greyed-out single row (no edit affordance).

**Visual test (manual):** run dev, add an entry via dev tools, confirm edit flow.

**Commit:**

```bash
git commit -m "feat(ui): TranscriptEntry with inline edit + retranslate"
```

---

## Task 10: SessionHeader + timer

**Files:**
- Create: `frontend/src/components/SessionHeader.tsx`
- Modify: `frontend/src/App.tsx` (mount header)

SessionHeader shows:
- Editable session name (click to rename → emits `rename(name)` via WS)
- Elapsed time `mm:ss` or `hh:mm:ss`, ticking every 1s while `recording && !paused`, computed from `sessionStartedAt`

**Commit:**

```bash
git commit -m "feat(ui): SessionHeader with editable name + live timer"
```

---

## Task 11: ControlBar v2 — Pause, Clear, Export

**Files:**
- Modify: `frontend/src/components/ControlBar.tsx`

State machine:
- Idle: [Start]
- Recording: [Pause] [Stop] [Clear] [Export ▾]
- Paused: [Resume] [Stop] [Clear] [Export ▾]

Clear opens a confirm dialog.
Export dropdown has `.txt`, `.md`, `.srt` — each fetches `/sessions/current/export?fmt=X` and triggers a download.

**Backend companion:** add `/sessions/current/export?fmt=` endpoint (mirrors existing saved-session export).

**Commit:**

```bash
git commit -m "feat(ui): pause/clear/export controls; current-session export endpoint"
```

---

## Task 12: StartDialog

**Files:**
- Create: `frontend/src/components/StartDialog.tsx`
- Modify: `ControlBar.tsx` (show dialog instead of sending `start` directly)

Modal with:
- Name input (default: today's date/time)
- Source selector (Microphone / System audio)
- Translation engine (Google / OpenAI)
- [Cancel] [Start]

On Start → `rename(name)` then `send("start", {device_index})`.

**Commit:**

```bash
git commit -m "feat(ui): StartDialog for session name + quick settings"
```

---

## Task 13: History drawer

**Files:**
- Create: `frontend/src/components/History.tsx`
- Modify: `App.tsx` (mount drawer + toggle button in header)

Drawer fetched via `GET /sessions`, lists `{name, date, entry_count, duration}`. Clicking a row fetches full session, renders read-only TranscriptView with an Export dropdown and a Delete button.

**Commit:**

```bash
git commit -m "feat(ui): history drawer for past sessions"
```

---

## Task 14: Styling pass

**Files:** `App.css` and any component CSS.

Polish:
- Partial row: italic, opacity 0.6
- Edited indicator dots
- Timer in mono font
- Drawer slide animation
- Export dropdown alignment

**Commit:**

```bash
git commit -m "style: v2 polish pass"
```

---

## Task 15: End-to-end manual test

Follow this checklist:

1. Start backend, start Tauri dev
2. Click Start → dialog appears, name "Test Lec", click Start
3. Speak for ~15 seconds with natural pauses
   - Confirm partial row appears and updates live
   - Confirm it finalizes into a normal row with Chinese translation on pause
   - Confirm timer ticks
4. Double-click an English entry, edit, Enter — confirm edited dot appears and translation stays unchanged
5. Click ↻ on that entry — confirm Chinese re-translates
6. Click Pause — confirm timer freezes and audio stops
7. Click Resume — confirm timer resumes
8. Click Clear — confirm dialog, then list empties
9. Speak again, then Stop — confirm file appears in history drawer under correct name
10. Open it from history, export as .md, open file — confirm content matches

If any step fails, open an issue in the plan or hand back for a targeted fix.

**Commit (only if cleanup is needed):**

```bash
git commit -m "fix: post-e2e cleanups"
```

---

## Done criteria

- All backend tests pass: `pytest backend/tests/ -v`
- All tasks above committed
- Manual checklist in Task 15 passes
- No console errors in Tauri dev
- Session JSON files in `~/LiveScribe/sessions/` contain `id` and `edited` fields per entry
