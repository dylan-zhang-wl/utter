"""P3 task 3 — `utter listen`, driven end to end without a microphone.

The command is the assembly: microphone → VAD → Whisper → session → batched
translation → two files. These tests replace the microphone and the model with
scripted stand-ins so the wiring can be checked without a meeting, a model
load, or touching the dictation daemon the author has running.
"""

import io

import numpy as np
import pytest

from backend.cli import main
from backend.vad import SAMPLE_RATE


class ScriptedMic:
    """Hands over a fixed run of audio, then reports itself stopped."""

    device_name = "测试麦克风"

    def __init__(self, seconds=1.0, utterances=1):
        chunk = np.full(int(SAMPLE_RATE * seconds), 0.2, dtype=np.float32)
        self._pending = [chunk] * utterances
        self.stopped_reason = None
        self.stopped = 0

    def start(self):
        return self

    def stop(self):
        self.stopped += 1

    def chunks(self):
        if self._pending:
            yield self._pending.pop(0)
        else:
            self.stopped_reason = "音频放完了"


class ScriptedStt:
    id = "scripted"
    display_name = "Scripted"

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def is_available(self):
        return True, ""

    def transcribe(self, audio, language=None, initial_prompt=None):
        self.calls += 1
        return self.texts.pop(0) if self.texts else ""


@pytest.fixture(autouse=True)
def always_speech(monkeypatch):
    """The VAD is tested elsewhere; here every chunk is speech so the assembly
    is what is under test."""
    import backend.vad as vad

    class OneUtterance:
        def __init__(self, **_kw):
            pass

        def feed(self, chunk):
            return [vad.SpeechEnd(start_sample=0, end_sample=len(chunk), audio=chunk)]

        def flush(self):
            return []

    monkeypatch.setattr(vad, "VadSegmenter", OneUtterance)
    import backend.cli  # noqa: F401  (imports VadSegmenter lazily, inside the function)


def run(monkeypatch, texts, translate=None, title="测试会议"):
    out = io.StringIO()
    stt = ScriptedStt(texts)
    code = main(["listen", "--title", title], stdout=out, stt=stt,
                translate=translate, mic=ScriptedMic(utterances=len(texts) or 1))
    return code, out.getvalue()


def test_a_meeting_is_transcribed_and_both_records_written(monkeypatch, tmp_path):
    code, output = run(monkeypatch, ["So the question of equivalence"])

    assert code == 0
    assert "So the question of equivalence" in output
    assert "逐字记录" in output and "中英对照" in output


def test_the_two_files_actually_exist_afterwards(monkeypatch):
    from backend.listen import list_sessions

    run(monkeypatch, ["Venuti argues that fluency is an ideology"])

    sessions = list_sessions()
    assert sessions, "会话目录没建出来"
    latest = sessions[0]
    assert (latest / "原文.md").exists()
    assert (latest / "对照.md").exists()
    assert (latest / "entries.jsonl").exists()


def test_the_translation_lands_next_to_the_english(monkeypatch):
    from backend.listen import list_sessions

    def translate(text, context=None):
        # The queue sends numbered lines; reply in kind.
        return "\n".join(f"{i + 1}. 译文{i + 1}"
                         for i in range(len(text.splitlines())))

    run(monkeypatch, ["Venuti argues that"], translate=translate)

    bilingual = (list_sessions()[0] / "对照.md").read_text(encoding="utf-8")
    assert "Venuti argues that" in bilingual
    assert "译文1" in bilingual


def test_a_silence_hallucination_never_reaches_the_record(monkeypatch):
    """A meeting is mostly silence between speakers, which is exactly when
    Whisper produces a subtitle credit. Dictation meets this occasionally; a
    lecture meets it constantly."""
    from backend.listen import list_sessions

    code, output = run(monkeypatch, ["谢谢观看", "So the question of equivalence"])

    assert code == 0
    assert "静音幻觉" in output
    verbatim = (list_sessions()[0] / "原文.md").read_text(encoding="utf-8")
    assert "So the question of equivalence" in verbatim, "真话应该留下"
    assert "谢谢观看" not in verbatim, "幻觉不该进记录"


def test_translation_being_unavailable_still_transcribes(monkeypatch):
    """铁律 8. No key, no network, no translation — but the meeting is still
    captured, which is the part that cannot be redone."""
    from backend.listen import list_sessions

    code, output = run(monkeypatch, ["So the question of equivalence"],
                       translate=None)

    assert code == 0
    verbatim = (list_sessions()[0] / "原文.md").read_text(encoding="utf-8")
    assert "So the question of equivalence" in verbatim


def test_the_verbatim_file_keeps_fillers_the_screen_dropped(monkeypatch):
    from backend.listen import list_sessions

    code, output = run(monkeypatch, ["So um the question of, you know, equivalence"])

    latest = list_sessions()[0]
    verbatim = (latest / "原文.md").read_text(encoding="utf-8")
    assert "um" in verbatim, "逐字版必须留着口水词"
    assert "um" not in output.split("逐字记录")[0], "屏幕上不该显示口水词"


def test_a_running_dictation_daemon_is_pointed_out(monkeypatch):
    """In an in-person meeting the room microphone hears the author too, so a
    dictated note lands in the meeting transcript regardless. Better said than
    discovered."""
    import backend.instance_lock as lock_mod

    class Owner:
        pid = 4321

        def message(self):
            return "running"

    monkeypatch.setattr(lock_mod.InstanceLock, "existing_owner",
                        lambda self: Owner())

    _, output = run(monkeypatch, ["hello"])
    assert "抢麦克风" in output
    assert "4321" in output


def test_the_microphone_is_released_even_if_something_goes_wrong(monkeypatch):
    """A held-open microphone is how the next dictation silently records
    nothing."""
    mic = ScriptedMic()

    class Exploding(ScriptedStt):
        def transcribe(self, audio, language=None, initial_prompt=None):
            raise RuntimeError("模型炸了")

    out = io.StringIO()
    code = main(["listen"], stdout=out, stt=Exploding([]), translate=None, mic=mic)

    assert code == 0
    assert mic.stopped >= 1, "麦克风没关"


# --- 铁律 3's exception, end to end ---------------------------------------------


def test_the_recording_is_gone_when_the_meeting_ends(monkeypatch):
    """It exists so a garbled line can be checked mid-meeting. It holds other
    people's voices, so it does not outlive the meeting."""
    from backend.listen import list_sessions

    code, output = run(monkeypatch, ["So the question of equivalence"])

    assert code == 0
    latest = list_sessions()[0]
    assert not (latest / "audio.wav").exists(), "录音应该在结束时删掉"
    assert (latest / "原文.md").exists(), "记录必须留下"


def test_no_audio_skips_the_recording_entirely(monkeypatch):
    from backend.listen import list_sessions

    out = io.StringIO()
    main(["listen", "--no-audio"], stdout=out, stt=ScriptedStt(["hello"]),
         translate=None, mic=ScriptedMic())

    assert not (list_sessions()[0] / "audio.wav").exists()


def test_a_recording_left_by_a_crash_is_swept_at_startup(monkeypatch):
    """The stop-time deletion cannot run if the process died. This is the other
    half of the same promise."""
    from backend import config as cfg
    from backend.listen import LISTEN_DIRNAME
    from backend.session_audio import SessionRecording

    stale = cfg.DEFAULT_DIR / LISTEN_DIRNAME / "20260101-000000"
    old = SessionRecording(stale)
    old.start()
    old.write(np.zeros(1600, dtype=np.float32))
    old._close()
    assert old.path.exists()

    _, output = run(monkeypatch, ["hello"])

    assert not old.path.exists(), "上次崩溃留下的录音应该被清掉"
    assert "没删干净" in output
