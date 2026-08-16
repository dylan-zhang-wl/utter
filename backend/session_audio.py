"""会话录音 — the one place audio is allowed on disk, and only for a while.

铁律 3 says audio never reaches disk, and for dictation that is simply right: a
sentence that came back wrong gets said again. A meeting cannot be repeated. A
garbled line in a ninety-minute lecture is unrecoverable once the audio is
gone, and the author decided on 2026-08-16 to keep the recording for the length
of the session and delete it at the end.

So this is 铁律 3's first exception, and its bounds are the whole point of
putting it in a module of its own:

  * only during a listen session, never for dictation;
  * only inside that session's own directory;
  * deleted when the session ends, and swept on the next launch if the process
    died before it could — a recording of other people talking must not
    outlive the meeting because something crashed.

The file is plain 16-bit PCM. Not because it is small, but because writing it
costs nothing per chunk and a half-written WAV is still readable up to the
point it stops.
"""

from __future__ import annotations

import logging
import os
import struct
import threading
from pathlib import Path

log = logging.getLogger(__name__)

AUDIO_FILE = "audio.wav"
SAMPLE_RATE = 16000

#: A meeting this long is almost certainly a session somebody forgot to stop.
#: 16 kHz mono 16-bit is 115 MB an hour, so four hours is under half a gigabyte
#: — the cap is here to bound a forgotten session, not to ration a real one.
MAX_HOURS = 4.0


def _header(data_bytes: int) -> bytes:
    """A RIFF header for 16 kHz mono 16-bit PCM."""
    return (
        b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
        + b"data" + struct.pack("<I", data_bytes)
    )


class SessionRecording:
    """Writes a session's audio, and makes sure it does not survive the session.

    Every method is safe to call out of order and none of them raise: the
    recording is a convenience, and the transcript — the thing that cannot be
    redone — is written by an entirely separate path (铁律 8).
    """

    def __init__(self, directory: Path, *, max_hours: float = MAX_HOURS):
        self.path = Path(directory) / AUDIO_FILE
        self._max_bytes = int(max_hours * 3600 * SAMPLE_RATE * 2)
        self._fh = None
        self._written = 0
        self._lock = threading.Lock()
        self._capped = False

    # -- writing ----------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._fh is not None:
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                self._fh = self.path.open("wb")
                self._fh.write(_header(0))
                os.chmod(self.path, 0o600)   # it holds other people's voices
                self._written = 0
            except OSError:
                log.warning("会话录音开不了，转录不受影响", exc_info=True)
                self._fh = None

    def write(self, chunk) -> None:
        """Append float32 audio in [-1, 1]. Never raises."""
        with self._lock:
            if self._fh is None or chunk is None or len(chunk) == 0:
                return
            if self._written >= self._max_bytes:
                if not self._capped:
                    log.warning("会话录音到达 %.0f 小时上限，之后不再写入录音"
                                "（转录继续）", self._max_bytes
                                / (3600 * SAMPLE_RATE * 2))
                    self._capped = True
                return
            try:
                import numpy as np

                pcm = np.clip(np.asarray(chunk, dtype="float32"), -1.0, 1.0)
                data = (pcm * 32767.0).astype("<i2").tobytes()
                self._fh.write(data)
                self._written += len(data)
            except (OSError, ValueError):
                log.warning("会话录音写入失败，转录不受影响", exc_info=True)

    def _close(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            fh.seek(0)
            fh.write(_header(self._written))   # RIFF sizes are only known now
            fh.close()
        except OSError:
            log.warning("会话录音收尾失败", exc_info=True)

    # -- the part that matters --------------------------------------------

    def discard(self) -> bool:
        """Close and delete. This is what 结束 calls.

        Returns whether the file is gone. Called twice, or on a session that
        never recorded, it still reports success — the postcondition is "no
        recording on disk", not "a deletion happened".
        """
        self._close()
        try:
            if not self.path.exists():
                # Nothing there — because recording never started, because it
                # was already discarded, or because the directory was never
                # writable. All three satisfy "no recording on disk", which is
                # the postcondition; a deletion having occurred is not.
                return True
            self.path.unlink()
            return True
        except OSError:
            log.warning("会话录音删不掉：%s —— 请手动删除", self.path, exc_info=True)
            return False

    @property
    def seconds(self) -> float:
        return self._written / (SAMPLE_RATE * 2)

    @property
    def megabytes(self) -> float:
        return self._written / (1024 * 1024)


def sweep(base_dir: Path | None = None) -> int:
    """Delete recordings left behind by sessions that died. Returns how many.

    Run at launch. Without this, a crash mid-meeting leaves a recording of
    other people's conversation sitting in the author's home directory
    indefinitely — which is exactly the outcome the 结束 deletion exists to
    prevent, arriving by a different route.
    """
    from backend import config as _config
    from backend.listen import LISTEN_DIRNAME

    root = Path(base_dir) if base_dir else _config.DEFAULT_DIR / LISTEN_DIRNAME
    if not root.exists():
        return 0
    removed = 0
    for directory in root.iterdir():
        audio = directory / AUDIO_FILE
        if not audio.exists():
            continue
        try:
            audio.unlink()
            removed += 1
            log.info("清掉上次没删干净的会话录音：%s", audio)
        except OSError:
            log.warning("清不掉 %s", audio, exc_info=True)
    return removed
