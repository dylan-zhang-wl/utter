"""One dictation daemon per machine, and a clear message when there are two.

Found the hard way. A daemon started at 10:26 was still running at 21:30 — the
author had quit its terminal tab and moved on, but the process outlived it.
Both instances held a hotkey listener, so every press woke both; both opened
the microphone; and the numbers from the surviving one were nonsense.

Nothing about that is visible from the outside. The old process prints to a
terminal nobody is looking at, takes no CPU while idle, and the only symptom is
that dictation gets worse in ways that look like model or microphone problems —
which is exactly where two evenings of diagnosis went.

A lock file with a pid in it. Not fancy: this guards against forgetting, not
against a determined race.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AlreadyRunning:
    pid: int

    def message(self) -> str:
        return (
            f"已经有一个 Utter 听写在跑了（进程号 {self.pid}）。\n"
            "\n"
            "  两个实例会同时抢热键和麦克风，录到的音频会缺一大截，\n"
            "  而且症状看起来像是模型或麦克风坏了。\n"
            "\n"
            f"  先停掉那个：  kill {self.pid}\n"
            "  然后再启动这一个。"
        )


#: A dictation daemon, however it was launched. Both spellings exist in the
#: wild: `utter dictate` from the installed console script, and
#: `python -m backend.cli dictate` from inside the repo, which is how the
#: eleven-hour one was started.
def _is_dictation(command: str) -> bool:
    if "dictate" not in command:
        return False
    if "/utter " in command or command.endswith("/utter"):
        return True
    return "backend.cli" in command or "backend/cli" in command


def _alive(pid: int) -> bool:
    """Is this pid a live process?

    signal 0 asks the kernel without delivering anything. EPERM means the
    process exists but belongs to someone else — still alive, still a conflict.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class InstanceLock:
    """Hold the lock for as long as the daemon runs.

    A lock file left behind by a crash is not an error: the pid inside it is
    checked, and a stale one is taken over rather than refused. Refusing to
    start because of a file left by a process that died months ago would be a
    worse failure than the one this prevents.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.held = False

    def existing_owner(self) -> AlreadyRunning | None:
        return self._from_lock_file() or self._from_process_list()

    def _from_lock_file(self) -> AlreadyRunning | None:
        try:
            pid = int(self.path.read_text().strip())
        except (OSError, ValueError):
            return None
        if pid == os.getpid() or not _alive(pid):
            return None
        return AlreadyRunning(pid=pid)

    def _from_process_list(self) -> AlreadyRunning | None:
        """Look for a sibling that never wrote a lock file.

        The lock file alone was not enough, and the gap showed up the day it
        shipped: the daemon that had been running since that morning predated
        the feature, so it held no lock, and a new daemon started happily
        alongside it. Both grabbed the hotkey; both grabbed the microphone.

        Any daemon whose lock file was lost — a crash between writing and
        cleanup, a cleared temp directory — lands in the same place. Asking the
        process table costs one `ps` at startup and closes both.
        """
        try:
            listing = subprocess.run(
                ["ps", "-Ao", "pid=,command="],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None

        mine = os.getpid()
        for line in listing.splitlines():
            pid_text, _, command = line.strip().partition(" ")
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            if pid == mine or not _is_dictation(command):
                continue
            return AlreadyRunning(pid=pid)
        return None

    def acquire(self) -> AlreadyRunning | None:
        """Take the lock, or say who has it. Never raises."""
        owner = self.existing_owner()
        if owner is not None:
            return owner
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(str(os.getpid()))
            self.held = True
        except OSError:
            # An unwritable lock directory must not stop the author dictating.
            # The guard is a convenience; the tool is the point.
            pass
        return None

    def release(self) -> None:
        if not self.held:
            return
        try:
            if self.path.read_text().strip() == str(os.getpid()):
                self.path.unlink()
        except (OSError, ValueError):
            pass
        self.held = False

    def __enter__(self) -> "InstanceLock":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.release()
