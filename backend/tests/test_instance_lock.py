"""One daemon per machine.

Written after finding a dictation daemon that had been running for eleven
hours, forgotten, holding a hotkey listener and PortAudio alongside the one the
author was actually testing with. Every press woke both.
"""

import os

import pytest

from backend.instance_lock import InstanceLock


@pytest.fixture(autouse=True)
def no_real_processes(monkeypatch):
    """An empty process table unless a test says otherwise.

    Without this the suite reads the machine it runs on: adding the sibling
    scan made four existing tests fail on the author's laptop, because a real
    daemon was running at the time. A test that passes or fails depending on
    what else is open is not a test.
    """
    import backend.instance_lock as mod

    class _Empty:
        stdout = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Empty())


def test_the_first_daemon_gets_the_lock(tmp_path):
    lock = InstanceLock(tmp_path / "dictate.pid")
    assert lock.acquire() is None
    assert lock.path.read_text() == str(os.getpid())


def test_a_second_daemon_is_told_who_has_it(tmp_path):
    path = tmp_path / "dictate.pid"
    path.write_text(str(os.getppid()))  # a real, living process that is not us

    owner = InstanceLock(path).acquire()
    assert owner is not None and owner.pid == os.getppid()
    assert str(os.getppid()) in owner.message()
    assert "kill" in owner.message()


def test_a_lock_left_by_a_dead_process_is_taken_over(tmp_path):
    """A crash must not lock the author out of their own tool until they go
    hunting for a stale file."""
    path = tmp_path / "dictate.pid"
    path.write_text("999999")  # far above any live pid on macOS

    assert InstanceLock(path).acquire() is None


def test_rubbish_in_the_lock_file_is_not_fatal(tmp_path):
    path = tmp_path / "dictate.pid"
    path.write_text("not a pid")
    assert InstanceLock(path).acquire() is None


def test_releasing_removes_the_file(tmp_path):
    lock = InstanceLock(tmp_path / "dictate.pid")
    lock.acquire()
    lock.release()
    assert not lock.path.exists()


def test_release_leaves_someone_elses_lock_alone(tmp_path):
    """Otherwise a daemon that lost a race would delete the winner's lock on
    its way out, and the guard would silently stop guarding."""
    path = tmp_path / "dictate.pid"
    lock = InstanceLock(path)
    lock.acquire()
    path.write_text("12345")  # somebody else took over
    lock.release()
    assert path.read_text() == "12345"


def test_an_unwritable_directory_does_not_stop_dictation(tmp_path):
    """The guard is a convenience. The tool is the point."""
    blocked = tmp_path / "nope"
    blocked.write_text("i am a file, not a directory")
    assert InstanceLock(blocked / "dictate.pid").acquire() is None


# --- the gap the lock file alone left ------------------------------------------


def test_a_sibling_without_a_lock_file_is_still_found(tmp_path, monkeypatch):
    """The day the lock shipped it failed to catch anything: the daemon that
    had been running since that morning predated the feature, held no lock, and
    a new one started happily beside it. Both took the hotkey and the
    microphone."""
    import backend.instance_lock as mod

    class _Ps:
        stdout = "  501 /usr/bin/python -m backend.cli dictate --timing\n 9999 /bin/zsh\n"

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Ps())

    owner = mod.InstanceLock(tmp_path / "dictate.pid").acquire()
    assert owner is not None and owner.pid == 501


def test_the_console_script_spelling_is_recognised(tmp_path, monkeypatch):
    import backend.instance_lock as mod

    class _Ps:
        stdout = " 742 /home/u/.venvs/utter/bin/python /home/u/.venvs/utter/bin/utter dictate\n"

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Ps())
    assert mod.InstanceLock(tmp_path / "dictate.pid").acquire().pid == 742


def test_unrelated_processes_are_not_mistaken_for_a_daemon(tmp_path, monkeypatch):
    """`grep dictate` would match this session's own editor. Be specific."""
    import backend.instance_lock as mod

    class _Ps:
        stdout = (
            " 100 /bin/zsh\n"
            " 101 vim backend/tests/test_dictate_something.py\n"
            " 102 /usr/bin/python -m backend.cli doctor\n"
            " 103 grep dictate\n"
        )

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Ps())
    assert mod.InstanceLock(tmp_path / "dictate.pid").acquire() is None


def test_a_broken_ps_does_not_stop_dictation(tmp_path, monkeypatch):
    import backend.instance_lock as mod

    def boom(*a, **k):
        raise OSError("no ps here")

    monkeypatch.setattr(mod.subprocess, "run", boom)
    assert mod.InstanceLock(tmp_path / "dictate.pid").acquire() is None
