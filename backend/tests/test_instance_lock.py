"""One daemon per machine.

Written after finding a dictation daemon that had been running for eleven
hours, forgotten, holding a hotkey listener and PortAudio alongside the one the
author was actually testing with. Every press woke both.
"""

import os

from backend.instance_lock import InstanceLock


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
