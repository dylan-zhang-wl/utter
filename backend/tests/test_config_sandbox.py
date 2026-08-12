"""The guard that keeps the suite out of the user's real settings.

This file is here because the guard is invisible: it lives in an autouse
fixture, it is easy to delete during a refactor, and nothing else would notice
until someone's live config changed under them. Twice.
"""

import pathlib

from backend import config as cfg


def test_the_default_directory_is_not_the_real_one():
    assert cfg.DEFAULT_DIR != pathlib.Path.home() / "Utter"
    assert "pytest" in str(cfg.DEFAULT_DIR)


def test_saving_without_a_directory_lands_in_the_sandbox(utter_dir_is_never_the_real_one):
    """`save()` with no `base_dir` is the call that did the damage: it reaches
    for DEFAULT_DIR, and inside a daemon nobody passes one."""
    written = cfg.save(cfg.AppConfig())

    assert written == utter_dir_is_never_the_real_one / "config.json"
    assert not str(written).startswith(str(pathlib.Path.home() / "Utter"))


def test_a_daemon_that_persists_cannot_reach_the_real_config(utter_dir_is_never_the_real_one):
    """The exact shape of the bug: a test builds a daemon, calls something that
    persists, and the daemon writes wherever DEFAULT_DIR points."""
    cfg.save(cfg.AppConfig(polish_level="heavy"))

    reloaded = cfg.load()
    assert reloaded.polish_level == "heavy"
    assert (utter_dir_is_never_the_real_one / "config.json").exists()


def test_the_opt_out_exists_for_tests_about_the_real_path(real_utter_dir):
    assert real_utter_dir == pathlib.Path.home() / "Utter"


def test_the_session_archive_lands_in_the_sandbox_too(utter_dir_is_never_the_real_one):
    """The config was guarded and the archive sitting next to it was not.

    A day of real use turned up 1,299 session files in the author's live
    archive, written by test runs, mixed in among the real ones. Two separate
    import-time bindings caused it: `save_dir`'s default was evaluated when
    config.py was imported, and scratchpad.py had done
    `from backend.config import DEFAULT_DIR`, which copies the value rather
    than following it.
    """
    from backend.scratchpad import SessionArchive

    archive = SessionArchive()
    assert str(archive.path).startswith(str(utter_dir_is_never_the_real_one))


def test_save_dir_follows_the_sandbox(utter_dir_is_never_the_real_one):
    """A plain default would have been frozen at import and would point home."""
    assert cfg.AppConfig().save_dir == str(utter_dir_is_never_the_real_one / "sessions")


def test_writing_an_utterance_cannot_reach_the_real_archive(tmp_path):
    """The end-to-end version: archive something and prove where it went."""
    import pathlib

    from backend.scratchpad import SessionArchive, Utterance

    archive = SessionArchive()
    archive.append(Utterance(index=0, mode="dictate", text="测试",
                             raw_text="测试", start_sec=0.0, end_sec=1.0))

    assert archive.path.exists()
    assert not str(archive.path).startswith(str(pathlib.Path.home() / "Utter"))
