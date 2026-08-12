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
