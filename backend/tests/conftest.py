"""No test may touch the user's real `~/Utter`.

Twice now the suite has quietly rewritten the author's live settings. The first
time it lost thirty vocabulary terms. The second time it set `polish_level` to
`heavy` — because `test_changing_the_level_rebuilds_too` sets heavy, and the
daemon it built saved that to the real config file on the real machine — and
the author found their dictation being polished at a level they never chose.

The tests were not obviously wrong. `set_polish()` persists, which is correct
behaviour and exactly what those tests exist to check; the daemon under test
simply had no idea it was not a real one. So the guard belongs here rather than
in each test: every test gets its own directory, whether it thought to ask for
one or not, and a test that genuinely wants the default location has to say so
by using `real_utter_dir`.
"""

from __future__ import annotations

import pathlib

import pytest


@pytest.fixture(autouse=True)
def utter_dir_is_never_the_real_one(tmp_path, monkeypatch):
    """Point `~/Utter` at a per-test directory.

    `save()` and `load()` read `DEFAULT_DIR` off the module when they run, not
    when they were imported, so patching it here reaches every caller —
    including the ones several frames down inside a daemon.
    """
    import backend.config as config

    # Deliberately not created. `save()` makes it, and `migrate_legacy()`
    # refuses to migrate into a directory that already exists — correctly, so
    # that it cannot overwrite a real installation. A fixture that helpfully
    # pre-made it turned that guard into two failing tests.
    sandbox = tmp_path / "Utter"
    monkeypatch.setattr(config, "DEFAULT_DIR", sandbox)
    return sandbox


@pytest.fixture
def real_utter_dir(monkeypatch):
    """Opt back out, for the handful of tests that are *about* the real path.

    Restores `DEFAULT_DIR` to the value the module was born with. It still does
    not permit writing there — read the constant, do not save through it.
    """
    import backend.config as config

    monkeypatch.setattr(config, "DEFAULT_DIR", pathlib.Path.home() / "Utter")
    return config.DEFAULT_DIR
