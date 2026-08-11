"""backend/app.py — the .app entry point.

Untested until now, which was a gap worth closing: it became the primary way
Utter is launched, and the two bugs that cost the author a morning both lived
here. Neither raised anything. One showed nothing on screen for seventy-nine
seconds; the other let macOS answer a denied microphone with silence.
"""

import sys
import types

import pytest

from backend import app as app_mod


# --- microphone permission ------------------------------------------------------


def _fake_avfoundation(status, request_result=None, calls=None):
    module = types.SimpleNamespace(
        AVMediaTypeAudio="soun",
        AVCaptureDevice=types.SimpleNamespace(
            authorizationStatusForMediaType_=lambda _t: status,
            requestAccessForMediaType_completionHandler_=(
                lambda _t, handler: (calls.append(1) if calls is not None else None,
                                     handler(request_result))[-1]
            ),
        ),
    )
    return module


@pytest.fixture
def av(monkeypatch):
    def install(status, request_result=None, calls=None):
        monkeypatch.setitem(
            sys.modules, "AVFoundation",
            _fake_avfoundation(status, request_result, calls),
        )
    return install


def test_an_already_granted_microphone_is_not_asked_for_again(av):
    calls = []
    av(3, calls=calls)
    assert app_mod.request_microphone(app_mod.log) is True
    assert calls == [], "asking again would show a prompt for no reason"


def test_an_undetermined_microphone_is_requested(av):
    calls = []
    av(0, request_result=True, calls=calls)
    assert app_mod.request_microphone(app_mod.log) is True
    assert calls == [1], "the prompt only appears if something asks; PortAudio never does"


def test_a_refused_request_stops_startup(av):
    av(0, request_result=False)
    assert app_mod.request_microphone(app_mod.log) is False


def test_a_denied_microphone_explains_itself(av, monkeypatch):
    """The failure this replaces was silent: macOS hands a denied app exact
    zeros rather than an error, so dictation said 「没听到」 and meant it."""
    said = {}
    monkeypatch.setattr(app_mod, "alert",
                        lambda title, body, **kw: said.update(title=title, body=body, **kw))
    av(2)

    assert app_mod.request_microphone(app_mod.log) is False
    assert "麦克风" in said["title"]
    assert "静音" in said["body"], "must name the symptom, not just the permission"
    assert "Privacy_Microphone" in said["settings_url"]


def test_missing_bindings_do_not_block_dictation(monkeypatch):
    """铁律 8's shape: not being able to check a permission must not cost the
    words. Let the recording try and fail loudly instead."""
    monkeypatch.setitem(sys.modules, "AVFoundation", None)
    monkeypatch.delitem(sys.modules, "AVFoundation")

    import builtins

    real_import = builtins.__import__

    def no_avfoundation(name, *args, **kwargs):
        if name == "AVFoundation":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_avfoundation)
    assert app_mod.request_microphone(app_mod.log) is True


def test_microphone_status_is_reported_in_words(av):
    av(3)
    assert app_mod.microphone_status() == "已授权"
    av(2)
    assert app_mod.microphone_status() == "已拒绝"


# --- startup order ---------------------------------------------------------------


def test_the_slow_work_happens_after_the_run_loop_starts():
    """Everything slow — the provider lookup, the accessibility wait, the model
    warm-up — must be behind the menu bar item, not in front of it. Doing it in
    front produced seventy-nine seconds of nothing on screen, and the author
    correctly concluded the app had not opened.

    Asserted on the source because the alternative is standing up AppKit in a
    test, and the property being protected is an ordering one.
    """
    import inspect

    source = inspect.getsource(app_mod._run)
    install = source.index("menu.install()")
    for slow in ("get_stt_provider", "_wait_for_accessibility", "daemon.start()"):
        assert source.index(slow) > install, f"{slow} runs before the icon appears"
    assert source.index("threading.Thread") > install
