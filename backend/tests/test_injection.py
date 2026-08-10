"""P2a Task 5 — putting text where the cursor is.

This is the module 铁律 9, 12 and 13 were written for, so most of these tests
are about refusing to do things:

  9  never rewrite what was already injected
  12 clipboard paste, never per-character typing, and restore what was there
  13 buffer when focus has moved, never force the text somewhere else

The pasteboard and keyboard are faked throughout. A test run must not type into
whatever window happens to be frontmost, and must not clobber a real clipboard.
"""

import pytest

from backend import injection


class FakePasteboard:
    def __init__(self, items=None):
        self.items = items or [{"public.utf8-plain-text": b"the user's own clipboard"}]
        self.change_count = 1

    def snapshot(self):
        return [dict(item) for item in self.items]

    def restore(self, snapshot):
        self.items = [dict(item) for item in snapshot]
        self.change_count += 1

    def set_text(self, text):
        self.items = [{"public.utf8-plain-text": text.encode()}]
        self.change_count += 1

    @property
    def text(self):
        raw = self.items[0].get("public.utf8-plain-text") if self.items else None
        return raw.decode() if raw else None


class FakeKeyboard:
    def __init__(self):
        self.pastes = 0
        self.typed = []

    def paste(self):
        self.pastes += 1

    def type(self, text):  # must never be called for long text (铁律 12)
        self.typed.append(text)


def target(pid=100, name="Obsidian"):
    return injection.Target(pid=pid, name=name, bundle_id="md.obsidian")


@pytest.fixture
def rig(monkeypatch):
    board, keys = FakePasteboard(), FakeKeyboard()
    state = {"front": target(), "alive": {100, 200}, "activated": [], "notes": []}

    monkeypatch.setattr(injection, "_frontmost", lambda: state["front"])
    monkeypatch.setattr(injection, "_is_running", lambda pid: pid in state["alive"])
    monkeypatch.setattr(injection, "_activate", lambda pid: state["activated"].append(pid))
    monkeypatch.setattr(injection, "_notify", lambda title, body: state["notes"].append(body))

    injector = injection.Injector(pasteboard=board, keyboard=keys)
    order = []
    board.restore = lambda snap, _o=order, _b=board: (_o.append("restore"),
                                                      FakePasteboard.restore(_b, snap))[1]
    keys.paste = lambda _o=order, _k=keys: (_o.append("paste"), setattr(_k, "pastes", _k.pastes + 1))
    injector.settle = lambda _o=order: _o.append("settle")
    state["order"] = order
    return injector, board, keys, state


# --- 铁律 12: clipboard, not keystrokes ---------------------------------------


def test_injects_via_clipboard_paste(rig):
    injector, board, keys, _ = rig
    injector.lock_target()
    injector.inject(0, "Translation is rewriting.")

    assert keys.pastes == 1
    assert keys.typed == [], "铁律 12: never simulate per-character typing"


def test_restores_the_previous_clipboard(rig):
    injector, board, keys, _ = rig
    injector.lock_target()
    injector.inject(0, "dictated text")

    assert board.text == "the user's own clipboard"


def test_restores_a_non_text_clipboard(rig):
    """Restoring only public.utf8-plain-text silently destroys a copied image.
    The author will not connect the loss to having dictated."""
    injector, board, keys, _ = rig
    board.items = [{"public.png": b"\x89PNG-pretend-image"}]
    injector.lock_target()
    injector.inject(0, "dictated text")

    assert board.items == [{"public.png": b"\x89PNG-pretend-image"}]


def test_restores_all_flavours_of_a_rich_clipboard(rig):
    injector, board, keys, _ = rig
    board.items = [{
        "public.utf8-plain-text": b"plain",
        "public.rtf": b"{\\rtf1 rich}",
        "public.html": b"<b>rich</b>",
    }]
    injector.lock_target()
    injector.inject(0, "dictated")

    assert set(board.items[0]) == {"public.utf8-plain-text", "public.rtf", "public.html"}


def test_clipboard_is_restored_even_when_the_paste_fails(rig, monkeypatch):
    injector, board, keys, _ = rig
    monkeypatch.setattr(keys, "paste", lambda: (_ for _ in ()).throw(RuntimeError("no")))
    injector.lock_target()
    injector.inject(0, "dictated text")

    assert board.text == "the user's own clipboard"


# --- 铁律 9: never rewrite what was injected ----------------------------------


def test_the_same_utterance_is_never_injected_twice(rig):
    injector, board, keys, _ = rig
    injector.lock_target()
    injector.inject(0, "Translation is rewriting.")
    injector.inject(0, "Translation is a form of rewriting.")

    assert keys.pastes == 1, "铁律 9: one utterance, one injection, no corrections"


def test_a_later_utterance_still_injects(rig):
    injector, board, keys, _ = rig
    injector.lock_target()
    injector.inject(0, "first")
    injector.inject(1, "second")

    assert keys.pastes == 2


# --- 铁律 13: buffer when focus has moved -------------------------------------


def test_buffers_when_focus_has_moved_away(rig):
    injector, board, keys, state = rig
    injector.lock_target()
    state["front"] = target(pid=200, name="Safari")
    injector.inject(0, "dictated while elsewhere")

    assert keys.pastes == 0
    assert injector.pending == 1


def test_buffered_text_is_not_lost(rig):
    injector, board, keys, state = rig
    injector.lock_target()
    state["front"] = target(pid=200, name="Safari")
    injector.inject(0, "dictated while elsewhere")

    assert "dictated while elsewhere" in injector.pending_text


def test_returning_to_the_target_flushes(rig):
    injector, board, keys, state = rig
    injector.lock_target()
    state["front"] = target(pid=200, name="Safari")
    injector.inject(0, "one")
    injector.inject(1, "two")

    state["front"] = target(pid=100)
    injector.flush()

    assert keys.pastes == 1, "the backlog goes in as one paste, not one per utterance"
    assert injector.pending == 0


def test_flush_reactivates_the_target(rig):
    injector, board, keys, state = rig
    injector.lock_target()
    state["front"] = target(pid=200, name="Safari")
    injector.inject(0, "text")
    injector.flush()

    assert 100 in state["activated"]


def test_flush_with_nothing_pending_does_nothing(rig):
    injector, board, keys, _ = rig
    injector.lock_target()
    injector.flush()

    assert keys.pastes == 0


def test_no_target_locked_means_buffer(rig):
    """Scratchpad mode never locks a target; injection must be a no-op, not a
    crash or a paste into whatever is in front."""
    injector, board, keys, _ = rig
    injector.inject(0, "text")

    assert keys.pastes == 0
    assert injector.pending == 1


# --- the target going away ----------------------------------------------------


def test_a_quit_target_falls_back_to_the_clipboard(rig):
    injector, board, keys, state = rig
    injector.lock_target()
    state["alive"].discard(100)
    injector.inject(0, "dictated text")

    assert board.text == "dictated text", "left on the clipboard for the user"
    assert keys.pastes == 0


def test_a_quit_target_notifies(rig):
    """Silence here is the failure this project refuses — the user must be told
    where their words went."""
    injector, board, keys, state = rig
    injector.lock_target()
    state["alive"].discard(100)
    injector.inject(0, "dictated text")

    assert state["notes"], "must tell the user the text is on the clipboard"
    assert "⌘V" in state["notes"][0] or "clipboard" in state["notes"][0].lower()


def test_a_quit_target_never_raises(rig):
    injector, board, keys, state = rig
    injector.lock_target()
    state["alive"].discard(100)
    injector.inject(0, "text")  # must not raise


def test_injection_failure_reports_rather_than_raising(rig, monkeypatch):
    injector, board, keys, _ = rig
    monkeypatch.setattr(keys, "paste", lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    injector.lock_target()
    result = injector.inject(0, "text")

    assert result.injected is False
    assert result.reason


# --- target locking -----------------------------------------------------------


def test_lock_captures_the_frontmost_application(rig):
    injector, board, keys, state = rig
    injector.lock_target()

    assert injector.target.pid == 100
    assert injector.target.name == "Obsidian"


def test_lock_can_be_given_an_explicit_target(rig):
    injector, board, keys, _ = rig
    injector.lock_target(target(pid=999, name="Word"))

    assert injector.target.pid == 999


def test_relocking_replaces_the_target(rig):
    injector, board, keys, state = rig
    injector.lock_target()
    state["front"] = target(pid=200, name="Safari")
    injector.lock_target()

    assert injector.target.pid == 200


def test_release_clears_the_target(rig):
    injector, board, keys, _ = rig
    injector.lock_target()
    injector.release()

    assert injector.target is None


# --- what must never be built -------------------------------------------------


FORBIDDEN_APIS = {
    # Design §4.1e. Both put text into a non-frontmost window, both work on
    # native Cocoa fields, and both fail silently on Electron, browsers and
    # terminals — which is most of what the author writes in. Random failure is
    # worse than clear failure: the user cannot tell which segment was lost.
    "CGEventPostToPid",
    "AXUIElementSetAttributeValue",
    "kAXValueAttribute",
}


def _referenced_names(path):
    """Identifiers the code actually uses, ignoring strings and comments.

    Checked via AST rather than substring search precisely so the module can
    keep explaining in prose why these APIs are rejected — that explanation is
    the most valuable comment in the file, and a naive grep would forbid it.
    """
    import ast

    tree = ast.parse(open(path).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
            if node.asname:
                names.add(node.asname)
    return names


@pytest.mark.parametrize("api", sorted(FORBIDDEN_APIS))
def test_rejected_injection_apis_are_never_called(api):
    assert api not in _referenced_names(injection.__file__)


def test_the_rejection_is_still_explained_in_the_source():
    """The reasoning must survive refactors — someone will otherwise 'improve'
    this module by reaching for the API that looks more direct."""
    source = open(injection.__file__).read()
    assert "CGEventPostToPid" in source, "keep the explanation, just never call it"


# --- the paste/restore race (reported 2026-08-10) ----------------------------


def test_the_clipboard_is_restored_only_after_the_paste_settles(rig):
    """⌘V only posts a keystroke; the application reads the pasteboard some
    milliseconds later on its own event loop.

    Restoring straight after the post meant the application read what was there
    *before* — the author watched a previously-copied shell command appear where
    their dictation should have been.
    """
    injector, board, keys, state = rig
    injector.lock_target()
    injector.inject(0, "the dictated text")

    assert state["order"] == ["paste", "settle", "restore"]


def test_settle_waits_long_enough_to_outlast_an_event_loop_turn():
    assert injection.PASTE_SETTLE_SECONDS >= 0.15


# --- activation must land before the keystroke (reported 2026-08-10) ---------


def test_focus_elsewhere_brings_the_original_window_back(rig, monkeypatch):
    """The author's stated expectation: text belongs where they were when they
    started talking, whatever they switched to while saying it."""
    injector, board, keys, state = rig
    injector.lock_target()
    state["front"] = target(pid=200, name="WeChat")

    def activate_and_wait(pid, timeout=1.5):
        state["activated"].append(pid)
        state["front"] = target(pid=pid, name="Obsidian")  # it really came forward
        return True

    monkeypatch.setattr(injection, "_activate_and_wait", activate_and_wait)
    result = injector.inject(0, "text")

    assert result.injected is True
    assert state["activated"] == [100]


def test_nothing_is_typed_if_the_window_does_not_come_forward(rig, monkeypatch):
    """activateWithOptions_ only *requests* activation. Typing before it lands
    sent the author's dictation into WeChat while the log said Claude."""
    injector, board, keys, state = rig
    injector.lock_target()
    state["front"] = target(pid=200, name="WeChat")
    monkeypatch.setattr(injection, "_activate_and_wait", lambda pid, timeout=1.5: False)

    result = injector.inject(0, "text")

    assert keys.pastes == 0, "must not type into whatever happens to be in front"
    assert result.buffered is True
    assert "WeChat" in result.reason


def test_activation_wait_polls_until_frontmost(monkeypatch):
    calls = {"n": 0}

    def frontmost():
        calls["n"] += 1
        return injection.Target(pid=100, name="Late") if calls["n"] > 3 else \
               injection.Target(pid=999, name="Other")

    monkeypatch.setattr(injection, "_activate", lambda pid: None)
    monkeypatch.setattr(injection, "_frontmost", frontmost)

    assert injection._activate_and_wait(100, timeout=2.0) is True
    assert calls["n"] > 3, "it waited rather than trusting the request"


def test_activation_wait_gives_up_rather_than_hanging(monkeypatch):
    monkeypatch.setattr(injection, "_activate", lambda pid: None)
    monkeypatch.setattr(injection, "_frontmost", lambda: injection.Target(pid=999, name="Other"))

    assert injection._activate_and_wait(100, timeout=0.2) is False
