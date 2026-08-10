"""P2a Task 2 — the global hotkey.

Split like the VAD: HotkeyMatcher is a pure state machine over key press and
release, and every behavioural test drives it directly. HotkeyListener is the
thin pynput wiring, tested with pynput patched.

The permission test matters more than it looks. macOS requires Accessibility
for global key capture, and without it pynput's listener receives nothing **and
raises nothing**. A dictation tool that starts cleanly and never hears its
hotkey is precisely the silent failure this project's rules exist to prevent, so
the listener refuses to start rather than pretending.
"""

import pytest

from backend import hotkey


class Key:
    """Stand-in for pynput key objects — only identity and equality matter."""

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Key) and other.name == self.name

    def __hash__(self):
        return hash(("Key", self.name))

    def __repr__(self):
        return f"<{self.name}>"


CMD, ALT, SHIFT, D = Key("cmd"), Key("alt"), Key("shift"), Key("d")


def matcher(mode="push", combination=(CMD, ALT)):
    events = []
    m = hotkey.HotkeyMatcher(
        combination=frozenset(combination), mode=mode, on_event=events.append
    )
    return m, events


def kinds(events):
    return [e.kind for e in events]


# --- push to talk ------------------------------------------------------------


def test_push_starts_when_the_combination_completes():
    m, events = matcher("push")
    m.press(CMD)
    assert kinds(events) == []
    m.press(ALT)
    assert kinds(events) == ["start"]


def test_push_stops_on_release():
    m, events = matcher("push")
    m.press(CMD)
    m.press(ALT)
    m.release(ALT)

    assert kinds(events) == ["start", "stop"]


def test_push_stops_when_any_key_of_the_combination_goes_up():
    m, events = matcher("push")
    m.press(CMD)
    m.press(ALT)
    m.release(CMD)

    assert kinds(events) == ["start", "stop"]


def test_push_ignores_unrelated_keys():
    m, events = matcher("push")
    m.press(SHIFT)
    m.press(D)
    m.release(D)

    assert events == []


def test_extra_keys_held_do_not_block_the_combination():
    """The user may already be holding shift when they reach for the hotkey."""
    m, events = matcher("push")
    m.press(SHIFT)
    m.press(CMD)
    m.press(ALT)

    assert kinds(events) == ["start"]


def test_push_does_not_restart_while_already_held():
    """Key repeat fires press over and over. One dictation, not fifty."""
    m, events = matcher("push")
    m.press(CMD)
    m.press(ALT)
    for _ in range(5):
        m.press(ALT)

    assert kinds(events) == ["start"]


def test_release_without_press_is_ignored():
    m, events = matcher("push")
    m.release(ALT)
    assert events == []


def test_push_can_run_twice():
    m, events = matcher("push")
    for _ in range(2):
        m.press(CMD)
        m.press(ALT)
        m.release(ALT)
        m.release(CMD)

    assert kinds(events) == ["start", "stop", "start", "stop"]


# --- toggle ------------------------------------------------------------------


def test_toggle_starts_on_the_first_press():
    m, events = matcher("toggle")
    m.press(CMD)
    m.press(ALT)

    assert kinds(events) == ["start"]


def test_toggle_does_not_stop_on_release():
    """That is the whole point — you let go and keep talking."""
    m, events = matcher("toggle")
    m.press(CMD)
    m.press(ALT)
    m.release(ALT)
    m.release(CMD)

    assert kinds(events) == ["start"]


def test_toggle_stops_on_the_second_press():
    m, events = matcher("toggle")
    for _ in range(2):
        m.press(CMD)
        m.press(ALT)
        m.release(ALT)
        m.release(CMD)

    assert kinds(events) == ["start", "stop"]


def test_toggle_alternates():
    m, events = matcher("toggle")
    for _ in range(3):
        m.press(CMD)
        m.press(ALT)
        m.release(ALT)
        m.release(CMD)

    assert kinds(events) == ["start", "stop", "start"]


def test_toggle_ignores_key_repeat():
    m, events = matcher("toggle")
    m.press(CMD)
    m.press(ALT)
    for _ in range(5):
        m.press(ALT)

    assert kinds(events) == ["start"]


# --- events ------------------------------------------------------------------


def test_events_carry_a_monotonic_timestamp():
    """Task 3 measures from the stop event, so it cannot be wall clock."""
    m, events = matcher("push")
    m.press(CMD)
    m.press(ALT)
    m.release(ALT)

    assert events[0].at < events[1].at
    assert events[0].at > 0


def test_mode_is_reported_on_the_event():
    m, events = matcher("toggle")
    m.press(CMD)
    m.press(ALT)

    assert events[0].mode == "toggle"


def test_reset_clears_held_keys():
    """After a permission loss or a re-grab, stale held keys would otherwise
    make the next press look like a completion."""
    m, events = matcher("push")
    m.press(CMD)
    m.reset()
    m.press(ALT)

    assert events == []


def test_unknown_mode_is_rejected_at_construction():
    with pytest.raises(ValueError):
        hotkey.HotkeyMatcher(combination=frozenset({CMD}), mode="karaoke", on_event=lambda e: None)


def test_empty_combination_is_rejected():
    with pytest.raises(ValueError):
        hotkey.HotkeyMatcher(combination=frozenset(), mode="push", on_event=lambda e: None)


# --- parsing -----------------------------------------------------------------


def test_parses_a_pynput_style_combination():
    """One slot per component; each slot holds the keys that satisfy it."""
    slots = hotkey.parse_combination("<cmd>+<alt>+d")
    assert len(slots) == 3


def test_malformed_combination_raises_at_parse_not_at_first_press():
    """A typo in the config must fail at startup, when the user is looking."""
    with pytest.raises(hotkey.HotkeyError):
        hotkey.parse_combination("<not-a-key>+q")


def test_empty_combination_string_raises():
    with pytest.raises(hotkey.HotkeyError):
        hotkey.parse_combination("")


# --- the pynput listener -----------------------------------------------------


def test_unavailable_without_accessibility(monkeypatch):
    monkeypatch.setattr(hotkey, "_accessibility_trusted", lambda: False)
    available, reason = hotkey.HotkeyListener(on_event=lambda e: None).is_available()

    assert available is False
    assert "辅助功能" in reason or "Accessibility" in reason


def test_reason_says_what_to_do(monkeypatch):
    """Read by someone whose hotkey silently does nothing."""
    monkeypatch.setattr(hotkey, "_accessibility_trusted", lambda: False)
    _, reason = hotkey.HotkeyListener(on_event=lambda e: None).is_available()

    assert "系统设置" in reason or "System Settings" in reason


def test_available_with_accessibility(monkeypatch):
    monkeypatch.setattr(hotkey, "_accessibility_trusted", lambda: True)
    available, reason = hotkey.HotkeyListener(on_event=lambda e: None).is_available()

    assert available is True
    assert reason == ""


def test_start_refuses_without_permission(monkeypatch):
    """Refusing beats a listener that runs and hears nothing."""
    monkeypatch.setattr(hotkey, "_accessibility_trusted", lambda: False)

    with pytest.raises(hotkey.HotkeyError):
        hotkey.HotkeyListener(on_event=lambda e: None).start()


def test_start_creates_a_listener(monkeypatch, fake_pynput):
    monkeypatch.setattr(hotkey, "_accessibility_trusted", lambda: True)
    listener = hotkey.HotkeyListener(on_event=lambda e: None)
    listener.start()

    assert fake_pynput["listeners"][0].started is True
    listener.close()


def test_close_stops_the_listener(monkeypatch, fake_pynput):
    monkeypatch.setattr(hotkey, "_accessibility_trusted", lambda: True)
    listener = hotkey.HotkeyListener(on_event=lambda e: None)
    listener.start()
    listener.close()

    assert fake_pynput["listeners"][0].stopped is True


def test_close_without_start_is_harmless(monkeypatch):
    hotkey.HotkeyListener(on_event=lambda e: None).close()


def test_listener_is_a_context_manager(monkeypatch, fake_pynput):
    monkeypatch.setattr(hotkey, "_accessibility_trusted", lambda: True)
    with hotkey.HotkeyListener(on_event=lambda e: None):
        pass

    assert fake_pynput["listeners"][0].stopped is True


def test_key_events_reach_the_matcher(monkeypatch, fake_pynput):
    monkeypatch.setattr(hotkey, "_accessibility_trusted", lambda: True)
    events = []
    listener = hotkey.HotkeyListener(combination="<cmd>+<alt>", on_event=events.append)
    listener.start()

    from pynput import keyboard as kb

    stub = fake_pynput["listeners"][0]
    stub.on_press(kb.Key.cmd)
    stub.on_press(kb.Key.alt)

    assert kinds(events) == ["start"]
    listener.close()


@pytest.fixture
def fake_pynput(monkeypatch):
    state = {"listeners": []}

    class FakeListener:
        def __init__(self, on_press=None, on_release=None, **kwargs):
            self.on_press = on_press
            self.on_release = on_release
            self.started = False
            self.stopped = False
            state["listeners"].append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def canonical(self, key):
            return key

    monkeypatch.setattr(hotkey.keyboard, "Listener", FakeListener)
    return state


# --- side-specific modifiers (added after measuring pynput on 2026-08-10) ----


def test_right_option_alone_is_a_valid_combination():
    """The best push-to-talk key available: nothing is bound to right Option,
    and it is comfortable to hold for the length of a sentence."""
    from pynput import keyboard as kb

    slots = hotkey.parse_combination("<alt_r>")
    assert slots == frozenset({frozenset({kb.Key.alt_r})})


def test_side_agnostic_modifier_accepts_either_side():
    from pynput import keyboard as kb

    slot = next(iter(hotkey.parse_combination("<alt>")))
    assert kb.Key.alt_l in slot
    assert kb.Key.alt_r in slot


def test_right_option_does_not_fire_on_left_option():
    """pynput's canonical() folds alt_r into alt, which would make these the
    same key. Measured on macOS: right arrives as Key.alt_r, left as Key.alt."""
    from pynput import keyboard as kb

    events = []
    m = hotkey.HotkeyMatcher(
        combination=hotkey.parse_combination("<alt_r>"), mode="push", on_event=events.append
    )
    m.press(kb.Key.alt)
    assert events == []

    m.press(kb.Key.alt_r)
    assert [e.kind for e in events] == ["start"]


def test_side_agnostic_fires_on_either_side():
    from pynput import keyboard as kb

    for key in (kb.Key.alt_l, kb.Key.alt_r):
        events = []
        m = hotkey.HotkeyMatcher(
            combination=hotkey.parse_combination("<alt>"), mode="push", on_event=events.append
        )
        m.press(key)
        assert [e.kind for e in events] == ["start"], key


def test_modifiers_skip_canonicalisation(monkeypatch, fake_pynput):
    """canonical() would erase the side distinction before the matcher sees it."""
    from pynput import keyboard as kb

    monkeypatch.setattr(hotkey, "_accessibility_trusted", lambda: True)
    listener = hotkey.HotkeyListener(combination="<alt_r>", on_event=lambda e: None)
    listener.start()
    fake_pynput["listeners"][0].canonical = lambda key: kb.Key.alt  # the folding

    assert listener._canonical(kb.Key.alt_r) is kb.Key.alt_r
    listener.close()


# --- double tap --------------------------------------------------------------


def tap(m, keys=(CMD, ALT)):
    for k in keys:
        m.press(k)
    for k in keys:
        m.release(k)


def test_a_single_tap_does_nothing():
    """Single-press toggle on a bare modifier is a trap: right Option is a key
    people brush against, and an accidental toggle silently records whatever is
    said next."""
    m, events = matcher("double_toggle")
    tap(m)
    assert events == []


def test_two_quick_taps_start():
    m, events = matcher("double_toggle")
    tap(m)
    tap(m)
    assert kinds(events) == ["start"]


def test_two_more_taps_stop():
    m, events = matcher("double_toggle")
    for _ in range(4):
        tap(m)
    assert kinds(events) == ["start", "stop"]


def test_slow_taps_do_not_count(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(hotkey.time, "perf_counter", lambda: clock[0])

    m, events = matcher("double_toggle")
    tap(m)
    clock[0] = hotkey.DOUBLE_TAP_SECONDS + 0.1
    tap(m)

    assert events == [], "two presses a second apart are two separate touches"


def test_a_third_tap_does_not_immediately_retrigger():
    """After a double tap fires, the counter resets — otherwise tap-tap-tap
    would start and immediately stop."""
    m, events = matcher("double_toggle")
    tap(m)
    tap(m)
    tap(m)
    assert kinds(events) == ["start"]


def test_double_tap_survives_holding():
    m, events = matcher("double_toggle")
    tap(m)
    m.press(CMD)
    m.press(ALT)
    assert kinds(events) == ["start"]
