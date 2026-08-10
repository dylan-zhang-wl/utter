"""P2a Task 3 — per-stage latency.

Deliberately built before injection. "It feels slow" cannot be acted on;
"injection took 800ms" can. Every task after this one is measurable the day it
lands rather than the week someone gets around to profiling it.

Targets agreed 2026-08-10: 1.5s without polish, 3s with. Past 3s the user starts
doubting whether the thing worked at all.
"""

import pytest

from backend import timing


def test_records_a_stage():
    watch = timing.Stopwatch()
    with watch.span("transcription"):
        pass

    assert [s.name for s in watch.stages] == ["transcription"]


def test_stages_keep_their_order():
    watch = timing.Stopwatch()
    for name in ("capture", "transcription", "injection"):
        with watch.span(name):
            pass

    assert [s.name for s in watch.stages] == ["capture", "transcription", "injection"]


def test_measures_elapsed_time():
    watch = timing.Stopwatch()
    with watch.span("work"):
        sum(range(200_000))

    assert watch.stages[0].ms > 0


def test_uses_a_monotonic_clock(monkeypatch):
    """Wall clock can step backwards mid-dictation — NTP, DST, a manual change —
    and a negative duration in the report would be nonsense."""
    ticks = iter([100.0, 100.5])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks))

    watch = timing.Stopwatch()
    with watch.span("work"):
        pass

    assert watch.stages[0].ms == pytest.approx(500.0)


def test_a_raising_stage_still_closes():
    """Otherwise one failure loses the timings for everything before it."""
    watch = timing.Stopwatch()
    with pytest.raises(ValueError):
        with watch.span("transcription"):
            raise ValueError("model exploded")

    assert [s.name for s in watch.stages] == ["transcription"]
    assert watch.stages[0].failed is True


def test_a_successful_stage_is_not_marked_failed():
    watch = timing.Stopwatch()
    with watch.span("work"):
        pass

    assert watch.stages[0].failed is False


def test_skipped_stage_shows_a_note_not_a_duration():
    """A zero would read as "polish took no time", which is a different and
    wrong claim from "polish did not run"."""
    watch = timing.Stopwatch()
    watch.mark("transcription", 1180.0)
    watch.skip("polish", "off")

    polish_line = next(l for l in watch.report().splitlines() if "polish" in l)
    assert "off" in polish_line
    assert "ms" not in polish_line


def test_skipped_stage_is_not_counted_as_a_measurement():
    watch = timing.Stopwatch()
    watch.skip("polish", "off")

    assert watch.stages[0].skipped is True
    assert watch.stages[0].ms is None


def test_mark_records_an_externally_measured_stage():
    """The hotkey-to-buffer gap is measured from the hotkey event timestamp,
    outside any span."""
    watch = timing.Stopwatch()
    watch.mark("hotkey → buffer closed", 8.0)

    assert watch.stages[0].ms == 8.0


def test_total_sums_the_measured_stages():
    watch = timing.Stopwatch()
    watch.mark("a", 10.0)
    watch.mark("b", 32.0)

    assert watch.total_ms == pytest.approx(42.0)


def test_total_ignores_skipped_stages():
    watch = timing.Stopwatch()
    watch.mark("a", 10.0)
    watch.skip("polish", "off")

    assert watch.total_ms == pytest.approx(10.0)


def test_report_lists_every_stage_with_its_time():
    watch = timing.Stopwatch()
    watch.mark("transcription", 1180.0)
    watch.mark("clipboard + paste", 45.0)

    report = watch.report()
    assert "transcription" in report
    assert "1180" in report
    assert "45" in report


def test_report_includes_the_total():
    watch = timing.Stopwatch()
    watch.mark("transcription", 1180.0)

    assert "total" in watch.report().lower()


def test_report_carries_dropped_chunks():
    """From the bounded queue in Task 1. A drop means audio was thrown away, so
    it cannot be a debug-log-only fact."""
    watch = timing.Stopwatch(dropped_chunks=3)
    watch.mark("transcription", 1180.0)

    report = watch.report()
    assert "3" in report
    assert "丢弃" in report


def test_report_omits_drops_when_there_were_none():
    watch = timing.Stopwatch(dropped_chunks=0)
    watch.mark("transcription", 1180.0)

    assert "丢弃" not in watch.report()


def test_dropped_audio_says_it_was_the_beginning():
    """The user needs to know *which* part went missing. Losing the opening of
    a sentence looks like the model mishearing, not like lost audio."""
    watch = timing.Stopwatch(dropped_chunks=30)
    watch.mark("transcription", 1180.0)

    assert "开头" in watch.report()


def test_report_shows_how_much_audio_was_captured():
    """The only way to tell "the model misheard me" from "the recording was
    shorter than what I said" — which is exactly the confusion that hid a
    ten-second buffer cap for a day."""
    watch = timing.Stopwatch(audio_seconds=8.4)
    watch.mark("transcription", 1180.0)

    assert "8.4" in watch.report()


def test_audio_duration_is_omitted_when_unknown():
    watch = timing.Stopwatch()
    watch.mark("transcription", 1180.0)

    assert "录到音频" not in watch.report()


def test_report_flags_exceeding_the_target():
    """1.5s without polish was the agreed ceiling. Silently printing 2400 ms
    would let a regression pass as normal."""
    watch = timing.Stopwatch(target_ms=1500)
    watch.mark("transcription", 2400.0)

    assert "⚠" in watch.report() or "over" in watch.report().lower()


def test_report_does_not_flag_when_inside_the_target():
    watch = timing.Stopwatch(target_ms=1500)
    watch.mark("transcription", 1180.0)

    assert "⚠" not in watch.report()


def test_report_of_nothing_is_not_a_crash():
    assert timing.Stopwatch().report()


def test_over_target_is_queryable_not_only_printable():
    """The daemon may want to log a warning without parsing its own output."""
    watch = timing.Stopwatch(target_ms=1500)
    watch.mark("transcription", 2400.0)

    assert watch.over_target is True


def test_nested_spans_are_recorded_separately():
    watch = timing.Stopwatch()
    with watch.span("outer"):
        with watch.span("inner"):
            pass

    assert {s.name for s in watch.stages} == {"outer", "inner"}


def test_report_says_where_the_text_went():
    """Until this existed the report showed a duration for an injection that
    might never have happened. Not saying where text went is the same failure
    as losing it, one step later."""
    watch = timing.Stopwatch()
    watch.mark("clipboard + paste", 45.0)
    watch.note_target("Microsoft Word")

    assert "Microsoft Word" in watch.report()


def test_report_says_when_injection_failed_and_why():
    watch = timing.Stopwatch()
    watch.mark("clipboard + paste", 45.0)
    watch.note_target("Safari", failed="focus is on Terminal")

    report = watch.report()
    assert "未注入" in report
    assert "focus is on Terminal" in report
