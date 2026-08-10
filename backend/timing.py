"""Where the time went, per dictation.

Built before injection on purpose. "It feels slow" is not actionable;
"clipboard + paste took 800 ms" is. Landing the instrument first means every
later piece of P2a can be judged the day it is written.

Targets agreed with the author on 2026-08-10: **1.5 s without polish, 3 s with**.
Past three seconds the user stops trusting that it worked and presses the hotkey
again, which is its own failure.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Stage:
    name: str
    ms: float | None  # None means it did not run
    note: str = ""
    failed: bool = False

    @property
    def skipped(self) -> bool:
        return self.ms is None


@dataclass
class Stopwatch:
    dropped_chunks: int = 0
    audio_seconds: float | None = None
    held_seconds: float | None = None
    """Wall clock from key down to key up. Without it, `audio_seconds` cannot be
    read: 22.6 seconds of audio is either a 22.6-second hold that worked or a
    60-second hold that lost two thirds of itself, and those need opposite fixes."""
    speech_seconds: float | None = None
    """How much of that audio the VAD heard anyone talking in. Separates "the
    model dropped what I said" from "the recording contains three seconds of
    speech and twenty of me holding the key while thinking"."""
    overflows: int = 0
    """PortAudio input overflows — audio the operating system discarded before
    our callback ran. Counted since day one and never once shown, which is the
    silent loss this project keeps promising not to do."""
    target_note: str | None = None
    repetition_note: str | None = None
    segments_note: str | None = None
    target_ms: float | None = None
    stages: list[Stage] = field(default_factory=list)

    @contextmanager
    def span(self, name: str):
        """Time a block. A raising block is still recorded, and marked failed.

        Otherwise one failure would erase the timings of everything before it,
        exactly when you most want to know where the time went.
        """
        started = time.perf_counter()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            self.stages.append(Stage(name=name, ms=elapsed, failed=failed))

    def note_target(self, name: str, failed: str = "") -> None:
        """Record where the text was sent, and whether it arrived."""
        self.target_note = f"注入 → {name}" if not failed else f"⚠ 未注入 → {name}：{failed}"

    def note_route(self, *, pressed, target, before, after, result) -> None:
        """The whole journey on one line.

        Four facts decide where dictated text lands — which app was frontmost
        when the key went down, which one the injector was aiming at, which was
        frontmost when it fired, and which ended up frontmost after. The daemon
        knew all four and printed none, so several rounds went into inferring
        them from the outside.
        """
        verdict = "✅ 已注入" if result.injected else (
            "⏸ 已缓冲" if result.buffered else "⚠ 未注入")
        self.target_note = (
            f"按下时 {pressed or '?'} → 目标 {target or '?'} → "
            f"注入前前台 {before or '?'} → 注入后前台 {after or '?'}\n"
            f"  {verdict}" + (f"：{result.reason}" if result.reason else "")
        )

    def note_segments(self, count: int) -> None:
        self.segments_note = f"长句在停顿处切成 {count} 小段分别转录（这样标点才有依据）"

    def note_repetition(self, removed: int) -> None:
        self.repetition_note = (
            f"⚠ 模型复读了，已折叠 {removed} 处重复。"
            "这段建议重说一遍——原始输出在存档里"
        )

    def mark(self, name: str, ms: float) -> None:
        """Record a stage measured elsewhere — the hotkey gap, for instance,
        which is the delta between two event timestamps rather than a block."""
        self.stages.append(Stage(name=name, ms=ms))

    def skip(self, name: str, note: str) -> None:
        """Record that a stage did not run.

        Not the same as recording zero: "polish took 0 ms" and "polish is off"
        are different claims, and only one of them is true.
        """
        self.stages.append(Stage(name=name, ms=None, note=note))

    @property
    def total_ms(self) -> float:
        return sum(s.ms for s in self.stages if s.ms is not None)

    @property
    def over_target(self) -> bool:
        return self.target_ms is not None and self.total_ms > self.target_ms

    def report(self) -> str:
        if not self.stages:
            return "no stages recorded"

        width = max(len(s.name) for s in self.stages)
        width = max(width, len("total"))

        lines = []
        for stage in self.stages:
            if stage.skipped:
                value = f"({stage.note})"
            else:
                value = f"{stage.ms:.0f} ms"
                if stage.failed:
                    value += "  [failed]"
            lines.append(f"  {stage.name:<{width}}  {value:>12}")

        lines.append("  " + "─" * (width + 14))
        # The warning goes after the aligned field, not inside it, so an
        # over-target run does not knock the column out of line — which is
        # precisely the run you want to read at a glance.
        total = f"  {'total':<{width}}  {f'{self.total_ms:.0f} ms':>12}"
        if self.over_target:
            total += f"   ⚠ over {self.target_ms:.0f} ms"
        lines.append(total)

        if self.audio_seconds is not None:
            # Printed on every dictation, not only on failure. It is the only
            # way the user can tell "the model misheard me" apart from "the
            # recording was shorter than what I said".
            line = f"  录到音频 {self.audio_seconds:.1f}s"
            if self.held_seconds is not None:
                line += f"（按住 {self.held_seconds:.1f}s"
                if self.speech_seconds is not None:
                    line += f"，其中说话 {self.speech_seconds:.1f}s"
                line += "）"
            elif self.speech_seconds is not None:
                line += f"（其中说话 {self.speech_seconds:.1f}s）"
            lines.append(line)

            # The microphone opens ~220ms after the key goes down (audio_source
            # measured it), so a small gap is expected and not worth shouting
            # about. A whole second is not: that is speech that was said and
            # never recorded, and it looks from the outside exactly like the
            # model failing.
            if self.held_seconds is not None and self.held_seconds - self.audio_seconds > 1.0:
                missing = self.held_seconds - self.audio_seconds
                lines.append(
                    f"  ⚠ 有 {missing:.1f}s 没录进来（按住的时间比录到的音频长这么多）。"
                    "这不是模型的问题，是录音丢了"
                )

        if self.overflows:
            lines.append(
                f"  ⚠ 系统丢了 {self.overflows} 次输入缓冲（录音线程没跟上）"
            )

        if self.target_note:
            lines.append(f"  {self.target_note}")

        if self.segments_note:
            lines.append(f"  {self.segments_note}")

        if self.repetition_note:
            lines.append(f"  {self.repetition_note}")

        if self.dropped_chunks:
            lost = self.dropped_chunks / 10
            lines.append(
                f"  ⚠ 丢弃了 {self.dropped_chunks} 块音频（约 {lost:.1f} 秒，"
                "是你说的话的开头部分）"
            )

        return "\n".join(lines)
