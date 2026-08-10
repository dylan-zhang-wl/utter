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
    target_note: str | None = None
    repetition_note: str | None = None
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
            lines.append(f"  录到音频 {self.audio_seconds:.1f}s")

        if self.target_note:
            lines.append(f"  {self.target_note}")

        if self.repetition_note:
            lines.append(f"  {self.repetition_note}")

        if self.dropped_chunks:
            lost = self.dropped_chunks / 10
            lines.append(
                f"  ⚠ 丢弃了 {self.dropped_chunks} 块音频（约 {lost:.1f} 秒，"
                "是你说的话的开头部分）"
            )

        return "\n".join(lines)
