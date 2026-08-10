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

        if self.dropped_chunks:
            lost = self.dropped_chunks / 10
            lines.append(
                f"  ⚠ 丢弃了 {self.dropped_chunks} 块音频（约 {lost:.1f} 秒，"
                "是你说的话的开头部分）"
            )

        return "\n".join(lines)
