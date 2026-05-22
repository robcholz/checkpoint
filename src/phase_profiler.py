from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from src.pytorch_hook import PyTorchCheckpointHook


@dataclass
class PhaseRecord:
    phase: str
    step: int | None
    duration_sec: float


class PhaseProfiler:
    def __init__(self) -> None:
        self.records: list[PhaseRecord] = []

    def time_phase(self, phase: str, step: int | None, fn: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        try:
            return fn()
        finally:
            self.records.append(
                PhaseRecord(
                    phase=phase,
                    step=step,
                    duration_sec=time.perf_counter() - start,
                )
            )

    def summary(self) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for record in self.records:
            grouped[record.phase].append(record.duration_sec)

        result: dict[str, dict[str, float | int]] = {}
        for phase, durations in grouped.items():
            total = sum(durations)
            result[phase] = {
                "count": len(durations),
                "total_sec": total,
                "avg_sec": total / len(durations),
                "max_sec": max(durations),
            }
        return result

    def as_dicts(self) -> list[dict[str, float | int | str | None]]:
        return [
            {
                "phase": record.phase,
                "step": record.step,
                "duration_sec": record.duration_sec,
            }
            for record in self.records
        ]


class PhaseProfilingHook(PyTorchCheckpointHook):
    def __init__(
        self, wrapped: PyTorchCheckpointHook, profiler: PhaseProfiler | None = None
    ) -> None:
        self.wrapped = wrapped
        self.profiler = profiler or PhaseProfiler()
        self.raw_foreground_profiler = PhaseProfiler()
        self._raw_foreground_step: int | None = None
        self._raw_foreground_start: float | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    @property
    def history(self) -> Any:
        return self.wrapped.history

    @property
    def last_result(self) -> Any:
        return self.wrapped.last_result

    def _begin_raw_foreground(self, step: int) -> None:
        self._raw_foreground_step = step
        self._raw_foreground_start = time.perf_counter()

    def _end_raw_foreground(self, phase: str) -> None:
        if self._raw_foreground_start is not None:
            duration = time.perf_counter() - self._raw_foreground_start
            self.raw_foreground_profiler.records.append(
                PhaseRecord(
                    phase=phase,
                    step=self._raw_foreground_step,
                    duration_sec=duration,
                )
            )
        self._raw_foreground_start = None
        self._raw_foreground_step = None

    def save_checkpoint(self, step: int) -> None:
        return self.profiler.time_phase(
            "hook.save_checkpoint",
            step,
            lambda: self.wrapped.save_checkpoint(step),
        )

    def load_checkpoint(self, checkpoint_path, *, map_location=None, **kwargs):
        return self.profiler.time_phase(
            "hook.load_checkpoint",
            None,
            lambda: self.wrapped.load_checkpoint(
                checkpoint_path,
                map_location=map_location,
                **kwargs,
            ),
        )

    def forward_begin(self, step: int) -> None:
        result = self.profiler.time_phase(
            "hook.forward_begin",
            step,
            lambda: self.wrapped.forward_begin(step),
        )
        self._begin_raw_foreground(step)
        return result

    def forward_end(self, step: int) -> None:
        self._end_raw_foreground("raw_foreground_forward")
        return self.profiler.time_phase(
            "hook.forward_end",
            step,
            lambda: self.wrapped.forward_end(step),
        )

    def backward_begin(self, step: int) -> None:
        result = self.profiler.time_phase(
            "hook.backward_begin",
            step,
            lambda: self.wrapped.backward_begin(step),
        )
        self._begin_raw_foreground(step)
        return result

    def backward_end(self, step: int) -> None:
        self._end_raw_foreground("raw_foreground_backward")
        return self.profiler.time_phase(
            "hook.backward_end",
            step,
            lambda: self.wrapped.backward_end(step),
        )

    def update_begin(self, step: int) -> None:
        result = self.profiler.time_phase(
            "hook.update_begin",
            step,
            lambda: self.wrapped.update_begin(step),
        )
        self._begin_raw_foreground(step)
        return result

    def update_end(self, step: int) -> None:
        self._end_raw_foreground("raw_foreground_update")
        return self.profiler.time_phase(
            "hook.update_end",
            step,
            lambda: self.wrapped.update_end(step),
        )

    def wait_for_pending_persistence(self) -> None:
        wait_fn = getattr(self.wrapped, "wait_for_pending_persistence", None)
        if wait_fn is None:
            return None

        return self.profiler.time_phase(
            "hook.wait_for_pending_persistence",
            None,
            wait_fn,
        )
