from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from src.baseline_hook import BaselineCheckpointConfig, BaselineCheckpointHook


@dataclass
class AsyncCheckpointResult:
    step: int
    tag: str
    path: Path
    transfer_duration_sec: float
    persistence_duration_sec: float | None = None
    total_duration_sec: float | None = None


class AsyncCheckpointHook(BaselineCheckpointHook):
    """
    Async checkpoint baseline:

    1. Foreground training blocks while model / optimizer checkpoint state is
       transferred from GPU-backed tensors into CPU-owned tensors.
    2. Once the CPU checkpoint payload is complete, training resumes.
    3. torch.save runs on a background thread against that CPU payload.
    4. Before starting another checkpoint, the hook waits for previous
       background persistence to finish. This prevents concurrent checkpoint
       writes without blocking unrelated forward/backward work.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        config: BaselineCheckpointConfig | None = None,
        checkpoint_builder: Callable[[int], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            config=config,
            checkpoint_builder=checkpoint_builder,
        )
        self.last_result: AsyncCheckpointResult | None = None
        self.history: list[AsyncCheckpointResult] = []

        self._pending_thread: threading.Thread | None = None
        self._pending_result: AsyncCheckpointResult | None = None
        self._pending_error: BaseException | None = None
        self._pending_started_step: int | None = None
        self._pending_lock = threading.Lock()

    def save_checkpoint(self, step: int) -> None:
        self._join_previous_persist_if_needed()

        tag = f"{self.config.tag_prefix}_{step}"
        path = self._checkpoint_path(step)

        transfer_start = time.perf_counter()
        checkpoint = self._build_cpu_checkpoint(step, tag)
        transfer_duration = time.perf_counter() - transfer_start

        result = AsyncCheckpointResult(
            step=step,
            tag=path.stem,
            path=path,
            transfer_duration_sec=transfer_duration,
        )
        worker = threading.Thread(
            target=self._persist_checkpoint_worker,
            args=(checkpoint, path, result),
            daemon=True,
            name=f"async-checkpoint-{step}",
        )

        with self._pending_lock:
            self._pending_thread = worker
            self._pending_result = result
            self._pending_error = None
            self._pending_started_step = step

        worker.start()

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        load_model: bool = True,
        load_optimizer: bool = True,
        load_rng_state: bool = True,
    ) -> dict[str, Any]:
        self.wait_for_pending_persistence()
        return super().load_checkpoint(
            checkpoint_path,
            map_location=map_location,
            load_model=load_model,
            load_optimizer=load_optimizer,
            load_rng_state=load_rng_state,
        )

    def forward_begin(self, step: int) -> None:
        return

    def forward_end(self, step: int) -> None:
        return

    def backward_begin(self, step: int) -> None:
        return

    def backward_end(self, step: int) -> None:
        return

    def update_begin(self, step: int) -> None:
        return

    def update_end(self, step: int) -> None:
        return

    def wait_for_pending_persistence(self) -> None:
        thread: threading.Thread | None
        with self._pending_lock:
            thread = self._pending_thread

        if thread is None:
            return

        thread.join()
        self._finalize_pending_thread(thread)

    def transfer_timing_summary(self) -> dict[str, float | None]:
        total_transfer = 0.0
        count = 0
        for result in self.history:
            total_transfer += result.transfer_duration_sec
            count += 1

        return {
            "mo_foreground_avg_sec": (
                total_transfer / count if count > 0 else None
            ),
            "mo_full_avg_sec": (
                total_transfer / count if count > 0 else None
            ),
            "mo_foreground_total_sec": total_transfer,
            "mo_full_total_sec": total_transfer,
            "mo_count": count,
            "gradient_foreground_avg_sec": None,
            "gradient_full_avg_sec": None,
            "gradient_foreground_total_sec": 0.0,
            "gradient_full_total_sec": 0.0,
            "gradient_count": 0,
        }

    def _join_previous_persist_if_needed(self) -> None:
        thread: threading.Thread | None
        with self._pending_lock:
            thread = self._pending_thread

        if thread is None:
            return

        if thread.is_alive():
            thread.join()

        self._finalize_pending_thread(thread)

    def _finalize_pending_thread(self, thread: threading.Thread) -> None:
        with self._pending_lock:
            if self._pending_thread is not thread:
                return

            error = self._pending_error
            result = self._pending_result

            self._pending_thread = None
            self._pending_result = None
            self._pending_error = None
            self._pending_started_step = None

        if result is not None:
            self.last_result = result
            self.history.append(result)

        if error is not None:
            raise RuntimeError("Background checkpoint persistence failed.") from error

    def _persist_checkpoint_worker(
        self,
        checkpoint: dict[str, Any],
        checkpoint_path: Path,
        result: AsyncCheckpointResult,
    ) -> None:
        persistence_start = time.perf_counter()
        error: BaseException | None = None

        try:
            torch.save(checkpoint, checkpoint_path)
        except BaseException as exc:
            error = exc

        persistence_duration = time.perf_counter() - persistence_start
        result.persistence_duration_sec = persistence_duration
        result.total_duration_sec = result.transfer_duration_sec + persistence_duration

        with self._pending_lock:
            self._pending_error = error

    def _build_cpu_checkpoint(self, step: int, tag: str) -> dict[str, Any]:
        checkpoint = self._build_checkpoint(step, tag)
        return self._clone_to_cpu(checkpoint)

    def _clone_to_cpu(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()

        if isinstance(value, dict):
            return {key: self._clone_to_cpu(item) for key, item in value.items()}

        if isinstance(value, list):
            return [self._clone_to_cpu(item) for item in value]

        if isinstance(value, tuple):
            return tuple(self._clone_to_cpu(item) for item in value)

        return value
