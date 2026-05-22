from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from src.baseline_hook import BaselineCheckpointConfig, BaselineCheckpointHook


@dataclass
class AsyncOCheckpointResult:
    step: int
    tag: str
    path: Path
    transfer_duration_sec: float | None = None
    transfer_sync_duration_sec: float | None = None
    transfer_full_duration_sec: float | None = None
    transfer_enqueue_started_at: float | None = None
    persistence_duration_sec: float | None = None
    total_duration_sec: float | None = None


class AsyncOCheckpointHook(BaselineCheckpointHook):
    """
    Async checkpoint with optimizer-phase barrier.

    Behavior:
    1. `save_checkpoint(step)` schedules GPU->CPU transfer of the checkpoint
       payload on a separate CUDA stream and returns immediately.
    2. That transfer is allowed to overlap with the same step's forward and
       backward passes because those phases do not mutate model / optimizer
       state.
    3. `backward_end(step)` is the synchronization barrier. If the transfer for
       that same step has not finished yet, training waits there before
       optimizer state mutation begins.
    4. After transfer completes, checkpoint persistence continues in a
       background CPU thread via `torch.save`.

    This keeps checkpoints loadable because the serialized payload uses the same
    structure as the baseline hook.
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
        self.last_result: AsyncOCheckpointResult | None = None
        self.history: list[AsyncOCheckpointResult] = []

        self._pending_thread: threading.Thread | None = None
        self._pending_result: AsyncOCheckpointResult | None = None
        self._pending_error: BaseException | None = None
        self._pending_started_step: int | None = None
        self._pending_transfer_event: torch.cuda.Event | None = None
        self._pending_lock = threading.Lock()
        self._transfer_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )

    def save_checkpoint(self, step: int) -> None:
        self._join_previous_persist_if_needed()

        tag = f"{self.config.tag_prefix}_{step}"
        path = self._checkpoint_path(step)
        result = AsyncOCheckpointResult(
            step=step,
            tag=path.stem,
            path=path,
        )

        checkpoint, transfer_event = self._build_cpu_checkpoint_async(step, tag, result)
        worker = threading.Thread(
            target=self._persist_checkpoint_worker,
            args=(checkpoint, path, result, transfer_event),
            daemon=True,
            name=f"async-o-checkpoint-{step}",
        )

        with self._pending_lock:
            self._pending_thread = worker
            self._pending_result = result
            self._pending_error = None
            self._pending_started_step = step
            self._pending_transfer_event = transfer_event

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
        pending_started_step = self._pending_started_step
        if pending_started_step is not None and step == pending_started_step:
            self.wait_for_pending_transfer()

    def update_begin(self, step: int) -> None:
        return

    def update_end(self, step: int) -> None:
        return

    def wait_for_pending_transfer(self) -> None:
        event: torch.cuda.Event | None
        result: AsyncOCheckpointResult | None
        with self._pending_lock:
            event = self._pending_transfer_event
            result = self._pending_result

        if event is None:
            return

        sync_start = time.perf_counter()
        event.synchronize()
        sync_duration = time.perf_counter() - sync_start

        if result is not None:
            result.transfer_sync_duration_sec = sync_duration
            if result.transfer_enqueue_started_at is not None:
                result.transfer_full_duration_sec = (
                    time.perf_counter() - result.transfer_enqueue_started_at
                )

        with self._pending_lock:
            if self._pending_transfer_event is event:
                self._pending_transfer_event = None

    def wait_for_pending_persistence(self) -> None:
        thread: threading.Thread | None
        with self._pending_lock:
            thread = self._pending_thread

        if thread is None:
            return

        thread.join()
        self._finalize_pending_thread(thread)

    def transfer_timing_summary(self) -> dict[str, float | None]:
        total_enqueue = 0.0
        total_sync = 0.0
        total_full = 0.0
        count = 0

        for result in self.history:
            if result.transfer_duration_sec is not None:
                total_enqueue += result.transfer_duration_sec
            if result.transfer_sync_duration_sec is not None:
                total_sync += result.transfer_sync_duration_sec
            if result.transfer_full_duration_sec is not None:
                total_full += result.transfer_full_duration_sec
            count += 1

        return {
            "mo_foreground_avg_sec": (
                (total_enqueue + total_sync) / count if count > 0 else None
            ),
            "mo_full_avg_sec": (
                total_full / count if count > 0 else None
            ),
            "mo_foreground_total_sec": total_enqueue + total_sync,
            "mo_full_total_sec": total_full,
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
            self._pending_transfer_event = None

        if result is not None:
            self.last_result = result
            self.history.append(result)

        if error is not None:
            raise RuntimeError("Background checkpoint persistence failed.") from error

    def _persist_checkpoint_worker(
        self,
        checkpoint: dict[str, Any],
        checkpoint_path: Path,
        result: AsyncOCheckpointResult,
        transfer_event: torch.cuda.Event | None,
    ) -> None:
        error: BaseException | None = None

        try:
            if transfer_event is not None:
                transfer_event.synchronize()

            persistence_start = time.perf_counter()
            torch.save(checkpoint, checkpoint_path)
            result.persistence_duration_sec = time.perf_counter() - persistence_start
        except BaseException as exc:
            error = exc

        if (
            result.transfer_duration_sec is not None
            and result.persistence_duration_sec is not None
        ):
            result.total_duration_sec = (
                result.transfer_duration_sec + result.persistence_duration_sec
            )

        with self._pending_lock:
            self._pending_error = error

    def _build_cpu_checkpoint_async(
        self,
        step: int,
        tag: str,
        result: AsyncOCheckpointResult,
    ) -> tuple[dict[str, Any], torch.cuda.Event | None]:
        checkpoint = self._build_checkpoint(step, tag)

        if self._transfer_stream is None:
            transfer_start = time.perf_counter()
            cpu_checkpoint = self._clone_to_cpu_sync(checkpoint)
            duration = time.perf_counter() - transfer_start
            result.transfer_duration_sec = duration
            result.transfer_enqueue_started_at = transfer_start
            result.transfer_full_duration_sec = duration
            result.transfer_sync_duration_sec = 0.0
            return cpu_checkpoint, None

        transfer_start = time.perf_counter()
        current_stream = torch.cuda.current_stream()
        transfer_stream = self._transfer_stream
        transfer_stream.wait_stream(current_stream)

        with torch.cuda.stream(transfer_stream):
            cpu_checkpoint = self._clone_to_cpu_async(checkpoint)

        event = torch.cuda.Event()
        event.record(transfer_stream)

        result.transfer_duration_sec = time.perf_counter() - transfer_start
        result.transfer_enqueue_started_at = transfer_start
        return cpu_checkpoint, event

    def _clone_to_cpu_sync(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()

        if isinstance(value, dict):
            return {key: self._clone_to_cpu_sync(item) for key, item in value.items()}

        if isinstance(value, list):
            return [self._clone_to_cpu_sync(item) for item in value]

        if isinstance(value, tuple):
            return tuple(self._clone_to_cpu_sync(item) for item in value)

        return value

    def _clone_to_cpu_async(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.device.type != "cuda":
                return tensor.cpu().clone()

            cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
            cpu_tensor.copy_(tensor, non_blocking=True)
            return cpu_tensor

        if isinstance(value, dict):
            return {key: self._clone_to_cpu_async(item) for key, item in value.items()}

        if isinstance(value, list):
            return [self._clone_to_cpu_async(item) for item in value]

        if isinstance(value, tuple):
            return tuple(self._clone_to_cpu_async(item) for item in value)

        return value
