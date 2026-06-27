from __future__ import annotations

import threading
import time
import os
import warnings
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch.optim import _functional as optim_functional

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
except Exception:  # pragma: no cover - optional acceleration backend
    DeepSpeedCPUAdam = None

from src.baseline_hook import BaselineCheckpointConfig, BaselineCheckpointHook
from src.pytorch_hook import (
    CheckpointRequest,
    OptimizerParamSnapshot,
    ParameterSnapshot,
)
from src.rust_replay import adamw_update as rust_adamw_update


@dataclass
class GoCkptCheckpointConfig(BaselineCheckpointConfig):
    overlap_steps: int = 7
    reconstruction_queue_depth: int | None = None
    transfer_chunk_mb: float = 0.0


@dataclass
class GoCkptCheckpointResult:
    start_step: int
    target_step: int
    tag: str
    path: Path
    transfer_duration_sec: float = 0.0
    transfer_sync_duration_sec: float = 0.0
    transfer_count: int = 0
    gradient_duration_sec: float = 0.0
    gradient_submit_duration_sec: float = 0.0
    gradient_sync_duration_sec: float = 0.0
    gradient_count: int = 0
    reconstruction_duration_sec: float = 0.0
    reconstruction_backpressure_sec: float = 0.0
    persistence_duration_sec: float | None = None
    total_duration_sec: float | None = None


@dataclass
class GoCkptAbandonedWindow:
    start_step: int
    target_step: int
    tag: str
    reason: str
    transferred_partitions: int
    total_partitions: int
    reconstructed_blocks: int
    total_blocks: int


@dataclass
class PendingCheckpointWorker:
    result: GoCkptCheckpointResult
    thread: threading.Thread | None = None
    error: BaseException | None = None


@dataclass
class GoCkptPendingGradientTransfer:
    step: int
    gradients: dict[str, torch.Tensor | None]
    source_refs: dict[str, torch.Tensor]
    event: torch.cuda.Event | None
    submitted_at: float
    submit_duration_sec: float = 0.0


@dataclass
class FlatReplayBlock:
    partition_index: int
    names: list[str]
    offsets: dict[str, tuple[int, int]]
    param_buffer: torch.Tensor
    exp_avg_buffer: torch.Tensor
    exp_avg_sq_buffer: torch.Tensor


@dataclass
class GoCkptRuntime:
    request: CheckpointRequest
    partitions: list[list[str]]
    partition_name_to_index: dict[str, int]
    transferred_partitions: set[int] = field(default_factory=set)
    partition_events: dict[int, torch.cuda.Event | None] = field(default_factory=dict)
    transferred_blocks: dict[str, ParameterSnapshot] = field(default_factory=dict)
    optimizer_param_groups_by_step: dict[int, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )
    flat_replay_blocks_by_name: dict[str, FlatReplayBlock] = field(default_factory=dict)
    flattened_partitions: set[int] = field(default_factory=set)
    flatten_lock: threading.Lock = field(default_factory=threading.Lock)
    result: GoCkptCheckpointResult | None = None
    reconstruction_executor: ThreadPoolExecutor | None = None
    reconstruction_futures: list[Future] = field(default_factory=list)
    partition_futures: dict[int, Future] = field(default_factory=dict)
    reconstruction_slots: threading.Semaphore | None = None
    reconstruction_error: BaseException | None = None
    reconstruction_lock: threading.Lock = field(default_factory=threading.Lock)
    reconstruction_started_at: float | None = None
    reconstruction_finished_at: float | None = None
    # Background transfer threads for overlapping with forward/backward
    partition_transfer_threads: dict[int, threading.Thread] = field(default_factory=dict)
    partition_transfer_errors: dict[int, BaseException | None] = field(default_factory=dict)
    transfer_blocks_lock: threading.Lock = field(default_factory=threading.Lock)


class GoCkptCheckpointHook(BaselineCheckpointHook):
    """
    Multi-step overlapped checkpointing with CPU-side reconstruction.

    One checkpoint request spans several training steps:
    - each step transfers one partition of model + optimizer state
    - early partitions capture gradients over subsequent steps
    - CPU replays AdamW updates to bring all partitions to the same target step
    - once all partitions reach the target version, a consistent checkpoint is
      persisted in the background
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
        if self.optimizer is None:
            raise ValueError("GoCkptCheckpointHook requires an optimizer.")

        self.overlap_steps = int(getattr(self.config, "overlap_steps", 7))
        if self.overlap_steps <= 0:
            raise ValueError("overlap_steps must be positive.")

        self.last_result: GoCkptCheckpointResult | None = None
        self.history: list[GoCkptCheckpointResult] = []
        self.abandoned_windows: list[GoCkptAbandonedWindow] = []

        self._active_runtime: GoCkptRuntime | None = None
        self._pending_workers: list[PendingCheckpointWorker] = []
        self._pending_lock = threading.Lock()
        self._persist_lock = threading.Lock()
        self._transfer_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self._gradient_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self._pending_gradient_refs: dict[int, dict[str, torch.Tensor | None]] = {}
        self._pending_gradient_transfers: dict[int, GoCkptPendingGradientTransfer] = {}
        self._ds_cpu_adam_local = threading.local()
        self._ringbuffer_pressure_lock = threading.Lock()
        self._ringbuffer_pressure_t0 = time.perf_counter()
        self._ringbuffer_total_capacity = 0
        self._ringbuffer_inflight = 0
        self.ringbuffer_pressure_samples: list[dict[str, float | int | str]] = []

        self._named_parameters = list(self.model.named_parameters())
        self._params_by_name = {name: param for name, param in self._named_parameters}
        self._param_name_by_obj = {
            id(param): name for name, param in self._named_parameters
        }
        self._buffers_by_name = dict(self.model.named_buffers())

        self._param_to_group: dict[int, dict[str, Any]] = {}
        self._param_to_state_id: dict[int, int] = {}
        self._build_optimizer_metadata()
        self._reconstruction_workers = max(
            1,
            min(self.overlap_steps, os.cpu_count() or self.overlap_steps),
        )
        queue_depth = getattr(self.config, "reconstruction_queue_depth", None)
        env_queue_depth = os.environ.get("GOCKPT_RECONSTRUCTION_QUEUE_DEPTH")
        if env_queue_depth is not None:
            queue_depth = int(env_queue_depth)
        if queue_depth is None:
            queue_depth = self._reconstruction_workers * 4
        self._reconstruction_queue_depth = int(queue_depth)
        if self._reconstruction_queue_depth <= 0:
            raise ValueError("reconstruction_queue_depth must be positive.")

        transfer_chunk_mb = float(getattr(self.config, "transfer_chunk_mb", 0.0))
        env_transfer_chunk_mb = os.environ.get("GOCKPT_TRANSFER_CHUNK_MB")
        if env_transfer_chunk_mb is not None:
            transfer_chunk_mb = float(env_transfer_chunk_mb)
        if transfer_chunk_mb < 0:
            raise ValueError("transfer_chunk_mb must be >= 0.")
        self._transfer_chunk_mb = transfer_chunk_mb
        self._transfer_chunk_bytes = int(transfer_chunk_mb * 1024 * 1024)

    def set_training_start_time(self, start_time: float) -> None:
        with self._ringbuffer_pressure_lock:
            self._ringbuffer_pressure_t0 = start_time
            self.ringbuffer_pressure_samples.clear()
            self._sample_ringbuffer_pressure_locked("training_start")

    def _sample_ringbuffer_pressure_locked(self, event: str) -> None:
        capacity = self._ringbuffer_total_capacity
        inflight = self._ringbuffer_inflight
        pressure = (inflight / capacity) if capacity > 0 else 0.0
        self.ringbuffer_pressure_samples.append(
            {
                "time_sec": time.perf_counter() - self._ringbuffer_pressure_t0,
                "event": event,
                "inflight": inflight,
                "capacity": capacity,
                "pressure": pressure,
                "pressure_percent": pressure * 100.0,
            }
        )

    def _add_ringbuffer_capacity(self, capacity: int) -> None:
        with self._ringbuffer_pressure_lock:
            self._ringbuffer_total_capacity += capacity
            self._sample_ringbuffer_pressure_locked("queue_start")

    def _remove_ringbuffer_capacity(self, capacity: int) -> None:
        with self._ringbuffer_pressure_lock:
            self._ringbuffer_total_capacity = max(
                0, self._ringbuffer_total_capacity - capacity
            )
            if self._ringbuffer_inflight > self._ringbuffer_total_capacity:
                self._ringbuffer_inflight = self._ringbuffer_total_capacity
            self._sample_ringbuffer_pressure_locked("queue_stop")

    def _increment_ringbuffer_inflight(self) -> None:
        with self._ringbuffer_pressure_lock:
            self._ringbuffer_inflight += 1
            self._sample_ringbuffer_pressure_locked("enqueue")

    def _decrement_ringbuffer_inflight(self) -> None:
        with self._ringbuffer_pressure_lock:
            self._ringbuffer_inflight = max(0, self._ringbuffer_inflight - 1)
            self._sample_ringbuffer_pressure_locked("complete")

    def save_checkpoint(self, step: int) -> None:
        self._join_previous_persist_if_needed()
        if self._active_runtime is not None:
            raise RuntimeError(
                "GoCkpt does not support overlapping checkpoint requests. "
                "Ensure the checkpoint interval exceeds overlap_steps."
            )

        target_step = step + self.overlap_steps
        tag = f"{self.config.tag_prefix}_{target_step}"
        path = self._checkpoint_path(target_step)

        request = CheckpointRequest(
            start_step=step,
            target_step=target_step,
            tag=tag,
        )
        partitions = self._build_partitions()
        partition_name_to_index = {
            name: partition_index
            for partition_index, partition in enumerate(partitions)
            for name in partition
        }
        result = GoCkptCheckpointResult(
            start_step=step,
            target_step=target_step,
            tag=tag,
            path=path,
        )
        self._active_runtime = GoCkptRuntime(
            request=request,
            partitions=partitions,
            partition_name_to_index=partition_name_to_index,
            result=result,
        )
        self._start_reconstruction_worker(self._active_runtime)

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        load_model: bool = True,
        load_optimizer: bool = True,
        load_rng_state: bool = True,
    ) -> dict[str, Any]:
        if self._active_runtime is not None:
            raise RuntimeError(
                "Cannot load a checkpoint while a GoCkpt request is still active."
            )
        self.wait_for_pending_persistence()
        return super().load_checkpoint(
            checkpoint_path,
            map_location=map_location,
            load_model=load_model,
            load_optimizer=load_optimizer,
            load_rng_state=load_rng_state,
        )

    def forward_begin(self, step: int) -> None:
        runtime = self._active_runtime
        if runtime is None or not self._is_step_in_request(runtime, step):
            return

        partition_index = step - runtime.request.start_step
        if partition_index >= len(runtime.partitions):
            return
        if partition_index in runtime.transferred_partitions:
            return

        # Spawn background thread for transfer so forward pass can proceed immediately
        # This overlaps GPU->CPU transfer with forward+backward computation
        self._spawn_partition_transfer_thread(runtime, partition_index, step)

    def forward_end(self, step: int) -> None:
        return

    def backward_begin(self, step: int) -> None:
        return

    def backward_end(self, step: int) -> None:
        runtime = self._active_runtime
        if runtime is None or not self._is_step_in_request(runtime, step):
            return

        # Wait for current step's partition transfer thread to complete
        # before capturing gradients, otherwise we might miss blocks
        partition_index = step - runtime.request.start_step
        if 0 <= partition_index < len(runtime.partitions):
            self._wait_partition_transfer_thread(runtime, partition_index)

        gradients_for_step: dict[str, torch.Tensor | None] = {}
        # Thread-safe copy of transferred_blocks to avoid iteration issues
        # while background transfer thread may be adding to it
        with runtime.transfer_blocks_lock:
            transferred_blocks_snapshot = dict(runtime.transferred_blocks)

        for name, snapshot in transferred_blocks_snapshot.items():
            if snapshot.version_step > step:
                continue

            param = self._params_by_name[name]
            grad = param.grad
            if grad is None:
                gradients_for_step[name] = None
                continue

            gradients_for_step[name] = grad.detach()

        if not gradients_for_step:
            return

        self._pending_gradient_refs[step] = gradients_for_step

    def update_begin(self, step: int) -> None:
        runtime = self._active_runtime
        if runtime is None or not self._is_step_in_request(runtime, step):
            return

        runtime.optimizer_param_groups_by_step[step] = (
            self._snapshot_optimizer_param_groups_by_name()
        )
        gradients_for_step = self._pending_gradient_refs.pop(step, None)
        if gradients_for_step:
            self._pending_gradient_transfers[step] = (
                self._submit_async_gradient_transfer(step, gradients_for_step)
            )
        self._wait_current_partition_transfer_before_update(runtime, step)

    def update_end(self, step: int) -> None:
        runtime = self._active_runtime
        if runtime is None:
            return

        if self._is_step_in_request(runtime, step):
            self._finish_pending_gradient_transfer(runtime, step)

        if step != runtime.request.target_step - 1:
            return

        result = runtime.result
        if result is None:
            raise RuntimeError("GoCkpt runtime is missing checkpoint result metadata.")

        pending = PendingCheckpointWorker(result=result)
        worker = threading.Thread(
            target=self._finalize_checkpoint_worker,
            args=(runtime, pending),
            daemon=True,
            name=f"gockpt-finalize-{result.target_step}",
        )
        pending.thread = worker
        with self._pending_lock:
            self._pending_workers.append(pending)

        self._active_runtime = None
        worker.start()

    def wait_for_pending_persistence(self) -> None:
        if self._active_runtime is not None:
            runtime = self._active_runtime
            self._record_abandoned_window(
                runtime,
                "training ended before checkpoint target step was reached",
            )
            self._stop_reconstruction_worker(runtime)
            self._pending_gradient_refs.clear()
            self._pending_gradient_transfers.clear()
            self._active_runtime = None

        while True:
            with self._pending_lock:
                pending_workers = list(self._pending_workers)
            if not pending_workers:
                return

            for pending in pending_workers:
                if pending.thread is not None:
                    pending.thread.join()
            for pending in pending_workers:
                self._finalize_pending_worker(pending)

    def _join_previous_persist_if_needed(self) -> None:
        self._finalize_completed_workers()

    def _finalize_completed_workers(self) -> None:
        with self._pending_lock:
            pending_workers = list(self._pending_workers)

        for pending in pending_workers:
            thread = pending.thread
            if thread is not None and thread.is_alive():
                continue
            if thread is not None:
                thread.join()
            self._finalize_pending_worker(pending)

    def _finalize_pending_worker(self, pending: PendingCheckpointWorker) -> None:
        with self._pending_lock:
            if pending not in self._pending_workers:
                return
            self._pending_workers.remove(pending)

        self.last_result = pending.result
        self.history.append(pending.result)

        if pending.error is not None:
            raise RuntimeError(
                "Background checkpoint persistence failed."
            ) from pending.error

    def transfer_timing_summary(self) -> dict[str, float | None]:
        total_transfer_enqueue = 0.0
        total_transfer_sync = 0.0
        total_transfer_full = 0.0
        total_transfer_count = 0
        total_gradient_submit = 0.0
        total_gradient_sync = 0.0
        total_gradient_full = 0.0
        total_gradient_count = 0

        for result in self.history:
            total_transfer_enqueue += result.transfer_duration_sec
            total_transfer_sync += result.transfer_sync_duration_sec
            total_transfer_full += (
                result.transfer_duration_sec + result.transfer_sync_duration_sec
            )
            total_transfer_count += result.transfer_count
            total_gradient_submit += result.gradient_submit_duration_sec
            total_gradient_sync += result.gradient_sync_duration_sec
            total_gradient_full += result.gradient_duration_sec
            total_gradient_count += result.gradient_count

        return {
            "mo_foreground_avg_sec": (
                (total_transfer_enqueue + total_transfer_sync) / total_transfer_count
                if total_transfer_count > 0
                else None
            ),
            "mo_full_avg_sec": (
                total_transfer_full / total_transfer_count
                if total_transfer_count > 0
                else None
            ),
            "mo_foreground_total_sec": total_transfer_enqueue + total_transfer_sync,
            "mo_full_total_sec": total_transfer_full,
            "mo_count": total_transfer_count,
            "gradient_foreground_avg_sec": (
                (total_gradient_submit + total_gradient_sync) / total_gradient_count
                if total_gradient_count > 0
                else None
            ),
            "gradient_full_avg_sec": (
                total_gradient_full / total_gradient_count
                if total_gradient_count > 0
                else None
            ),
            "gradient_foreground_total_sec": (
                total_gradient_submit + total_gradient_sync
            ),
            "gradient_full_total_sec": total_gradient_full,
            "gradient_count": total_gradient_count,
        }

    def _start_reconstruction_worker(self, runtime: GoCkptRuntime) -> None:
        runtime.reconstruction_executor = ThreadPoolExecutor(
            max_workers=self._reconstruction_workers,
            thread_name_prefix=f"gockpt-reconstruct-{runtime.request.target_step}",
        )
        # Bounded in-flight tasks form a simple ring buffer. If CPU replay falls
        # too far behind, enqueue blocks instead of growing memory unboundedly.
        capacity = max(1, self._reconstruction_queue_depth)
        runtime.reconstruction_slots = threading.Semaphore(capacity)
        self._add_ringbuffer_capacity(capacity)

    def _enqueue_reconstruction(
        self,
        runtime: GoCkptRuntime,
        step: int,
        gradients_for_step: dict[str, torch.Tensor | None],
        optimizer_param_groups: dict[str, dict[str, Any]],
    ) -> None:
        if runtime.reconstruction_error is not None:
            raise RuntimeError(
                "Background checkpoint reconstruction failed."
            ) from runtime.reconstruction_error

        executor = runtime.reconstruction_executor
        slots = runtime.reconstruction_slots
        if executor is None or slots is None:
            self._apply_reconstruction_task(
                runtime,
                step,
                gradients_for_step,
                optimizer_param_groups,
            )
            return

        gradients_by_partition: dict[int, dict[str, torch.Tensor | None]] = {}
        for name, grad in gradients_for_step.items():
            partition_index = runtime.partition_name_to_index[name]
            gradients_by_partition.setdefault(partition_index, {})[name] = grad

        if gradients_by_partition and runtime.reconstruction_started_at is None:
            runtime.reconstruction_started_at = time.perf_counter()

        for partition_index, partition_gradients in gradients_by_partition.items():
            acquire_start = time.perf_counter()
            slots.acquire()
            self._increment_ringbuffer_inflight()
            if runtime.result is not None:
                runtime.result.reconstruction_backpressure_sec += (
                    time.perf_counter() - acquire_start
                )
            previous_future = runtime.partition_futures.get(partition_index)
            future = executor.submit(
                self._apply_reconstruction_partition_task,
                runtime,
                step,
                partition_index,
                partition_gradients,
                optimizer_param_groups,
                previous_future,
            )
            runtime.partition_futures[partition_index] = future
            runtime.reconstruction_futures.append(future)

    def _wait_for_reconstruction(self, runtime: GoCkptRuntime) -> None:
        futures = list(runtime.reconstruction_futures)
        if futures:
            wait(futures)

        if runtime.reconstruction_error is not None:
            raise RuntimeError(
                "Background checkpoint reconstruction failed."
            ) from runtime.reconstruction_error

        if (
            runtime.reconstruction_started_at is not None
            and runtime.reconstruction_finished_at is None
        ):
            runtime.reconstruction_finished_at = time.perf_counter()
            if runtime.result is not None:
                runtime.result.reconstruction_duration_sec = (
                    runtime.reconstruction_finished_at
                    - runtime.reconstruction_started_at
                )

    def _stop_reconstruction_worker(self, runtime: GoCkptRuntime) -> None:
        executor = runtime.reconstruction_executor
        if executor is None:
            return

        self._wait_for_reconstruction(runtime)
        executor.shutdown(wait=True)
        self._remove_ringbuffer_capacity(max(1, self._reconstruction_queue_depth))
        runtime.reconstruction_executor = None
        runtime.reconstruction_slots = None

    def _apply_reconstruction_task(
        self,
        runtime: GoCkptRuntime,
        step: int,
        gradients_for_step: dict[str, torch.Tensor | None],
        optimizer_param_groups: dict[str, dict[str, Any]],
    ) -> None:
        reconstruction_start = time.perf_counter()
        waited_partitions: set[int] = set()
        for name in gradients_for_step:
            partition_index = runtime.partition_name_to_index[name]
            if partition_index in waited_partitions:
                continue
            self._wait_partition_transfer(runtime, partition_index)
            self._ensure_partition_flattened(runtime, partition_index)
            waited_partitions.add(partition_index)

        self._apply_cpu_adamw_updates(
            runtime,
            step,
            gradients_for_step,
            optimizer_param_groups,
        )

        if runtime.result is not None:
            runtime.result.reconstruction_duration_sec += (
                time.perf_counter() - reconstruction_start
            )

    def _apply_reconstruction_partition_task(
        self,
        runtime: GoCkptRuntime,
        step: int,
        partition_index: int,
        gradients_for_step: dict[str, torch.Tensor | None],
        optimizer_param_groups: dict[str, dict[str, Any]],
        previous_future: Future | None,
    ) -> None:
        slots = runtime.reconstruction_slots
        try:
            if previous_future is not None:
                previous_future.result()

            if runtime.reconstruction_error is not None:
                return

            reconstruction_start = time.perf_counter()
            self._wait_partition_transfer(runtime, partition_index)
            self._ensure_partition_flattened(runtime, partition_index)

            self._apply_cpu_adamw_updates(
                runtime,
                step,
                gradients_for_step,
                optimizer_param_groups,
            )

            # Parallel reconstruction duration is reported as wall time from the
            # first submitted task until all partition futures complete.
        except BaseException as exc:
            runtime.reconstruction_error = exc
            raise
        finally:
            if slots is not None:
                self._decrement_ringbuffer_inflight()
                slots.release()

    def _build_optimizer_metadata(self) -> None:
        if self.optimizer is None:
            return

        state_id = 0
        for group in self.optimizer.param_groups:
            group_template = {
                key: value for key, value in group.items() if key != "params"
            }
            group_param_ids: list[int] = []
            for param in group["params"]:
                param_id = id(param)
                self._param_to_group[param_id] = group_template.copy()
                self._param_to_state_id[param_id] = state_id
                group_param_ids.append(state_id)
                state_id += 1

            group_template["params"] = group_param_ids

    def _build_partitions(self) -> list[list[str]]:
        param_names = [name for name, _ in self._named_parameters]
        if not param_names:
            return [[] for _ in range(self.overlap_steps)]

        sizes = [self._estimate_parameter_bytes(name) for name in param_names]
        total_size = sum(sizes)
        target_partition_size = max(total_size / self.overlap_steps, 1.0)

        partitions: list[list[str]] = [[]]
        current_partition_size = 0.0
        for name, size in zip(param_names, sizes):
            remaining_names = len(param_names) - sum(len(part) for part in partitions)
            remaining_partitions = self.overlap_steps - len(partitions)

            if (
                current_partition_size >= target_partition_size
                and remaining_partitions > 0
                and remaining_names > remaining_partitions
            ):
                partitions.append([])
                current_partition_size = 0.0

            partitions[-1].append(name)
            current_partition_size += size

        while len(partitions) < self.overlap_steps:
            partitions.append([])

        return partitions

    def _estimate_parameter_bytes(self, name: str) -> int:
        param = self._params_by_name[name]
        size = param.numel() * param.element_size()
        opt_state = (
            self.optimizer.state.get(param, {}) if self.optimizer is not None else {}
        )
        exp_avg = opt_state.get("exp_avg")
        exp_avg_sq = opt_state.get("exp_avg_sq")
        if isinstance(exp_avg, torch.Tensor):
            size += exp_avg.numel() * exp_avg.element_size()
        else:
            size += param.numel() * 4
        if isinstance(exp_avg_sq, torch.Tensor):
            size += exp_avg_sq.numel() * exp_avg_sq.element_size()
        else:
            size += param.numel() * 4
        return size

    def _is_step_in_request(self, runtime: GoCkptRuntime, step: int) -> bool:
        return runtime.request.start_step <= step < runtime.request.target_step

    def _spawn_partition_transfer_thread(
        self,
        runtime: GoCkptRuntime,
        partition_index: int,
        step: int,
    ) -> None:
        """Spawn a background thread to do partition transfer.

        This allows forward_begin to return immediately, overlapping the
        GPU->CPU transfer with the forward+backward computation.
        """
        # Mark partition as being transferred (prevents duplicate transfers)
        runtime.transferred_partitions.add(partition_index)

        def transfer_worker():
            try:
                self._schedule_partition_transfer(runtime, partition_index, step)
            except BaseException as exc:
                runtime.partition_transfer_errors[partition_index] = exc

        thread = threading.Thread(
            target=transfer_worker,
            daemon=True,
            name=f"gockpt-transfer-{runtime.request.target_step}-p{partition_index}",
        )
        runtime.partition_transfer_threads[partition_index] = thread
        thread.start()

    def _wait_partition_transfer_thread(
        self,
        runtime: GoCkptRuntime,
        partition_index: int,
    ) -> None:
        """Wait for background transfer thread to complete."""
        thread = runtime.partition_transfer_threads.get(partition_index)
        if thread is not None:
            thread.join()
            runtime.partition_transfer_threads.pop(partition_index, None)

        error = runtime.partition_transfer_errors.get(partition_index)
        if error is not None:
            raise RuntimeError(
                f"Background partition transfer failed for partition {partition_index}"
            ) from error

    def _schedule_partition_transfer(
        self,
        runtime: GoCkptRuntime,
        partition_index: int,
        step: int,
    ) -> None:
        partition = runtime.partitions[partition_index]
        transfer_start = time.perf_counter()

        current_stream = (
            torch.cuda.current_stream() if self._transfer_stream is not None else None
        )
        if self._transfer_stream is not None and current_stream is not None:
            self._transfer_stream.wait_stream(current_stream)

        with (
            torch.cuda.stream(self._transfer_stream)
            if self._transfer_stream is not None
            else _nullcontext()
        ):
            for name in partition:
                param = self._params_by_name[name]
                optimizer_state = self._snapshot_optimizer_state(param)
                snapshot = ParameterSnapshot(
                    name=name,
                    param=self._copy_tensor_to_cpu(param.detach()),
                    grad=None,
                    optimizer_state=optimizer_state,
                    version_step=step,
                )
                # Thread-safe write to transferred_blocks
                with runtime.transfer_blocks_lock:
                    runtime.transferred_blocks[name] = snapshot

        event: torch.cuda.Event | None = None
        if self._transfer_stream is not None:
            event = torch.cuda.Event()
            event.record(self._transfer_stream)

        runtime.partition_events[partition_index] = event
        # Note: transferred_partitions is already set by _spawn_partition_transfer_thread
        if runtime.result is not None:
            runtime.result.transfer_duration_sec += time.perf_counter() - transfer_start
            runtime.result.transfer_count += 1

    def _snapshot_optimizer_state(
        self, param: torch.nn.Parameter
    ) -> OptimizerParamSnapshot:
        if self.optimizer is None:
            raise RuntimeError("GoCkpt requires an optimizer to snapshot state.")

        raw_state = self.optimizer.state.get(param, {})
        exp_avg = raw_state.get("exp_avg")
        exp_avg_sq = raw_state.get("exp_avg_sq")
        step = raw_state.get("step", 0)

        if not isinstance(exp_avg, torch.Tensor):
            exp_avg = torch.zeros_like(param.detach())
        if not isinstance(exp_avg_sq, torch.Tensor):
            exp_avg_sq = torch.zeros_like(param.detach())

        if isinstance(step, torch.Tensor):
            step_snapshot: int | torch.Tensor = self._copy_tensor_to_cpu(step.detach())
        else:
            step_snapshot = int(step)

        return OptimizerParamSnapshot(
            exp_avg=self._copy_tensor_to_cpu(exp_avg.detach()),
            exp_avg_sq=self._copy_tensor_to_cpu(exp_avg_sq.detach()),
            step=step_snapshot,
            param_group=self._snapshot_optimizer_group_for_param(param),
        )

    def _snapshot_optimizer_param_groups_by_name(self) -> dict[str, dict[str, Any]]:
        if self.optimizer is None:
            raise RuntimeError("GoCkpt requires an optimizer to snapshot param groups.")

        param_groups: dict[str, dict[str, Any]] = {}
        for group in self.optimizer.param_groups:
            group_snapshot = {
                key: self._clone_optimizer_value(value)
                for key, value in group.items()
                if key != "params"
            }
            for param in group["params"]:
                name = self._param_name_by_obj.get(id(param))
                if name is not None:
                    param_groups[name] = group_snapshot.copy()
        return param_groups

    def _snapshot_optimizer_group_for_param(
        self,
        param: torch.nn.Parameter,
    ) -> dict[str, Any]:
        if self.optimizer is None:
            raise RuntimeError("GoCkpt requires an optimizer to snapshot param groups.")

        for group in self.optimizer.param_groups:
            if any(candidate is param for candidate in group["params"]):
                return {
                    key: self._clone_optimizer_value(value)
                    for key, value in group.items()
                    if key != "params"
                }
        return self._param_to_group[id(param)].copy()

    def _snapshot_optimizer_param_groups_for_checkpoint(self) -> list[dict[str, Any]]:
        if self.optimizer is None:
            return []

        param_groups: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_snapshot = {
                key: self._clone_optimizer_value(value)
                for key, value in group.items()
                if key != "params"
            }
            group_snapshot["params"] = [
                self._param_to_state_id[id(param)]
                for param in group["params"]
                if id(param) in self._param_to_state_id
            ]
            param_groups.append(group_snapshot)
        return param_groups

    def _clone_optimizer_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, list):
            return [self._clone_optimizer_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone_optimizer_value(item) for item in value)
        if isinstance(value, dict):
            return {
                key: self._clone_optimizer_value(item) for key, item in value.items()
            }
        return value

    def _ensure_partition_flattened(
        self,
        runtime: GoCkptRuntime,
        partition_index: int,
    ) -> None:
        if partition_index in runtime.flattened_partitions:
            return

        with runtime.flatten_lock:
            if partition_index in runtime.flattened_partitions:
                return
            self._flatten_partition_snapshots(
                runtime,
                partition_index,
                runtime.partitions[partition_index],
            )
            runtime.flattened_partitions.add(partition_index)

    def _flatten_partition_snapshots(
        self,
        runtime: GoCkptRuntime,
        partition_index: int,
        partition: list[str],
    ) -> None:
        groups: dict[tuple[torch.dtype, torch.dtype, torch.dtype], list[str]] = {}
        for name in partition:
            snapshot = runtime.transferred_blocks[name]
            key = (
                snapshot.param.dtype,
                snapshot.optimizer_state.exp_avg.dtype,
                snapshot.optimizer_state.exp_avg_sq.dtype,
            )
            groups.setdefault(key, []).append(name)

        for names in groups.values():
            total_numel = sum(
                runtime.transferred_blocks[name].param.numel() for name in names
            )
            if total_numel == 0:
                continue

            first = runtime.transferred_blocks[names[0]]
            param_buffer = torch.empty(
                total_numel, dtype=first.param.dtype, device="cpu"
            )
            exp_avg_buffer = torch.empty(
                total_numel,
                dtype=first.optimizer_state.exp_avg.dtype,
                device="cpu",
            )
            exp_avg_sq_buffer = torch.empty(
                total_numel,
                dtype=first.optimizer_state.exp_avg_sq.dtype,
                device="cpu",
            )

            offsets: dict[str, tuple[int, int]] = {}
            offset = 0
            for name in names:
                snapshot = runtime.transferred_blocks[name]
                numel = snapshot.param.numel()
                end = offset + numel
                param_buffer[offset:end].copy_(snapshot.param.reshape(-1))
                exp_avg_buffer[offset:end].copy_(
                    snapshot.optimizer_state.exp_avg.reshape(-1)
                )
                exp_avg_sq_buffer[offset:end].copy_(
                    snapshot.optimizer_state.exp_avg_sq.reshape(-1)
                )

                snapshot.param = param_buffer[offset:end].view_as(snapshot.param)
                snapshot.optimizer_state.exp_avg = exp_avg_buffer[offset:end].view_as(
                    snapshot.optimizer_state.exp_avg
                )
                snapshot.optimizer_state.exp_avg_sq = exp_avg_sq_buffer[
                    offset:end
                ].view_as(snapshot.optimizer_state.exp_avg_sq)
                offsets[name] = (offset, end)
                offset = end

            block = FlatReplayBlock(
                partition_index=partition_index,
                names=names,
                offsets=offsets,
                param_buffer=param_buffer,
                exp_avg_buffer=exp_avg_buffer,
                exp_avg_sq_buffer=exp_avg_sq_buffer,
            )
            for name in names:
                runtime.flat_replay_blocks_by_name[name] = block

    def _copy_tensor_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cuda" or self._transfer_stream is None:
            return tensor.cpu().clone()

        return self._copy_cuda_tensor_to_pinned_cpu(tensor)

    def _copy_cuda_tensor_to_pinned_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
        if self._transfer_chunk_bytes <= 0:
            cpu_tensor.copy_(tensor, non_blocking=True)
            return cpu_tensor

        total_bytes = tensor.numel() * tensor.element_size()
        if total_bytes <= self._transfer_chunk_bytes:
            cpu_tensor.copy_(tensor, non_blocking=True)
            return cpu_tensor

        source = tensor if tensor.is_contiguous() else tensor.contiguous()
        source_flat = source.reshape(-1)
        target_flat = cpu_tensor.reshape(-1)
        chunk_elems = max(1, self._transfer_chunk_bytes // tensor.element_size())
        for start in range(0, source_flat.numel(), chunk_elems):
            end = min(start + chunk_elems, source_flat.numel())
            target_flat[start:end].copy_(source_flat[start:end], non_blocking=True)
        return cpu_tensor

    def _submit_async_gradient_transfer(
        self,
        step: int,
        gradients: dict[str, torch.Tensor | None],
    ) -> GoCkptPendingGradientTransfer:
        cpu_gradients: dict[str, torch.Tensor | None] = {}
        source_refs: dict[str, torch.Tensor] = {}
        submitted_at = time.perf_counter()

        if self._gradient_stream is None:
            for name, grad in gradients.items():
                if grad is None:
                    cpu_gradients[name] = None
                    continue
                source_refs[name] = grad
                cpu_gradients[name] = grad.cpu().clone()
            submit_duration = time.perf_counter() - submitted_at
            return GoCkptPendingGradientTransfer(
                step=step,
                gradients=cpu_gradients,
                source_refs=source_refs,
                event=None,
                submitted_at=submitted_at,
                submit_duration_sec=submit_duration,
            )

        current_stream = torch.cuda.current_stream()
        self._gradient_stream.wait_stream(current_stream)
        with torch.cuda.stream(self._gradient_stream):
            for name, grad in gradients.items():
                if grad is None:
                    cpu_gradients[name] = None
                    continue
                source_refs[name] = grad
                cpu_gradients[name] = self._copy_cuda_tensor_to_pinned_cpu(grad)

        event = torch.cuda.Event()
        event.record(self._gradient_stream)
        submit_duration = time.perf_counter() - submitted_at
        return GoCkptPendingGradientTransfer(
            step=step,
            gradients=cpu_gradients,
            source_refs=source_refs,
            event=event,
            submitted_at=submitted_at,
            submit_duration_sec=submit_duration,
        )

    def _finish_pending_gradient_transfer(
        self,
        runtime: GoCkptRuntime,
        step: int,
    ) -> None:
        pending = self._pending_gradient_transfers.pop(step, None)
        if pending is None:
            return

        sync_duration = 0.0
        if pending.event is not None:
            sync_start = time.perf_counter()
            pending.event.synchronize()
            sync_duration = time.perf_counter() - sync_start

        optimizer_param_groups = runtime.optimizer_param_groups_by_step.get(step)
        if optimizer_param_groups is None:
            optimizer_param_groups = self._snapshot_optimizer_param_groups_by_name()
            runtime.optimizer_param_groups_by_step[step] = optimizer_param_groups

        self._enqueue_reconstruction(
            runtime,
            step,
            pending.gradients,
            optimizer_param_groups,
        )

        if runtime.result is not None:
            runtime.result.gradient_duration_sec += (
                time.perf_counter() - pending.submitted_at
            )
            runtime.result.gradient_submit_duration_sec += pending.submit_duration_sec
            runtime.result.gradient_sync_duration_sec += sync_duration
            runtime.result.gradient_count += 1

    def _wait_partition_transfer(
        self, runtime: GoCkptRuntime, partition_index: int
    ) -> None:
        event = runtime.partition_events.get(partition_index)
        if event is None:
            return

        sync_start = time.perf_counter()
        event.synchronize()
        if runtime.result is not None:
            runtime.result.transfer_sync_duration_sec += (
                time.perf_counter() - sync_start
            )
        runtime.partition_events[partition_index] = None

    def _wait_current_partition_transfer_before_update(
        self,
        runtime: GoCkptRuntime,
        step: int,
    ) -> None:
        partition_index = step - runtime.request.start_step
        if partition_index < 0 or partition_index >= len(runtime.partitions):
            return
        if partition_index not in runtime.transferred_partitions:
            return

        # Wait for background transfer thread to complete first
        self._wait_partition_transfer_thread(runtime, partition_index)

        # The partition snapshot reads model parameters and optimizer states.
        # It must complete before optimizer.step() mutates those tensors.
        self._wait_partition_transfer(runtime, partition_index)

    def _apply_cpu_adamw_update(
        self,
        snapshot: ParameterSnapshot,
        grad: torch.Tensor,
        param_group: dict[str, Any],
    ) -> None:
        param = snapshot.param
        exp_avg = snapshot.optimizer_state.exp_avg
        exp_avg_sq = snapshot.optimizer_state.exp_avg_sq
        step_state = snapshot.optimizer_state.step
        group = param_group

        lr = float(group.get("lr", 1e-3))
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))
        weight_decay = float(group.get("weight_decay", 0.0))
        maximize = bool(group.get("maximize", False))

        grad = grad.to(param.dtype)
        if maximize:
            grad = -grad

        if isinstance(step_state, torch.Tensor):
            current_step = int(step_state.item())
        else:
            current_step = int(step_state)
        next_step = current_step + 1

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        bias_correction1 = 1 - beta1**next_step
        bias_correction2 = 1 - beta2**next_step
        denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
        step_size = lr / bias_correction1
        param.addcdiv_(exp_avg, denom, value=-step_size)

        if isinstance(step_state, torch.Tensor):
            step_state.fill_(next_step)
        else:
            snapshot.optimizer_state.step = next_step

    def _apply_cpu_adamw_updates(
        self,
        runtime: GoCkptRuntime,
        step: int,
        gradients_for_step: dict[str, torch.Tensor | None],
        optimizer_param_groups: dict[str, dict[str, Any]],
    ) -> None:
        if self._apply_flat_cpu_adamw_updates(
            runtime,
            step,
            gradients_for_step,
            optimizer_param_groups,
        ):
            return

        buckets: dict[
            tuple[Any, ...],
            tuple[dict[str, Any], list[ParameterSnapshot], list[torch.Tensor]],
        ] = {}

        for name, grad in gradients_for_step.items():
            snapshot = runtime.transferred_blocks[name]
            if grad is None:
                # Use max to handle out-of-order reconstruction task execution
                snapshot.version_step = max(snapshot.version_step, step + 1)
                continue

            param_group = optimizer_param_groups[name]
            signature = self._adamw_bucket_signature(snapshot, param_group)
            bucket = buckets.get(signature)
            if bucket is None:
                bucket = (param_group, [], [])
                buckets[signature] = bucket

            bucket[1].append(snapshot)
            bucket[2].append(grad)

        for param_group, snapshots, gradients in buckets.values():
            self._apply_cpu_adamw_bucket(snapshots, gradients, param_group)

        for name in gradients_for_step:
            # Use max to handle out-of-order reconstruction task execution
            snapshot = runtime.transferred_blocks[name]
            snapshot.version_step = max(snapshot.version_step, step + 1)

    def _apply_flat_cpu_adamw_updates(
        self,
        runtime: GoCkptRuntime,
        step: int,
        gradients_for_step: dict[str, torch.Tensor | None],
        optimizer_param_groups: dict[str, dict[str, Any]],
    ) -> bool:
        if not gradients_for_step:
            return True

        blocks: dict[int, FlatReplayBlock] = {}
        for name, grad in gradients_for_step.items():
            if grad is None:
                return False
            block = runtime.flat_replay_blocks_by_name.get(name)
            if block is None:
                return False
            blocks[id(block)] = block

        for block in blocks.values():
            block_names = [name for name in block.names if name in gradients_for_step]
            if len(block_names) != len(block.names):
                return False

            first_name = block_names[0]
            first_snapshot = runtime.transferred_blocks[first_name]
            first_group = optimizer_param_groups[first_name]
            signature = self._adamw_bucket_signature(first_snapshot, first_group)
            first_step = int(self._ensure_step_tensor(first_snapshot).item())
            for name in block_names[1:]:
                snapshot = runtime.transferred_blocks[name]
                if (
                    self._adamw_bucket_signature(snapshot, optimizer_param_groups[name])
                    != signature
                ):
                    return False
                if int(self._ensure_step_tensor(snapshot).item()) != first_step:
                    return False

        for block in blocks.values():
            self._apply_flat_cpu_adamw_block(
                runtime,
                block,
                gradients_for_step,
                optimizer_param_groups[block.names[0]],
            )

        for name in gradients_for_step:
            # Use max to handle out-of-order reconstruction task execution
            snapshot = runtime.transferred_blocks[name]
            snapshot.version_step = max(snapshot.version_step, step + 1)
        return True

    def _apply_flat_cpu_adamw_block(
        self,
        runtime: GoCkptRuntime,
        block: FlatReplayBlock,
        gradients_for_step: dict[str, torch.Tensor | None],
        param_group: dict[str, Any],
    ) -> None:
        grad_buffer = torch.empty_like(block.param_buffer)
        for name in block.names:
            grad = gradients_for_step[name]
            if grad is None:
                raise RuntimeError("Flat GoCkpt replay received a missing gradient.")
            start, end = block.offsets[name]
            grad_buffer[start:end].copy_(
                grad.to(
                    device=block.param_buffer.device, dtype=block.param_buffer.dtype
                ).reshape(-1)
            )

        first_snapshot = runtime.transferred_blocks[block.names[0]]
        step_tensor = self._ensure_step_tensor(first_snapshot)
        if self._apply_rust_cpu_adam_flat_block(
            block,
            grad_buffer,
            step_tensor,
            param_group,
        ):
            pass
        elif self._apply_deepspeed_cpu_adam_flat_block(
            block,
            grad_buffer,
            step_tensor,
            param_group,
        ):
            pass
        else:
            beta1, beta2 = param_group.get("betas", (0.9, 0.999))
            optim_functional.adamw(
                params=[block.param_buffer],
                grads=[grad_buffer],
                exp_avgs=[block.exp_avg_buffer],
                exp_avg_sqs=[block.exp_avg_sq_buffer],
                max_exp_avg_sqs=[],
                state_steps=[step_tensor],
                foreach=True,
                capturable=False,
                differentiable=False,
                fused=False,
                grad_scale=None,
                found_inf=None,
                has_complex=torch.is_complex(block.param_buffer),
                amsgrad=False,
                beta1=float(beta1),
                beta2=float(beta2),
                lr=float(param_group.get("lr", 1e-3)),
                weight_decay=float(param_group.get("weight_decay", 0.0)),
                eps=float(param_group.get("eps", 1e-8)),
                maximize=bool(param_group.get("maximize", False)),
            )

        for name in block.names:
            runtime.transferred_blocks[name].optimizer_state.step = step_tensor.clone()

    def _apply_rust_cpu_adam_flat_block(
        self,
        block: FlatReplayBlock,
        grad_buffer: torch.Tensor,
        step_tensor: torch.Tensor,
        param_group: dict[str, Any],
    ) -> bool:
        if os.environ.get("GOCKPT_CPU_REPLAY_BACKEND", "rust") != "rust":
            return False
        if bool(param_group.get("amsgrad", False)):
            return False
        if torch.is_complex(block.param_buffer):
            return False

        beta1, beta2 = param_group.get("betas", (0.9, 0.999))
        next_step = rust_adamw_update(
            block.param_buffer,
            grad_buffer,
            block.exp_avg_buffer,
            block.exp_avg_sq_buffer,
            int(step_tensor.item()),
            float(param_group.get("lr", 1e-3)),
            float(beta1),
            float(beta2),
            float(param_group.get("eps", 1e-8)),
            float(param_group.get("weight_decay", 0.0)),
            bool(param_group.get("maximize", False)),
        )
        if next_step is None:
            return False

        step_tensor.fill_(int(next_step))
        return True

    def _apply_deepspeed_cpu_adam_flat_block(
        self,
        block: FlatReplayBlock,
        grad_buffer: torch.Tensor,
        step_tensor: torch.Tensor,
        param_group: dict[str, Any],
    ) -> bool:
        if os.environ.get("GOCKPT_CPU_REPLAY_BACKEND") != "deepspeed":
            return False
        if DeepSpeedCPUAdam is None:
            return False
        if bool(param_group.get("amsgrad", False)):
            return False
        if torch.is_complex(block.param_buffer):
            return False

        opt = self._get_deepspeed_cpu_adam()
        beta1, beta2 = param_group.get("betas", (0.9, 0.999))
        next_step = int(step_tensor.item()) + 1
        grad = grad_buffer
        if bool(param_group.get("maximize", False)):
            grad = -grad

        opt.ds_opt_adam.adam_update(
            opt.opt_id,
            next_step,
            float(param_group.get("lr", 1e-3)),
            float(beta1),
            float(beta2),
            float(param_group.get("eps", 1e-8)),
            float(param_group.get("weight_decay", 0.0)),
            bool(param_group.get("bias_correction", True)),
            block.param_buffer,
            grad,
            block.exp_avg_buffer,
            block.exp_avg_sq_buffer,
        )
        step_tensor.fill_(next_step)
        return True

    def _adamw_bucket_signature(
        self,
        snapshot: ParameterSnapshot,
        param_group: dict[str, Any],
    ) -> tuple[Any, ...]:
        betas = param_group.get("betas", (0.9, 0.999))
        return (
            snapshot.param.dtype,
            float(param_group.get("lr", 1e-3)),
            float(betas[0]),
            float(betas[1]),
            float(param_group.get("eps", 1e-8)),
            float(param_group.get("weight_decay", 0.0)),
            bool(param_group.get("maximize", False)),
            bool(param_group.get("amsgrad", False)),
        )

    def _apply_cpu_adamw_bucket(
        self,
        snapshots: list[ParameterSnapshot],
        gradients: list[torch.Tensor],
        param_group: dict[str, Any],
    ) -> None:
        if not snapshots:
            return

        amsgrad = bool(param_group.get("amsgrad", False))
        if amsgrad:
            raise NotImplementedError(
                "GoCkpt CPU replay does not support AdamW amsgrad."
            )

        if self._apply_deepspeed_cpu_adam_bucket(snapshots, gradients, param_group):
            return

        params: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        exp_avgs: list[torch.Tensor] = []
        exp_avg_sqs: list[torch.Tensor] = []
        state_steps: list[torch.Tensor] = []

        for snapshot, grad in zip(snapshots, gradients):
            param = snapshot.param
            params.append(param)
            grads.append(grad.to(device=param.device, dtype=param.dtype))
            exp_avgs.append(snapshot.optimizer_state.exp_avg)
            exp_avg_sqs.append(snapshot.optimizer_state.exp_avg_sq)
            state_steps.append(self._ensure_step_tensor(snapshot))

        beta1, beta2 = param_group.get("betas", (0.9, 0.999))
        optim_functional.adamw(
            params=params,
            grads=grads,
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            max_exp_avg_sqs=[],
            state_steps=state_steps,
            foreach=True,
            capturable=False,
            differentiable=False,
            fused=False,
            grad_scale=None,
            found_inf=None,
            has_complex=any(torch.is_complex(param) for param in params),
            amsgrad=False,
            beta1=float(beta1),
            beta2=float(beta2),
            lr=float(param_group.get("lr", 1e-3)),
            weight_decay=float(param_group.get("weight_decay", 0.0)),
            eps=float(param_group.get("eps", 1e-8)),
            maximize=bool(param_group.get("maximize", False)),
        )

    def _apply_deepspeed_cpu_adam_bucket(
        self,
        snapshots: list[ParameterSnapshot],
        gradients: list[torch.Tensor],
        param_group: dict[str, Any],
    ) -> bool:
        if os.environ.get("GOCKPT_CPU_REPLAY_BACKEND") != "deepspeed":
            return False
        if DeepSpeedCPUAdam is None:
            return False
        if bool(param_group.get("amsgrad", False)):
            return False
        if any(torch.is_complex(snapshot.param) for snapshot in snapshots):
            return False

        opt = self._get_deepspeed_cpu_adam()
        beta1, beta2 = param_group.get("betas", (0.9, 0.999))
        lr = float(param_group.get("lr", 1e-3))
        eps = float(param_group.get("eps", 1e-8))
        weight_decay = float(param_group.get("weight_decay", 0.0))
        bias_correction = bool(param_group.get("bias_correction", True))
        maximize = bool(param_group.get("maximize", False))

        for snapshot, grad in zip(snapshots, gradients):
            param = snapshot.param
            step_tensor = self._ensure_step_tensor(snapshot)
            next_step = int(step_tensor.item()) + 1
            grad_tensor = grad.to(device=param.device, dtype=param.dtype)
            if maximize:
                grad_tensor = -grad_tensor

            opt.ds_opt_adam.adam_update(
                opt.opt_id,
                next_step,
                lr,
                float(beta1),
                float(beta2),
                eps,
                weight_decay,
                bias_correction,
                param,
                grad_tensor,
                snapshot.optimizer_state.exp_avg,
                snapshot.optimizer_state.exp_avg_sq,
            )
            step_tensor.fill_(next_step)
        return True

    def _get_deepspeed_cpu_adam(self):
        opt = getattr(self._ds_cpu_adam_local, "optimizer", None)
        if opt is None:
            dummy = torch.nn.Parameter(torch.empty(0, dtype=torch.float32))
            opt = DeepSpeedCPUAdam([dummy], adamw_mode=True, fp32_optimizer_states=True)
            self._ds_cpu_adam_local.optimizer = opt
        return opt

    def _ensure_step_tensor(self, snapshot: ParameterSnapshot) -> torch.Tensor:
        step_state = snapshot.optimizer_state.step
        if isinstance(step_state, torch.Tensor):
            return step_state

        step_tensor = torch.tensor(float(step_state), dtype=torch.float32)
        snapshot.optimizer_state.step = step_tensor
        return step_tensor

    def _assemble_checkpoint(self, runtime: GoCkptRuntime) -> dict[str, Any]:
        model_state: dict[str, Any] = {}
        for name, value in self.model.state_dict().items():
            if name in runtime.transferred_blocks:
                model_state[name] = runtime.transferred_blocks[name].param
            else:
                model_state[name] = value.detach().cpu().clone()

        optimizer_state: dict[int, dict[str, Any]] = {}
        for name, param in self._named_parameters:
            snapshot = runtime.transferred_blocks.get(name)
            if snapshot is None:
                continue

            state_id = self._param_to_state_id[id(param)]
            entry: dict[str, Any] = {
                "exp_avg": snapshot.optimizer_state.exp_avg,
                "exp_avg_sq": snapshot.optimizer_state.exp_avg_sq,
                "step": snapshot.optimizer_state.step,
            }
            optimizer_state[state_id] = entry

        checkpoint: dict[str, Any] = {
            "step": runtime.request.target_step,
            "tag": runtime.request.tag,
            "time_unix": time.time(),
            "model": model_state,
            "optimizer": {
                "state": optimizer_state,
                "param_groups": self._snapshot_optimizer_param_groups_for_checkpoint(),
            },
        }
        if self.config.save_rng_state:
            checkpoint["rng_state"] = self._capture_rng_state()
        checkpoint["metadata"] = {
            "start_step": runtime.request.start_step,
            "target_step": runtime.request.target_step,
            "overlap_steps": self.overlap_steps,
            "reconstruction_queue_depth": self._reconstruction_queue_depth,
            "transfer_chunk_mb": self._transfer_chunk_mb,
            "num_partitions": len(runtime.partitions),
        }
        return checkpoint

    def _record_abandoned_window(self, runtime: GoCkptRuntime, reason: str) -> None:
        result = runtime.result
        tag = result.tag if result is not None else runtime.request.tag
        window = GoCkptAbandonedWindow(
            start_step=runtime.request.start_step,
            target_step=runtime.request.target_step,
            tag=tag,
            reason=reason,
            transferred_partitions=len(runtime.transferred_partitions),
            total_partitions=len(runtime.partitions),
            reconstructed_blocks=sum(
                1
                for snapshot in runtime.transferred_blocks.values()
                if snapshot.version_step == runtime.request.target_step
            ),
            total_blocks=len(self._named_parameters),
        )
        self.abandoned_windows.append(window)
        warnings.warn(
            "Discarding incomplete GoCkpt window "
            f"{tag}: {reason}; no checkpoint will be written.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _finalize_checkpoint_worker(
        self,
        runtime: GoCkptRuntime,
        pending: PendingCheckpointWorker,
    ) -> None:
        result = runtime.result
        if result is None:
            pending.error = RuntimeError(
                "GoCkpt runtime is missing checkpoint result metadata."
            )
            return

        try:
            self._stop_reconstruction_worker(runtime)

            for partition_index in range(len(runtime.partitions)):
                # Wait for background transfer thread first
                self._wait_partition_transfer_thread(runtime, partition_index)
                self._wait_partition_transfer(runtime, partition_index)
                self._ensure_partition_flattened(runtime, partition_index)

            expected_version = runtime.request.target_step
            for snapshot in runtime.transferred_blocks.values():
                if snapshot.version_step != expected_version:
                    raise RuntimeError(
                        "GoCkpt reconstruction is incomplete; not all partitions "
                        f"reached target step {expected_version}."
                    )

            checkpoint = self._assemble_checkpoint(runtime)
            pending.error = self._persist_checkpoint_worker(
                checkpoint, result.path, result
            )
        except BaseException as exc:
            pending.error = exc

    def _persist_checkpoint_worker(
        self,
        checkpoint: dict[str, Any],
        checkpoint_path: Path,
        result: GoCkptCheckpointResult,
    ) -> BaseException | None:
        persistence_start = time.perf_counter()
        error: BaseException | None = None

        try:
            with self._persist_lock:
                torch.save(checkpoint, checkpoint_path)
        except BaseException as exc:
            error = exc

        persistence_duration = time.perf_counter() - persistence_start
        result.persistence_duration_sec = persistence_duration
        result.total_duration_sec = (
            result.transfer_duration_sec
            + result.gradient_duration_sec
            + result.reconstruction_duration_sec
            + result.reconstruction_backpressure_sec
            + persistence_duration
        )

        return error


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False
