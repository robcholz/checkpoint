from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.gockpt_hook import (
    GoCkptCheckpointConfig,
    GoCkptCheckpointHook,
    GoCkptCheckpointResult,
    GoCkptRuntime,
)


@dataclass
class GoCkptOCheckpointConfig(GoCkptCheckpointConfig):
    reconstruction_workers: int = 4


@dataclass
class GoCkptOCheckpointResult(GoCkptCheckpointResult):
    pass


@dataclass
class PendingGradientTransfer:
    step: int
    gradients: dict[str, torch.Tensor | None]
    source_refs: dict[str, torch.Tensor]
    event: torch.cuda.Event | None
    submitted_at: float


class GoCkptOCheckpointHook(GoCkptCheckpointHook):
    """
    Optimized GoCkpt variant with hidden gradient transfer.

    Difference from GoCkpt:
    - model/optimizer partition transfer is unchanged
    - gradient GPU->CPU transfer is submitted asynchronously after backward
    - that transfer may overlap with the current update and the next step's
      forward pass
    - the transfer must complete before the next backward begins, because that
      is when gradient buffers may be overwritten
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        config: GoCkptCheckpointConfig | None = None,
        checkpoint_builder=None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            config=config,
            checkpoint_builder=checkpoint_builder,
        )
        self.last_result: GoCkptOCheckpointResult | None = None
        self.history: list[GoCkptOCheckpointResult] = []
        self._gradient_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self._pending_gradient_transfers: dict[int, PendingGradientTransfer] = {}
        self._reconstruction_workers = max(
            1,
            int(getattr(self.config, "reconstruction_workers", 4)),
        )
        self._reconstruction_pool = ThreadPoolExecutor(
            max_workers=self._reconstruction_workers,
            thread_name_prefix="gockpt-o-reconstruct",
        )
        self._reconstruction_futures_by_partition: dict[int, Future[None]] = {}
        self._reconstruction_lock = threading.Lock()

    def save_checkpoint(self, step: int) -> None:
        self._pending_gradient_transfers.clear()
        self._reconstruction_futures_by_partition.clear()
        super().save_checkpoint(step)
        runtime = self._active_runtime
        if runtime is None or runtime.result is None:
            return

        runtime.result = GoCkptOCheckpointResult(
            start_step=runtime.result.start_step,
            target_step=runtime.result.target_step,
            tag=runtime.result.tag,
            path=runtime.result.path,
        )

    def backward_begin(self, step: int) -> None:
        runtime = self._active_runtime
        if runtime is None or not self._is_step_in_request(runtime, step):
            return

        self._finish_pending_gradient_transfer(runtime, step - 1)

    def backward_end(self, step: int) -> None:
        runtime = self._active_runtime
        if runtime is None or not self._is_step_in_request(runtime, step):
            return

        gradients = self._collect_gradients_for_step(runtime, step)
        if not gradients:
            return

        transfer = self._submit_async_gradient_transfer(step, gradients)
        self._pending_gradient_transfers[step] = transfer

    def update_begin(self, step: int) -> None:
        return

    def update_end(self, step: int) -> None:
        runtime = self._active_runtime
        if runtime is None:
            return

        if step != runtime.request.target_step - 1:
            return

        self._finish_pending_gradient_transfer(runtime, step)
        self._wait_for_reconstruction()

        for partition_index in range(len(runtime.partitions)):
            self._wait_partition_transfer(runtime, partition_index)

        expected_version = runtime.request.target_step
        for snapshot in runtime.transferred_blocks.values():
            if snapshot.version_step != expected_version:
                raise RuntimeError(
                    "GoCkpt-O reconstruction is incomplete; not all partitions "
                    f"reached target step {expected_version}."
                )

        checkpoint = self._assemble_checkpoint(runtime)
        result = runtime.result
        if result is None:
            raise RuntimeError("GoCkpt-O runtime is missing checkpoint metadata.")

        worker = threading.Thread(
            target=self._persist_checkpoint_worker,
            args=(checkpoint, result.path, result),
            daemon=True,
            name=f"gockpt-o-persist-{result.target_step}",
        )
        with self._pending_lock:
            self._pending_thread = worker
            self._pending_result = result
            self._pending_error = None

        self._active_runtime = None
        self._pending_gradient_transfers.clear()
        self._reconstruction_futures_by_partition.clear()
        worker.start()

    def _collect_gradients_for_step(
        self,
        runtime: GoCkptRuntime,
        step: int,
    ) -> dict[str, torch.Tensor | None]:
        gradients: dict[str, torch.Tensor | None] = {}
        for name in runtime.transferred_blocks:
            partition_index = runtime.partition_name_to_index[name]
            transfer_step = runtime.request.start_step + partition_index
            if step < transfer_step:
                continue

            param = self._params_by_name[name]
            grad = param.grad
            if grad is None:
                gradients[name] = None
                continue

            gradients[name] = grad.detach()

        return gradients

    def _submit_async_gradient_transfer(
        self,
        step: int,
        gradients: dict[str, torch.Tensor | None],
    ) -> PendingGradientTransfer:
        cpu_gradients: dict[str, torch.Tensor | None] = {}
        source_refs: dict[str, torch.Tensor] = {}

        if self._gradient_stream is None:
            for name, grad in gradients.items():
                if grad is None:
                    cpu_gradients[name] = None
                    continue
                source_refs[name] = grad
                cpu_gradients[name] = grad.cpu().clone()

            return PendingGradientTransfer(
                step=step,
                gradients=cpu_gradients,
                source_refs=source_refs,
                event=None,
                submitted_at=time.perf_counter(),
            )

        current_stream = torch.cuda.current_stream()
        self._gradient_stream.wait_stream(current_stream)
        submitted_at = time.perf_counter()

        with torch.cuda.stream(self._gradient_stream):
            for name, grad in gradients.items():
                if grad is None:
                    cpu_gradients[name] = None
                    continue

                source_refs[name] = grad
                cpu_tensor = torch.empty_like(grad, device="cpu", pin_memory=True)
                cpu_tensor.copy_(grad, non_blocking=True)
                cpu_gradients[name] = cpu_tensor

        event = torch.cuda.Event()
        event.record(self._gradient_stream)
        return PendingGradientTransfer(
            step=step,
            gradients=cpu_gradients,
            source_refs=source_refs,
            event=event,
            submitted_at=submitted_at,
        )

    def _finish_pending_gradient_transfer(
        self,
        runtime: GoCkptRuntime,
        step: int,
    ) -> None:
        pending = self._pending_gradient_transfers.pop(step, None)
        if pending is None:
            return

        if pending.event is not None:
            pending.event.synchronize()

        self._submit_reconstruction(runtime, pending)

        if runtime.result is not None:
            runtime.result.gradient_duration_sec += (
                time.perf_counter() - pending.submitted_at
            )

    def _submit_reconstruction(
        self,
        runtime: GoCkptRuntime,
        pending: PendingGradientTransfer,
    ) -> None:
        gradients_by_partition: dict[int, dict[str, torch.Tensor | None]] = {}
        for name, grad in pending.gradients.items():
            partition_index = runtime.partition_name_to_index[name]
            gradients_by_partition.setdefault(partition_index, {})[name] = grad

        for partition_index, gradients in gradients_by_partition.items():
            with self._reconstruction_lock:
                previous_future = self._reconstruction_futures_by_partition.get(
                    partition_index
                )
                future = self._reconstruction_pool.submit(
                    self._reconstruct_partition_worker,
                    runtime,
                    pending.step,
                    partition_index,
                    gradients,
                    previous_future,
                )
                self._reconstruction_futures_by_partition[partition_index] = future

    def _reconstruct_partition_worker(
        self,
        runtime: GoCkptRuntime,
        step: int,
        partition_index: int,
        gradients: dict[str, torch.Tensor | None],
        previous_future: Future[None] | None,
    ) -> None:
        if previous_future is not None:
            previous_future.result()

        reconstruction_start = time.perf_counter()
        self._wait_partition_transfer(runtime, partition_index)

        for name, grad in gradients.items():
            snapshot = runtime.transferred_blocks[name]
            if snapshot.version_step != step:
                raise RuntimeError(
                    "GoCkpt-O reconstruction order violation for "
                    f"{name}: expected version {step}, got {snapshot.version_step}."
                )
            if grad is not None:
                self._apply_cpu_adamw_update(snapshot, grad)
            snapshot.version_step = step + 1

        if runtime.result is not None:
            with self._reconstruction_lock:
                runtime.result.reconstruction_duration_sec += (
                    time.perf_counter() - reconstruction_start
                )

    def _wait_for_reconstruction(self) -> None:
        with self._reconstruction_lock:
            futures = list(self._reconstruction_futures_by_partition.values())

        for future in futures:
            future.result()
