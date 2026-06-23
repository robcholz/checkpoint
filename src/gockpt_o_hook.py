from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.gockpt_hook import (
    GoCkptCheckpointConfig,
    GoCkptCheckpointHook,
    GoCkptCheckpointResult,
    GoCkptRuntime,
    PendingCheckpointWorker,
)


@dataclass
class GoCkptOCheckpointConfig(GoCkptCheckpointConfig):
    pass


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
    submit_duration_sec: float = 0.0


class GoCkptOCheckpointHook(GoCkptCheckpointHook):
    """
    Optimized GoCkpt variant with hidden gradient transfer.

    Difference from GoCkpt:
    - model/optimizer partition transfer is unchanged
    - gradient GPU->CPU transfer is submitted asynchronously at update_begin
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

    def save_checkpoint(self, step: int) -> None:
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

        self._pending_gradient_refs[step] = gradients

    def update_begin(self, step: int) -> None:
        runtime = self._active_runtime
        if runtime is None or not self._is_step_in_request(runtime, step):
            return

        runtime.optimizer_param_groups_by_step[step] = (
            self._snapshot_optimizer_param_groups_by_name()
        )
        gradients = self._pending_gradient_refs.pop(step, None)
        if gradients:
            self._pending_gradient_transfers[step] = (
                self._submit_async_gradient_transfer(step, gradients)
            )
        self._wait_current_partition_transfer_before_update(runtime, step)

    def update_end(self, step: int) -> None:
        runtime = self._active_runtime
        if runtime is None:
            return

        self._finish_pending_gradient_transfer_if_ready(runtime, step)

        if step != runtime.request.target_step - 1:
            return

        result = runtime.result
        if result is None:
            raise RuntimeError("GoCkpt-O runtime is missing checkpoint metadata.")

        pending = PendingCheckpointWorker(result=result)
        worker = threading.Thread(
            target=self._finalize_o_checkpoint_worker,
            args=(runtime, step, pending),
            daemon=True,
            name=f"gockpt-o-finalize-{result.target_step}",
        )
        pending.thread = worker
        with self._pending_lock:
            self._pending_workers.append(pending)

        self._active_runtime = None
        worker.start()

    def _finalize_o_checkpoint_worker(
        self,
        runtime: GoCkptRuntime,
        final_step: int,
        pending: PendingCheckpointWorker,
    ) -> None:
        result = runtime.result
        if result is None:
            pending.error = RuntimeError(
                "GoCkpt-O runtime is missing checkpoint metadata."
            )
            return

        try:
            self._finish_pending_gradient_transfer(runtime, final_step)
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
                        "GoCkpt-O reconstruction is incomplete; not all partitions "
                        f"reached target step {expected_version}."
                    )

            checkpoint = self._assemble_checkpoint(runtime)
            pending.error = self._persist_checkpoint_worker(
                checkpoint, result.path, result
            )
        except BaseException as exc:
            pending.error = exc

    def _collect_gradients_for_step(
        self,
        runtime: GoCkptRuntime,
        step: int,
    ) -> dict[str, torch.Tensor | None]:
        gradients: dict[str, torch.Tensor | None] = {}
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
            submitted_at = time.perf_counter()
            for name, grad in gradients.items():
                if grad is None:
                    cpu_gradients[name] = None
                    continue
                source_refs[name] = grad
                cpu_gradients[name] = grad.cpu().clone()

            submit_duration = time.perf_counter() - submitted_at
            return PendingGradientTransfer(
                step=step,
                gradients=cpu_gradients,
                source_refs=source_refs,
                event=None,
                submitted_at=submitted_at,
                submit_duration_sec=submit_duration,
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
                cpu_gradients[name] = self._copy_cuda_tensor_to_pinned_cpu(grad)

        event = torch.cuda.Event()
        event.record(self._gradient_stream)
        submit_duration = time.perf_counter() - submitted_at
        return PendingGradientTransfer(
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

    def _finish_pending_gradient_transfer_if_ready(
        self,
        runtime: GoCkptRuntime,
        step: int,
    ) -> None:
        pending = self._pending_gradient_transfers.get(step)
        if pending is None:
            return
        if pending.event is not None and not pending.event.query():
            return

        self._finish_pending_gradient_transfer(runtime, step)
