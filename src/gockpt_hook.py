from __future__ import annotations

import threading
import time
from queue import Queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from src.baseline_hook import BaselineCheckpointConfig, BaselineCheckpointHook
from src.pytorch_hook import (
    CheckpointRequest,
    OptimizerParamSnapshot,
    ParameterSnapshot,
)


@dataclass
class GoCkptCheckpointConfig(BaselineCheckpointConfig):
    overlap_steps: int = 7


@dataclass
class GoCkptCheckpointResult:
    start_step: int
    target_step: int
    tag: str
    path: Path
    transfer_duration_sec: float = 0.0
    gradient_duration_sec: float = 0.0
    reconstruction_duration_sec: float = 0.0
    persistence_duration_sec: float | None = None
    total_duration_sec: float | None = None


@dataclass
class GoCkptRuntime:
    request: CheckpointRequest
    partitions: list[list[str]]
    partition_name_to_index: dict[str, int]
    transferred_partitions: set[int] = field(default_factory=set)
    partition_events: dict[int, torch.cuda.Event | None] = field(default_factory=dict)
    transferred_blocks: dict[str, ParameterSnapshot] = field(default_factory=dict)
    gradients_by_step: dict[int, dict[str, torch.Tensor | None]] = field(default_factory=dict)
    result: GoCkptCheckpointResult | None = None
    reconstruction_queue: Queue | None = None
    reconstruction_thread: threading.Thread | None = None
    reconstruction_error: BaseException | None = None


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

        self._active_runtime: GoCkptRuntime | None = None
        self._pending_thread: threading.Thread | None = None
        self._pending_result: GoCkptCheckpointResult | None = None
        self._pending_error: BaseException | None = None
        self._pending_lock = threading.Lock()
        self._transfer_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )

        self._named_parameters = list(self.model.named_parameters())
        self._params_by_name = {name: param for name, param in self._named_parameters}
        self._param_name_by_obj = {id(param): name for name, param in self._named_parameters}
        self._buffers_by_name = dict(self.model.named_buffers())

        self._param_to_group: dict[int, dict[str, Any]] = {}
        self._param_to_state_id: dict[int, int] = {}
        self._optimizer_param_groups_template: list[dict[str, Any]] = []
        self._build_optimizer_metadata()

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

        self._schedule_partition_transfer(runtime, partition_index, step)

    def forward_end(self, step: int) -> None:
        return

    def backward_begin(self, step: int) -> None:
        return

    def backward_end(self, step: int) -> None:
        runtime = self._active_runtime
        if runtime is None or not self._is_step_in_request(runtime, step):
            return

        grad_start = time.perf_counter()
        gradients_for_step: dict[str, torch.Tensor | None] = {}
        gradient_refs: dict[str, torch.Tensor] = {}
        for name, snapshot in runtime.transferred_blocks.items():
            if snapshot.version_step > step:
                continue

            param = self._params_by_name[name]
            grad = param.grad
            if grad is None:
                gradients_for_step[name] = None
                continue

            gradient_refs[name] = grad.detach()

        gradients_for_step.update(self._copy_gradients_to_cpu_blocking(gradient_refs))

        runtime.gradients_by_step[step] = gradients_for_step
        if runtime.result is not None:
            runtime.result.gradient_duration_sec += time.perf_counter() - grad_start

    def update_begin(self, step: int) -> None:
        runtime = self._active_runtime
        if runtime is None or not self._is_step_in_request(runtime, step):
            return

        gradients_for_step = runtime.gradients_by_step.pop(step, {})
        if not gradients_for_step:
            return

        self._enqueue_reconstruction(runtime, step, gradients_for_step)

    def update_end(self, step: int) -> None:
        runtime = self._active_runtime
        if runtime is None:
            return
        if step != runtime.request.target_step - 1:
            return

        self._wait_for_reconstruction(runtime)
        self._stop_reconstruction_worker(runtime)

        expected_version = runtime.request.target_step
        for snapshot in runtime.transferred_blocks.values():
            if snapshot.version_step != expected_version:
                raise RuntimeError(
                    "GoCkpt reconstruction is incomplete; not all partitions "
                    f"reached target step {expected_version}."
                )

        checkpoint = self._assemble_checkpoint(runtime)
        result = runtime.result
        if result is None:
            raise RuntimeError("GoCkpt runtime is missing checkpoint result metadata.")

        worker = threading.Thread(
            target=self._persist_checkpoint_worker,
            args=(checkpoint, result.path, result),
            daemon=True,
            name=f"gockpt-persist-{result.target_step}",
        )
        with self._pending_lock:
            self._pending_thread = worker
            self._pending_result = result
            self._pending_error = None

        self._active_runtime = None
        worker.start()

    def wait_for_pending_persistence(self) -> None:
        if self._active_runtime is not None:
            self._stop_reconstruction_worker(self._active_runtime)
            self._active_runtime = None

        thread: threading.Thread | None
        with self._pending_lock:
            thread = self._pending_thread

        if thread is None:
            return

        thread.join()
        self._finalize_pending_thread(thread)

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

        if result is not None:
            self.last_result = result
            self.history.append(result)

        if error is not None:
            raise RuntimeError("Background checkpoint persistence failed.") from error

    def _start_reconstruction_worker(self, runtime: GoCkptRuntime) -> None:
        queue: Queue = Queue()
        worker = threading.Thread(
            target=self._reconstruction_worker,
            args=(runtime,),
            daemon=True,
            name=f"gockpt-reconstruct-{runtime.request.target_step}",
        )
        runtime.reconstruction_queue = queue
        runtime.reconstruction_thread = worker
        worker.start()

    def _enqueue_reconstruction(
        self,
        runtime: GoCkptRuntime,
        step: int,
        gradients_for_step: dict[str, torch.Tensor | None],
    ) -> None:
        if runtime.reconstruction_error is not None:
            raise RuntimeError("Background checkpoint reconstruction failed.") from runtime.reconstruction_error

        if runtime.reconstruction_queue is None:
            self._apply_reconstruction_task(runtime, step, gradients_for_step)
            return

        runtime.reconstruction_queue.put((step, gradients_for_step))

    def _wait_for_reconstruction(self, runtime: GoCkptRuntime) -> None:
        if runtime.reconstruction_queue is not None:
            runtime.reconstruction_queue.join()

        if runtime.reconstruction_error is not None:
            raise RuntimeError("Background checkpoint reconstruction failed.") from runtime.reconstruction_error

    def _stop_reconstruction_worker(self, runtime: GoCkptRuntime) -> None:
        queue = runtime.reconstruction_queue
        worker = runtime.reconstruction_thread
        if queue is None or worker is None:
            return

        queue.put(None)
        queue.join()
        worker.join()
        runtime.reconstruction_queue = None
        runtime.reconstruction_thread = None

    def _reconstruction_worker(self, runtime: GoCkptRuntime) -> None:
        queue = runtime.reconstruction_queue
        if queue is None:
            return

        while True:
            task = queue.get()
            try:
                if task is None:
                    return

                step, gradients_for_step = task
                self._apply_reconstruction_task(runtime, step, gradients_for_step)
            except BaseException as exc:
                runtime.reconstruction_error = exc
            finally:
                queue.task_done()

    def _apply_reconstruction_task(
        self,
        runtime: GoCkptRuntime,
        step: int,
        gradients_for_step: dict[str, torch.Tensor | None],
    ) -> None:
        reconstruction_start = time.perf_counter()
        waited_partitions: set[int] = set()
        for name in gradients_for_step:
            partition_index = runtime.partition_name_to_index[name]
            if partition_index in waited_partitions:
                continue
            self._wait_partition_transfer(runtime, partition_index)
            waited_partitions.add(partition_index)

        for name, grad in gradients_for_step.items():
            snapshot = runtime.transferred_blocks[name]
            if grad is not None:
                self._apply_cpu_adamw_update(snapshot, grad)
            snapshot.version_step = step + 1

        if runtime.result is not None:
            runtime.result.reconstruction_duration_sec += (
                time.perf_counter() - reconstruction_start
            )

    def _build_optimizer_metadata(self) -> None:
        if self.optimizer is None:
            return

        state_id = 0
        for group in self.optimizer.param_groups:
            group_template = {key: value for key, value in group.items() if key != "params"}
            group_param_ids: list[int] = []
            for param in group["params"]:
                param_id = id(param)
                self._param_to_group[param_id] = group_template.copy()
                self._param_to_state_id[param_id] = state_id
                group_param_ids.append(state_id)
                state_id += 1

            group_template["params"] = group_param_ids
            self._optimizer_param_groups_template.append(group_template)

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
        opt_state = self.optimizer.state.get(param, {}) if self.optimizer is not None else {}
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

    def _schedule_partition_transfer(
        self,
        runtime: GoCkptRuntime,
        partition_index: int,
        step: int,
    ) -> None:
        partition = runtime.partitions[partition_index]
        transfer_start = time.perf_counter()

        current_stream = torch.cuda.current_stream() if self._transfer_stream is not None else None
        if self._transfer_stream is not None and current_stream is not None:
            self._transfer_stream.wait_stream(current_stream)

        with torch.cuda.stream(self._transfer_stream) if self._transfer_stream is not None else _nullcontext():
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
                runtime.transferred_blocks[name] = snapshot

        event: torch.cuda.Event | None = None
        if self._transfer_stream is not None:
            event = torch.cuda.Event()
            event.record(self._transfer_stream)

        runtime.partition_events[partition_index] = event
        runtime.transferred_partitions.add(partition_index)
        if runtime.result is not None:
            runtime.result.transfer_duration_sec += time.perf_counter() - transfer_start

    def _snapshot_optimizer_state(self, param: torch.nn.Parameter) -> OptimizerParamSnapshot:
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
            param_group=self._param_to_group[id(param)].copy(),
        )

    def _copy_tensor_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cuda" or self._transfer_stream is None:
            return tensor.cpu().clone()

        cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
        cpu_tensor.copy_(tensor, non_blocking=True)
        return cpu_tensor

    def _copy_gradients_to_cpu_blocking(
        self,
        gradients: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not gradients:
            return {}

        if self._transfer_stream is None:
            return {name: grad.cpu().clone() for name, grad in gradients.items()}

        copied: dict[str, torch.Tensor] = {}
        stream = torch.cuda.current_stream()
        for name, grad in gradients.items():
            if grad.device.type != "cuda":
                copied[name] = grad.cpu().clone()
                continue

            cpu_tensor = torch.empty_like(grad, device="cpu", pin_memory=True)
            cpu_tensor.copy_(grad, non_blocking=True)
            copied[name] = cpu_tensor

        event = torch.cuda.Event()
        event.record(stream)
        event.synchronize()
        return copied

    def _wait_partition_transfer(self, runtime: GoCkptRuntime, partition_index: int) -> None:
        event = runtime.partition_events.get(partition_index)
        if event is None:
            return

        event.synchronize()
        runtime.partition_events[partition_index] = None

    def _apply_cpu_adamw_update(
        self,
        snapshot: ParameterSnapshot,
        grad: torch.Tensor,
    ) -> None:
        param = snapshot.param
        exp_avg = snapshot.optimizer_state.exp_avg
        exp_avg_sq = snapshot.optimizer_state.exp_avg_sq
        step_state = snapshot.optimizer_state.step
        group = snapshot.optimizer_state.param_group

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
                "param_groups": [group.copy() for group in self._optimizer_param_groups_template],
            },
        }
        if self.config.save_rng_state:
            checkpoint["rng_state"] = self._capture_rng_state()
        checkpoint["metadata"] = {
            "start_step": runtime.request.start_step,
            "target_step": runtime.request.target_step,
            "overlap_steps": self.overlap_steps,
            "num_partitions": len(runtime.partitions),
        }
        return checkpoint

    def _persist_checkpoint_worker(
        self,
        checkpoint: dict[str, Any],
        checkpoint_path: Path,
        result: GoCkptCheckpointResult,
    ) -> None:
        persistence_start = time.perf_counter()
        error: BaseException | None = None

        try:
            torch.save(checkpoint, checkpoint_path)
        except BaseException as exc:
            error = exc

        persistence_duration = time.perf_counter() - persistence_start
        result.persistence_duration_sec = persistence_duration
        result.total_duration_sec = (
            result.transfer_duration_sec
            + result.gradient_duration_sec
            + result.reconstruction_duration_sec
            + persistence_duration
        )

        with self._pending_lock:
            self._pending_error = error


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False
