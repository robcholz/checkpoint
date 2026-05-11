# baseline_hook.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from src.pytorch_hook import PyTorchCheckpointHook


@dataclass
class BaselineCheckpointConfig:
    checkpoint_dir: str | Path = "./checkpoints/baseline"
    checkpoint_path: str | Path | None = None
    tag_prefix: str = "baseline_step"
    save_model: bool = True
    save_optimizer: bool = True
    save_rng_state: bool = False


@dataclass
class BaselineCheckpointResult:
    step: int
    tag: str
    path: Path
    duration_sec: float


class BaselineCheckpointHook(PyTorchCheckpointHook):
    """
    Synchronous torch.save baseline.

    This implements the same training-loop hook interface as GoCkpt,
    but only does normal blocking checkpoint save.

    Usage:
        if step % checkpoint_interval == 0:
            hook.save_checkpoint(step)

        hook.backward_begin(step)
        loss.backward()
        hook.backward_end(step)

        hook.update_begin(step)
        optimizer.step()
        hook.update_end(step)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        config: BaselineCheckpointConfig | None = None,
        checkpoint_builder: Callable[[int], dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config or BaselineCheckpointConfig()
        self.checkpoint_builder = checkpoint_builder

        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        if self.config.checkpoint_path is not None:
            Path(self.config.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.last_result: BaselineCheckpointResult | None = None
        self.history: list[BaselineCheckpointResult] = []

    def save_checkpoint(self, step: int) -> None:
        tag = f"{self.config.tag_prefix}_{step}"
        path = self._checkpoint_path(step)
        checkpoint = self._build_checkpoint(step, tag)
        duration = self.save_raw_checkpoint(checkpoint, path)

        result = BaselineCheckpointResult(
            step=step,
            tag=path.stem,
            path=path,
            duration_sec=duration,
        )

        self.last_result = result
        self.history.append(result)

    def backward_begin(self, step: int) -> None:
        # Baseline checkpoint does not need gradients.
        return

    def backward_end(self, step: int) -> None:
        # Baseline checkpoint does not capture param.grad.
        return

    def update_begin(self, step: int) -> None:
        # Baseline checkpoint does not capture model/optimizer blocks.
        return

    def update_end(self, step: int) -> None:
        # Baseline checkpoint does not reconstruct checkpoint state.
        return

    def _capture_rng_state(self) -> dict[str, Any]:
        rng_state: dict[str, Any] = {
            "torch_cpu": torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            rng_state["torch_cuda_all"] = torch.cuda.get_rng_state_all()

        return rng_state

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        load_model: bool = True,
        load_optimizer: bool = True,
        load_rng_state: bool = True,
    ) -> dict[str, Any]:
        ckpt = self.load_raw_checkpoint(checkpoint_path, map_location=map_location)

        if load_model and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])

        if load_optimizer and self.optimizer is not None and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        if load_rng_state and "rng_state" in ckpt:
            self._restore_rng_state(ckpt["rng_state"])

        return ckpt

    def _restore_rng_state(self, rng_state: dict[str, Any]) -> None:
        if "torch_cpu" in rng_state:
            torch.set_rng_state(rng_state["torch_cpu"])

        if torch.cuda.is_available() and "torch_cuda_all" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda_all"])

    @staticmethod
    def save_raw_checkpoint(
        checkpoint: dict[str, Any], checkpoint_path: str | Path
    ) -> float:
        start = time.perf_counter()
        torch.save(checkpoint, checkpoint_path)
        return time.perf_counter() - start

    @staticmethod
    def load_raw_checkpoint(
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        return torch.load(checkpoint_path, map_location=map_location)

    def _checkpoint_path(self, step: int) -> Path:
        if self.config.checkpoint_path is not None:
            return Path(self.config.checkpoint_path)
        tag = f"{self.config.tag_prefix}_{step}"
        return self.checkpoint_dir / f"{tag}.pt"

    def _build_checkpoint(self, step: int, tag: str) -> dict[str, Any]:
        if self.checkpoint_builder is not None:
            return self.checkpoint_builder(step)

        checkpoint: dict[str, Any] = {
            "step": step,
            "tag": tag,
            "time_unix": time.time(),
        }

        if self.config.save_model:
            checkpoint["model"] = self.model.state_dict()

        if self.config.save_optimizer and self.optimizer is not None:
            checkpoint["optimizer"] = self.optimizer.state_dict()

        if self.config.save_rng_state:
            checkpoint["rng_state"] = self._capture_rng_state()

        return checkpoint
