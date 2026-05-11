from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class OptimizerParamSnapshot:
    """
    One parameter's optimizer state snapshot.

    For AdamW:
        exp_avg    = first moment
        exp_avg_sq = second moment
        step       = optimizer step
    """

    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: int | torch.Tensor
    param_group: dict[str, Any]


@dataclass
class ParameterSnapshot:
    """
    One parameter block snapshot.

    This is the basic unit GoCkpt transfers/reconstructs.
    """

    name: str
    param: torch.Tensor
    grad: torch.Tensor | None
    optimizer_state: OptimizerParamSnapshot
    version_step: int


@dataclass
class CheckpointRequest:
    """
    One GoCkpt checkpoint request.
    """

    start_step: int
    target_step: int
    tag: str


@dataclass
class CheckpointRuntimeState:
    """
    Runtime state for one active GoCkpt process.
    """

    request: CheckpointRequest
    transferred_blocks: dict[str, ParameterSnapshot] = field(default_factory=dict)
    gradients_by_step: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)


@dataclass
class CheckpointLoadResult:
    checkpoint_path: Path
    loaded_step: int
    resume_step: int
    tag: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PyTorchCheckpointHook(ABC):
    """
    Minimal hook interface for reproducing GoCkpt in a PyTorch training loop.

    Required insertion points:

        save_checkpoint(step)
        backward_begin(step)
        update_begin(step)
    """

    @abstractmethod
    def save_checkpoint(self, step: int) -> None:
        """
        Called when checkpoint starts.

        Paper equivalent:
            save_checkpoint

        This should initialize one GoCkpt checkpoint request.
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> CheckpointLoadResult:
        """
        Restore model / optimizer / metadata from checkpoint.
        Returns:
            CheckpointLoadResult, including resume_step.
        Usually:
            loaded_step = checkpoint["step"]
            resume_step = loaded_step + 1
        """

        raise NotImplementedError

    @abstractmethod
    def backward_begin(self, step: int) -> None:
        """
        Called before loss.backward().

        Paper equivalent:
            backward_begin
        """
        raise NotImplementedError

    @abstractmethod
    def backward_end(self, step: int) -> None:
        """
        Called after loss.backward().

        This is where gradients exist, so GoCkpt can capture param.grad.
        """
        raise NotImplementedError

    @abstractmethod
    def update_begin(self, step: int) -> None:
        """
        Called before optimizer.step().

        Paper equivalent:
            update_begin

        This is where GoCkpt can capture model/optimizer blocks
        before the GPU optimizer updates them.
        """
        raise NotImplementedError

    @abstractmethod
    def update_end(self, step: int) -> None:
        """
        Called after optimizer.step().
        """
        raise NotImplementedError
