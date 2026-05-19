from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.async_hook import AsyncCheckpointHook
from src.async_o_hook import AsyncOCheckpointHook
from src.baseline_hook import BaselineCheckpointConfig, BaselineCheckpointHook
from src.gockpt_hook import GoCkptCheckpointConfig, GoCkptCheckpointHook
from src.gockpt_o_hook import GoCkptOCheckpointHook
from src.phase_profiler import PhaseProfiler, PhaseProfilingHook
from src.pytorch_hook import PyTorchCheckpointHook


HOOK_CLASSES = {
    "baseline": BaselineCheckpointHook,
    "async": AsyncCheckpointHook,
    "async_o": AsyncOCheckpointHook,
    "gockpt": GoCkptCheckpointHook,
    "gockpt_o": GoCkptOCheckpointHook,
}


@dataclass
class BenchmarkResult:
    hook_type: str
    checkpoint_path: str
    checkpoint_step: int
    passed: bool
    train_runtime_sec: float
    steps_per_sec: float
    checkpoint_size_bytes: int
    max_model_abs_diff: float
    max_optimizer_abs_diff: float
    max_resume_model_abs_diff: float
    max_resume_optimizer_abs_diff: float
    phase_summary: dict[str, dict[str, float | int]]
    checkpoint_result: dict[str, Any]
    error: str | None = None


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark checkpoint hooks for correctness and phase timing."
    )
    parser.add_argument(
        "--hook-types",
        nargs="+",
        default=list(HOOK_CLASSES.keys()),
        choices=tuple(HOOK_CLASSES.keys()),
    )
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--save-step", type=int, default=4)
    parser.add_argument("--overlap-steps", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark/checkpoint_runs")
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_batches(
    device: torch.device, steps: int, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    batches = []
    for _ in range(steps):
        x = torch.randn(4, 16, generator=generator)
        y = torch.randn(4, 8, generator=generator)
        batches.append((x.to(device), y.to(device)))
    return batches


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def clone_optimizer_state_portable(optimizer: AdamW) -> dict[str, Any]:
    state_dict = optimizer.state_dict()
    return _clone_to_cpu(state_dict)


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return value


def max_state_diff(left: dict[str, Any], right: dict[str, Any]) -> float:
    max_diff = 0.0
    for key, left_value in left.items():
        right_value = right[key]
        if isinstance(left_value, torch.Tensor):
            diff = (
                (left_value.detach().cpu() - right_value.detach().cpu())
                .abs()
                .max()
                .item()
            )
            max_diff = max(max_diff, float(diff))
    return max_diff


def is_within_tolerance(
    diff: float, reference: dict[str, Any], atol: float, rtol: float
) -> bool:
    max_reference = 0.0
    for value in reference.values():
        if isinstance(value, torch.Tensor):
            max_reference = max(
                max_reference, float(value.detach().cpu().abs().max().item())
            )
    return diff <= atol + rtol * max_reference


def max_optimizer_diff(left: dict[str, Any], right: dict[str, Any]) -> float:
    max_diff = 0.0
    left_state = left["state"]
    right_state = right["state"]
    if left["param_groups"] != right["param_groups"]:
        return float("inf")

    if set(left_state.keys()) != set(right_state.keys()):
        return float("inf")

    for state_id, left_entry in left_state.items():
        right_entry = right_state[state_id]
        if set(left_entry.keys()) != set(right_entry.keys()):
            return float("inf")
        for key, left_value in left_entry.items():
            right_value = right_entry[key]
            if isinstance(left_value, torch.Tensor):
                diff = (
                    (left_value.detach().cpu() - right_value.detach().cpu())
                    .abs()
                    .max()
                    .item()
                )
                max_diff = max(max_diff, float(diff))
            elif left_value != right_value:
                return float("inf")
    return max_diff


def is_optimizer_within_tolerance(
    diff: float, reference: dict[str, Any], atol: float, rtol: float
) -> bool:
    max_reference = 0.0
    for entry in reference["state"].values():
        for value in entry.values():
            if isinstance(value, torch.Tensor):
                max_reference = max(
                    max_reference, float(value.detach().cpu().abs().max().item())
                )
    return diff <= atol + rtol * max_reference


def create_hook(
    hook_type: str,
    model: nn.Module,
    optimizer: AdamW,
    checkpoint_dir: Path,
    overlap_steps: int,
) -> PhaseProfilingHook:
    hook_class = HOOK_CLASSES[hook_type]
    if hook_type in {"gockpt", "gockpt_o"}:
        config = GoCkptCheckpointConfig(
            checkpoint_dir=checkpoint_dir,
            tag_prefix=f"{hook_type}_step",
            save_model=True,
            save_optimizer=True,
            save_rng_state=True,
            overlap_steps=overlap_steps,
        )
    else:
        config = BaselineCheckpointConfig(
            checkpoint_dir=checkpoint_dir,
            tag_prefix=f"{hook_type}_step",
            save_model=True,
            save_optimizer=True,
            save_rng_state=True,
        )

    return PhaseProfilingHook(
        hook_class(model=model, optimizer=optimizer, config=config),
        profiler=PhaseProfiler(),
    )


def train_one_hook(args: argparse.Namespace, hook_type: str) -> BenchmarkResult:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    if (
        hook_type in {"gockpt", "gockpt_o"}
        and args.save_step + args.overlap_steps > args.steps
    ):
        raise ValueError("steps must be >= save_step + overlap_steps for GoCkpt hooks.")

    set_seed(args.seed)
    model = TinyModel().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    hook_dir = args.output_dir / hook_type
    hook_dir.mkdir(parents=True, exist_ok=True)
    hook = create_hook(
        hook_type=hook_type,
        model=model,
        optimizer=optimizer,
        checkpoint_dir=hook_dir,
        overlap_steps=args.overlap_steps,
    )
    batches = create_batches(device, args.steps, args.seed)
    reference_model_by_step: dict[int, dict[str, torch.Tensor]] = {}
    reference_optimizer_by_step: dict[int, dict[str, Any]] = {}

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_start = time.perf_counter()
    for step, (x, y) in enumerate(batches, start=1):
        reference_model_by_step[step] = clone_model_state(model)
        reference_optimizer_by_step[step] = clone_optimizer_state_portable(optimizer)

        if step == args.save_step:
            hook.save_checkpoint(step)

        hook.forward_begin(step)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        hook.forward_end(step)

        hook.backward_begin(step)
        loss.backward()
        hook.backward_end(step)

        hook.update_begin(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        hook.update_end(step)

    reference_model_by_step[args.steps + 1] = clone_model_state(model)
    reference_optimizer_by_step[args.steps + 1] = clone_optimizer_state_portable(
        optimizer
    )
    hook.wait_for_pending_persistence()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_runtime = time.perf_counter() - train_start

    if not hook.history:
        raise RuntimeError(f"{hook_type} did not produce a checkpoint.")

    checkpoint_result = hook.history[-1]
    checkpoint_path = Path(checkpoint_result.path)
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_step = int(raw_checkpoint["step"])
    if checkpoint_step not in reference_model_by_step:
        raise RuntimeError(
            f"{hook_type} checkpoint step {checkpoint_step} has no reference state."
        )

    set_seed(args.seed)
    loaded_model = TinyModel().to(device)
    loaded_optimizer = AdamW(loaded_model.parameters(), lr=1e-3, weight_decay=0.01)
    load_hook = create_hook(
        hook_type=hook_type,
        model=loaded_model,
        optimizer=loaded_optimizer,
        checkpoint_dir=hook_dir / "load_tmp",
        overlap_steps=args.overlap_steps,
    )
    load_hook.load_checkpoint(
        checkpoint_path,
        map_location=device,
        load_rng_state=False,
    )

    loaded_model_state = clone_model_state(loaded_model)
    loaded_optimizer_state = clone_optimizer_state_portable(loaded_optimizer)
    model_diff = max_state_diff(
        loaded_model_state, reference_model_by_step[checkpoint_step]
    )
    optimizer_diff = max_optimizer_diff(
        loaded_optimizer_state,
        reference_optimizer_by_step[checkpoint_step],
    )

    loaded_model.train()
    loaded_optimizer.zero_grad(set_to_none=True)
    for step in range(checkpoint_step, args.steps + 1):
        x, y = batches[step - 1]
        pred = loaded_model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        loaded_optimizer.step()
        loaded_optimizer.zero_grad(set_to_none=True)

    resume_model_state = clone_model_state(loaded_model)
    resume_optimizer_state = clone_optimizer_state_portable(loaded_optimizer)
    resume_model_diff = max_state_diff(
        resume_model_state,
        reference_model_by_step[args.steps + 1],
    )
    resume_optimizer_diff = max_optimizer_diff(
        resume_optimizer_state,
        reference_optimizer_by_step[args.steps + 1],
    )
    passed = (
        is_within_tolerance(
            model_diff, reference_model_by_step[checkpoint_step], args.atol, args.rtol
        )
        and is_optimizer_within_tolerance(
            optimizer_diff,
            reference_optimizer_by_step[checkpoint_step],
            args.atol,
            args.rtol,
        )
        and is_within_tolerance(
            resume_model_diff,
            reference_model_by_step[args.steps + 1],
            args.atol,
            args.rtol,
        )
        and is_optimizer_within_tolerance(
            resume_optimizer_diff,
            reference_optimizer_by_step[args.steps + 1],
            args.atol,
            args.rtol,
        )
    )

    return BenchmarkResult(
        hook_type=hook_type,
        checkpoint_path=str(checkpoint_path),
        checkpoint_step=checkpoint_step,
        passed=passed,
        train_runtime_sec=train_runtime,
        steps_per_sec=args.steps / train_runtime,
        checkpoint_size_bytes=checkpoint_path.stat().st_size,
        max_model_abs_diff=model_diff,
        max_optimizer_abs_diff=optimizer_diff,
        max_resume_model_abs_diff=resume_model_diff,
        max_resume_optimizer_abs_diff=resume_optimizer_diff,
        phase_summary=hook.profiler.summary(),
        checkpoint_result=_checkpoint_result_to_dict(checkpoint_result),
        error=None if passed else "checkpoint or resumed state differs from reference",
    )


def _checkpoint_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "__dataclass_fields__"):
        data = asdict(result)
    else:
        data = vars(result).copy()
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def phase_total(result: dict[str, Any], phase: str) -> float | None:
    summary = result.get("phase_summary", {})
    phase_data = summary.get(phase)
    if phase_data is None:
        return None
    return float(phase_data["total_sec"])


def build_performance_checks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hook = {result["hook_type"]: result for result in results if result["passed"]}
    checks: list[dict[str, Any]] = []

    def add_check(
        name: str,
        left_hook: str,
        left_phase: str,
        op: str,
        right_hook: str,
        right_phase: str,
    ) -> None:
        left_result = by_hook.get(left_hook)
        right_result = by_hook.get(right_hook)
        left_value = phase_total(left_result, left_phase) if left_result else None
        right_value = phase_total(right_result, right_phase) if right_result else None
        if left_value is None or right_value is None:
            passed = None
        elif op == "<":
            passed = left_value < right_value
        elif op == "<=":
            passed = left_value <= right_value
        else:
            raise ValueError(f"Unsupported performance check operator: {op}")

        checks.append(
            {
                "name": name,
                "left": {
                    "hook": left_hook,
                    "phase": left_phase,
                    "value_sec": left_value,
                },
                "operator": op,
                "right": {
                    "hook": right_hook,
                    "phase": right_phase,
                    "value_sec": right_value,
                },
                "passed": passed,
            }
        )

    add_check(
        "gockpt starts checkpoints faster than full baseline snapshot",
        "gockpt",
        "hook.save_checkpoint",
        "<",
        "baseline",
        "hook.save_checkpoint",
    )
    add_check(
        "gockpt_o starts checkpoints faster than full baseline snapshot",
        "gockpt_o",
        "hook.save_checkpoint",
        "<",
        "baseline",
        "hook.save_checkpoint",
    )
    add_check(
        "gockpt_o removes GoCkpt CPU replay from update_begin",
        "gockpt_o",
        "hook.update_begin",
        "<",
        "gockpt",
        "hook.update_begin",
    )
    add_check(
        "async_o has less blocking checkpoint start than async",
        "async_o",
        "hook.save_checkpoint",
        "<=",
        "async",
        "hook.save_checkpoint",
    )
    return checks


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.keep_output:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for hook_type in args.hook_types:
        try:
            result = train_one_hook(args, hook_type)
        except Exception as exc:
            result = BenchmarkResult(
                hook_type=hook_type,
                checkpoint_path="",
                checkpoint_step=-1,
                passed=False,
                train_runtime_sec=0.0,
                steps_per_sec=0.0,
                checkpoint_size_bytes=0,
                max_model_abs_diff=float("inf"),
                max_optimizer_abs_diff=float("inf"),
                max_resume_model_abs_diff=float("inf"),
                max_resume_optimizer_abs_diff=float("inf"),
                phase_summary={},
                checkpoint_result={},
                error=str(exc),
            )
        results.append(asdict(result))

    performance_checks = build_performance_checks(results)
    report = {
        "config": {
            "hook_types": args.hook_types,
            "steps": args.steps,
            "save_step": args.save_step,
            "overlap_steps": args.overlap_steps,
            "device": args.device,
            "seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "results": results,
        "performance_checks": performance_checks,
    }
    report_path = args.output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
