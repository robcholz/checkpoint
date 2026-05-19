from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_TYPES = ("baseline", "async", "async_o", "gockpt", "gockpt_o")


@dataclass
class FinetuneBenchmarkRun:
    hook_type: str
    output_dir: str
    command: list[str]
    returncode: int
    wall_time_sec: float
    train_runtime_sec: float | None
    train_steps_per_sec: float | None
    checkpoint_count: int
    checkpoint_files: list[str]
    phase_summary: dict[str, Any]
    checkpoint_results: list[dict[str, Any]]
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real Qwen finetune benchmark across checkpoint hooks."
    )
    parser.add_argument(
        "--hook-types",
        nargs="+",
        default=list(HOOK_TYPES),
        choices=HOOK_TYPES,
        help="Hook implementations to benchmark.",
    )
    parser.add_argument("--seq-len", type=int, default=512, choices=(256, 512))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=20, choices=(10, 20))
    parser.add_argument("--overlap-steps", type=int, default=7)
    parser.add_argument(
        "--gockpt-inflight-packets",
        "--gockpt-reconstruction-queue-depth",
        dest="gockpt_reconstruction_queue_depth",
        type=int,
        default=None,
        help="Max in-flight GoCkpt reconstruction packets forwarded to finetune.py.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/finetune_runs"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=12.0,
        help="Fail before launching runs unless GPU 0 has at least this much free memory.",
    )
    parser.add_argument(
        "--save-final-model",
        action="store_true",
        help="Persist the final Hugging Face model for each run. Disabled by default to focus on checkpoint timing.",
    )
    return parser.parse_args()


def get_gpu0_free_gb() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if completed.returncode != 0:
        return None

    lines = completed.stdout.strip().splitlines()
    if not lines:
        return None

    first_line = lines[0]
    return float(first_line) / 1024.0


def check_gpu_capacity(args: argparse.Namespace) -> None:
    if args.min_free_gb <= 0:
        return

    free_gb = get_gpu0_free_gb()
    if free_gb is None or free_gb >= args.min_free_gb:
        return

    raise RuntimeError(
        f"GPU 0 has only {free_gb:.2f} GiB free, below --min-free-gb "
        f"{args.min_free_gb:.2f}. Free the RTX 4090 or pass a lower threshold "
        "for a constrained smoke run."
    )


def build_command(args: argparse.Namespace, hook_type: str, run_dir: Path) -> list[str]:
    command = [
        str(args.python),
        str(REPO_ROOT / "finetune.py"),
        "--hook-type",
        hook_type,
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        "1",
        "--max-steps",
        str(args.max_steps),
        "--save-steps",
        str(args.save_steps),
        "--overlap-steps",
        str(args.overlap_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--logging-steps",
        str(args.logging_steps),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(run_dir),
        "--profile-phases",
    ]
    if args.gockpt_reconstruction_queue_depth is not None:
        command.extend([
            "--gockpt-inflight-packets",
            str(args.gockpt_reconstruction_queue_depth),
        ])
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if not args.save_final_model:
        command.append("--skip-final-model-save")
    return command


def run_one(args: argparse.Namespace, hook_type: str) -> FinetuneBenchmarkRun:
    run_dir = args.output_dir / hook_type
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args, hook_type, run_dir)

    log_path = run_dir / "stdout.log"
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    wall_time = time.perf_counter() - start

    summary_path = run_dir / "run_summary.json"
    summary: dict[str, Any] = {}
    error = None
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    elif completed.returncode != 0:
        error = f"run failed; see {log_path}"
    else:
        error = f"missing run summary at {summary_path}"

    checkpoint_files = summary.get("checkpoint_files", [])
    return FinetuneBenchmarkRun(
        hook_type=hook_type,
        output_dir=str(run_dir),
        command=command,
        returncode=completed.returncode,
        wall_time_sec=wall_time,
        train_runtime_sec=summary.get("train_runtime_sec"),
        train_steps_per_sec=summary.get("train_steps_per_sec"),
        checkpoint_count=len(checkpoint_files),
        checkpoint_files=checkpoint_files,
        phase_summary=summary.get("phase_summary", {}),
        checkpoint_results=summary.get("checkpoint_results", []),
        error=error,
    )


def phase_total(run: dict[str, Any], phase: str) -> float | None:
    phase_data = run.get("phase_summary", {}).get(phase)
    if phase_data is None:
        return None
    return float(phase_data["total_sec"])


def build_comparison(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_hook = {run["hook_type"]: run for run in runs if run["returncode"] == 0}
    baseline = by_hook.get("baseline")
    comparisons: dict[str, Any] = {}

    for hook_type, run in by_hook.items():
        item: dict[str, Any] = {
            "train_steps_per_sec": run.get("train_steps_per_sec"),
            "wall_time_sec": run.get("wall_time_sec"),
            "checkpoint_count": run.get("checkpoint_count"),
            "save_checkpoint_total_sec": phase_total(run, "hook.save_checkpoint"),
            "forward_begin_total_sec": phase_total(run, "hook.forward_begin"),
            "backward_end_total_sec": phase_total(run, "hook.backward_end"),
            "update_begin_total_sec": phase_total(run, "hook.update_begin"),
            "update_end_total_sec": phase_total(run, "hook.update_end"),
        }
        if baseline is not None and hook_type != "baseline":
            base_sps = baseline.get("train_steps_per_sec")
            run_sps = run.get("train_steps_per_sec")
            if base_sps and run_sps:
                item["speedup_vs_baseline"] = run_sps / base_sps
            base_ckpt = phase_total(baseline, "hook.save_checkpoint")
            run_ckpt = phase_total(run, "hook.save_checkpoint")
            if base_ckpt and run_ckpt is not None:
                item["save_checkpoint_time_ratio_vs_baseline"] = run_ckpt / base_ckpt
        comparisons[hook_type] = item

    return comparisons


def main() -> None:
    args = parse_args()
    check_gpu_capacity(args)

    if args.output_dir.exists() and not args.keep_output:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = [asdict(run_one(args, hook_type)) for hook_type in args.hook_types]
    report = {
        "config": {
            "hook_types": args.hook_types,
            "seq_len": args.seq_len,
            "max_steps": args.max_steps,
            "save_steps": args.save_steps,
            "overlap_steps": args.overlap_steps,
            "gockpt_reconstruction_queue_depth": args.gockpt_reconstruction_queue_depth,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "gradient_checkpointing": args.gradient_checkpointing,
            "save_final_model": args.save_final_model,
            "min_free_gb": args.min_free_gb,
        },
        "runs": runs,
        "comparison": build_comparison(runs),
    }

    report_path = args.output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    if any(run["returncode"] != 0 for run in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
