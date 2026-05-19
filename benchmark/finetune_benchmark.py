from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
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
    power_samples_file: str | None
    power_sample_count: int
    power_avg_w: float | None
    power_peak_w: float | None
    power_energy_j: float | None
    power_sampling_error: str | None
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
    parser.add_argument(
        "--power-sample-interval-sec",
        type=float,
        default=0.5,
        help="GPU power sampling interval in seconds. Set to 0 to disable power sampling.",
    )
    parser.add_argument(
        "--power-gpu-index",
        type=int,
        default=0,
        help="GPU index for nvidia-smi power sampling.",
    )
    args = parser.parse_args()
    if args.power_sample_interval_sec < 0:
        raise ValueError("--power-sample-interval-sec must be >= 0.")
    if args.power_gpu_index < 0:
        raise ValueError("--power-gpu-index must be >= 0.")
    return args


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


def query_gpu_power_w(gpu_index: int) -> tuple[float | None, str | None]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None, "nvidia-smi not found"

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if stderr:
            return None, stderr
        return None, "nvidia-smi returned non-zero exit status"

    lines = completed.stdout.strip().splitlines()
    if not lines:
        return None, "nvidia-smi returned empty power output"

    raw = lines[0].strip()
    if not raw:
        return None, "nvidia-smi returned blank power sample"
    if raw == "[Not Supported]":
        return None, "power draw query is not supported on this GPU"
    try:
        return float(raw), None
    except ValueError:
        return None, f"unable to parse nvidia-smi power sample: {raw!r}"


def summarize_power_samples(samples: list[dict[str, float]]) -> dict[str, float | int | None]:
    if not samples:
        return {
            "sample_count": 0,
            "duration_sec": None,
            "avg_power_w": None,
            "peak_power_w": None,
            "energy_j": None,
        }

    powers = [sample["power_w"] for sample in samples]
    energy_j = 0.0
    for previous, current in zip(samples, samples[1:]):
        dt = current["time_sec"] - previous["time_sec"]
        if dt <= 0:
            continue
        energy_j += 0.5 * (previous["power_w"] + current["power_w"]) * dt

    duration_sec = samples[-1]["time_sec"] - samples[0]["time_sec"]
    if duration_sec < 0:
        duration_sec = 0.0
    return {
        "sample_count": len(samples),
        "duration_sec": duration_sec,
        "avg_power_w": sum(powers) / len(powers),
        "peak_power_w": max(powers),
        "energy_j": energy_j,
    }


def sample_gpu_power_until_stopped(
    *,
    stop_event: threading.Event,
    gpu_index: int,
    interval_sec: float,
    samples: list[dict[str, float]],
    error_box: list[str],
) -> None:
    start = time.perf_counter()
    while True:
        power_w, error = query_gpu_power_w(gpu_index)
        elapsed = time.perf_counter() - start
        if power_w is not None:
            samples.append(
                {
                    "time_sec": elapsed,
                    "power_w": power_w,
                }
            )
        elif error and not error_box:
            error_box.append(error)
            break

        if stop_event.wait(interval_sec):
            break


def build_power_report(
    args: argparse.Namespace,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    power_runs: list[dict[str, Any]] = []
    for run in runs:
        run_dir = Path(run["output_dir"])
        samples_path = run_dir / "power_samples.json"
        samples_payload: dict[str, Any] = {}
        if samples_path.exists():
            with samples_path.open("r", encoding="utf-8") as handle:
                samples_payload = json.load(handle)

        power_runs.append(
            {
                "hook_type": run["hook_type"],
                "output_dir": run["output_dir"],
                "returncode": run["returncode"],
                "wall_time_sec": run["wall_time_sec"],
                "power_sampling_error": run.get("power_sampling_error"),
                "power_summary": samples_payload.get("summary", {}),
                "power_samples": samples_payload.get("samples", []),
            }
        )

    return {
        "config": {
            "hook_types": args.hook_types,
            "seq_len": args.seq_len,
            "max_steps": args.max_steps,
            "save_steps": args.save_steps,
            "overlap_steps": args.overlap_steps,
            "gockpt_reconstruction_queue_depth": args.gockpt_reconstruction_queue_depth,
            "gradient_checkpointing": args.gradient_checkpointing,
            "power_sample_interval_sec": args.power_sample_interval_sec,
            "power_gpu_index": args.power_gpu_index,
        },
        "runs": power_runs,
    }


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
    power_samples: list[dict[str, float]] = []
    power_error_box: list[str] = []
    power_stop_event = threading.Event()
    power_thread: threading.Thread | None = None
    if args.power_sample_interval_sec > 0:
        power_thread = threading.Thread(
            target=sample_gpu_power_until_stopped,
            kwargs={
                "stop_event": power_stop_event,
                "gpu_index": args.power_gpu_index,
                "interval_sec": args.power_sample_interval_sec,
                "samples": power_samples,
                "error_box": power_error_box,
            },
            daemon=True,
            name=f"gpu-power-sampler-{hook_type}",
        )
        power_thread.start()

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        returncode = process.wait()

    if power_thread is not None:
        power_stop_event.set()
        power_thread.join()

    wall_time = time.perf_counter() - start

    power_summary = summarize_power_samples(power_samples)
    power_samples_path = run_dir / "power_samples.json"
    with power_samples_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "hook_type": hook_type,
                "gpu_index": args.power_gpu_index,
                "sample_interval_sec": args.power_sample_interval_sec,
                "summary": power_summary,
                "error": power_error_box[0] if power_error_box else None,
                "samples": power_samples,
            },
            handle,
            indent=2,
        )

    summary_path = run_dir / "run_summary.json"
    summary: dict[str, Any] = {}
    error = None
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    elif returncode != 0:
        error = f"run failed; see {log_path}"
    else:
        error = f"missing run summary at {summary_path}"

    checkpoint_files = summary.get("checkpoint_files", [])
    return FinetuneBenchmarkRun(
        hook_type=hook_type,
        output_dir=str(run_dir),
        command=command,
        returncode=returncode,
        wall_time_sec=wall_time,
        train_runtime_sec=summary.get("train_runtime_sec"),
        train_steps_per_sec=summary.get("train_steps_per_sec"),
        checkpoint_count=len(checkpoint_files),
        checkpoint_files=checkpoint_files,
        phase_summary=summary.get("phase_summary", {}),
        checkpoint_results=summary.get("checkpoint_results", []),
        power_samples_file=str(power_samples_path),
        power_sample_count=int(power_summary["sample_count"]),
        power_avg_w=(
            float(power_summary["avg_power_w"])
            if power_summary["avg_power_w"] is not None
            else None
        ),
        power_peak_w=(
            float(power_summary["peak_power_w"])
            if power_summary["peak_power_w"] is not None
            else None
        ),
        power_energy_j=(
            float(power_summary["energy_j"])
            if power_summary["energy_j"] is not None
            else None
        ),
        power_sampling_error=power_error_box[0] if power_error_box else None,
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
            "power_sample_interval_sec": args.power_sample_interval_sec,
            "power_gpu_index": args.power_gpu_index,
        },
        "runs": runs,
        "comparison": build_comparison(runs),
    }

    report_path = args.output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    power_report_path = args.output_dir / "power.json"
    with power_report_path.open("w", encoding="utf-8") as handle:
        json.dump(build_power_report(args, runs), handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"power report saved to {power_report_path}")
    if any(run["returncode"] != 0 for run in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
