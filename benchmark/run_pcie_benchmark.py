#!/usr/bin/env python3
"""Sweep GoCkpt GPU-to-CPU transfer chunk sizes with real finetune runs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOOK_TYPES = ("gockpt", "gockpt_o")
FOREGROUND_PHASES = (
    "hook.save_checkpoint",
    "hook.forward_begin",
    "hook.backward_begin",
    "hook.backward_end",
    "hook.update_begin",
    "hook.update_end",
)
FINAL_DRAIN_PHASE = "hook.wait_for_pending_persistence"
HOOK_LABELS = {
    "gockpt": "GoCkpt",
    "gockpt_o": "GoCkpt-O",
}
HOOK_COLORS = {
    "gockpt": "#ef6c00",
    "gockpt_o": "#c62828",
}


@dataclass
class ChunkRunSummary:
    transfer_chunk_mb: float
    output_dir: str
    command: list[str]
    returncode: int
    report_path: str
    metrics: dict[str, Any]


def parse_chunk_list(value: str) -> list[float]:
    chunks: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        chunk = float(item)
        if chunk < 0:
            raise argparse.ArgumentTypeError("transfer chunk sizes must be >= 0")
        chunks.append(chunk)
    if not chunks:
        raise argparse.ArgumentTypeError("at least one transfer chunk size is required")
    return chunks


def format_chunk_label(chunk: float) -> str:
    if chunk == int(chunk):
        return str(int(chunk))
    return str(chunk).replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real GoCkpt PCIe transfer chunk-size benchmark."
    )
    parser.add_argument(
        "--transfer-chunk-mb",
        type=parse_chunk_list,
        required=True,
        help="Comma-separated transfer chunk sizes in MiB, for example 0,4,8,16,32.",
    )
    parser.add_argument(
        "--images-folder",
        type=Path,
        required=True,
        help="Output image folder. Relative paths are placed under benchmark/images/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/pcie_runs"),
        help="Directory for per-chunk benchmark reports.",
    )
    parser.add_argument(
        "--hook-types",
        nargs="+",
        default=list(DEFAULT_HOOK_TYPES),
        choices=DEFAULT_HOOK_TYPES,
        help="Hook implementations to sweep.",
    )
    parser.add_argument("--seq-len", type=int, default=512, choices=(256, 512))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=20, choices=(10, 20))
    parser.add_argument("--overlap-steps", type=int, default=7)
    parser.add_argument("--gockpt-inflight-packets", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Accepted for compatibility. Gradient checkpointing is enabled by default.",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=12.0)
    parser.add_argument(
        "--power-sample-interval-sec",
        type=float,
        default=0.0,
        help="Forwarded to finetune_benchmark.py. Default disables power sampling.",
    )
    return parser.parse_args()


def resolve_images_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / "benchmark" / "images" / path


def phase_total(run: dict[str, Any], phase: str) -> float:
    phase_data = run.get("phase_summary", {}).get(phase)
    if not isinstance(phase_data, dict):
        return 0.0
    return float(phase_data.get("total_sec") or 0.0)


def checkpoint_result_sum(run: dict[str, Any], key: str) -> float:
    total = 0.0
    for result in run.get("checkpoint_results", []):
        value = result.get(key)
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def collect_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for run in report.get("runs", []):
        hook_type = run.get("hook_type")
        if not isinstance(hook_type, str):
            continue
        foreground = sum(phase_total(run, phase) for phase in FOREGROUND_PHASES)
        final_drain = phase_total(run, FINAL_DRAIN_PHASE)
        metrics[hook_type] = {
            "returncode": run.get("returncode"),
            "train_steps_per_sec": run.get("train_steps_per_sec"),
            "train_runtime_sec": run.get("train_runtime_sec"),
            "wall_time_sec": run.get("wall_time_sec"),
            "checkpoint_count": run.get("checkpoint_count"),
            "foreground_checkpoint_time_sec": foreground,
            "foreground_plus_final_drain_sec": foreground + final_drain,
            "final_drain_sec": final_drain,
            "save_checkpoint_sec": phase_total(run, "hook.save_checkpoint"),
            "forward_begin_sec": phase_total(run, "hook.forward_begin"),
            "backward_begin_sec": phase_total(run, "hook.backward_begin"),
            "backward_end_sec": phase_total(run, "hook.backward_end"),
            "update_begin_sec": phase_total(run, "hook.update_begin"),
            "update_end_sec": phase_total(run, "hook.update_end"),
            "transfer_duration_sec": checkpoint_result_sum(run, "transfer_duration_sec"),
            "gradient_duration_sec": checkpoint_result_sum(run, "gradient_duration_sec"),
            "reconstruction_duration_sec": checkpoint_result_sum(run, "reconstruction_duration_sec"),
            "reconstruction_backpressure_sec": checkpoint_result_sum(run, "reconstruction_backpressure_sec"),
            "persistence_duration_sec": checkpoint_result_sum(run, "persistence_duration_sec"),
        }
    return metrics


def build_command(args: argparse.Namespace, chunk: float, run_dir: Path) -> list[str]:
    command = [
        str(args.python),
        str(REPO_ROOT / "benchmark" / "finetune_benchmark.py"),
        "--hook-types",
        *args.hook_types,
        "--seq-len",
        str(args.seq_len),
        "--max-steps",
        str(args.max_steps),
        "--save-steps",
        str(args.save_steps),
        "--overlap-steps",
        str(args.overlap_steps),
        "--gockpt-inflight-packets",
        str(args.gockpt_inflight_packets),
        "--gockpt-transfer-chunk-mb",
        str(chunk),
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
        "--min-free-gb",
        str(args.min_free_gb),
        "--power-sample-interval-sec",
        str(args.power_sample_interval_sec),
    ]
    if not args.no_gradient_checkpointing:
        command.append("--gradient-checkpointing")
    return command


def run_chunk(args: argparse.Namespace, chunk: float) -> ChunkRunSummary:
    label = format_chunk_label(chunk)
    run_dir = args.output_dir / f"chunk_{label}mb"
    if run_dir.exists() and not args.keep_output:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(args, chunk, run_dir)
    log_path = run_dir / "pcie_benchmark_stdout.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    report_path = run_dir / "report.json"
    report: dict[str, Any] = {}
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)

    return ChunkRunSummary(
        transfer_chunk_mb=chunk,
        output_dir=str(run_dir),
        command=command,
        returncode=completed.returncode,
        report_path=str(report_path),
        metrics=collect_metrics(report) if report else {},
    )


def plot_metric(
    summaries: list[ChunkRunSummary],
    images_dir: Path,
    metric_key: str,
    filename: str,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for hook_type in DEFAULT_HOOK_TYPES:
        xs: list[float] = []
        ys: list[float] = []
        for summary in summaries:
            hook_metrics = summary.metrics.get(hook_type, {})
            value = hook_metrics.get(metric_key)
            if isinstance(value, (int, float)):
                xs.append(summary.transfer_chunk_mb)
                ys.append(float(value))
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            color=HOOK_COLORS.get(hook_type),
            label=HOOK_LABELS.get(hook_type, hook_type),
        )
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.2f}", fontsize=8, ha="center", va="bottom")

    ax.set_title(title)
    ax.set_xlabel("Transfer chunk size (MiB, 0 = whole tensor)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(images_dir / filename, dpi=180)
    plt.close(fig)


def write_summary(summaries: list[ChunkRunSummary], args: argparse.Namespace, images_dir: Path) -> Path:
    args_payload = vars(args).copy()
    args_payload["transfer_chunk_mb"] = args.transfer_chunk_mb
    args_payload["images_folder"] = str(args.images_folder)
    args_payload["output_dir"] = str(args.output_dir)
    args_payload["python"] = str(args.python)

    payload = {
        "config": args_payload,
        "images_dir": str(images_dir),
        "runs": [asdict(summary) for summary in summaries],
    }
    summary_path = args.output_dir / "pcie_benchmark_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with (images_dir / "pcie_benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return summary_path


def main() -> None:
    args = parse_args()
    args.output_dir = (REPO_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    images_dir = resolve_images_dir(args.images_folder)
    images_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[ChunkRunSummary] = []
    for chunk in args.transfer_chunk_mb:
        print(f"running transfer chunk {chunk:g} MiB")
        summary = run_chunk(args, chunk)
        summaries.append(summary)
        if summary.returncode != 0:
            print(f"chunk {chunk:g} MiB failed; see {summary.output_dir}/pcie_benchmark_stdout.log")
            break

    summary_path = write_summary(summaries, args, images_dir)
    plot_metric(
        summaries,
        images_dir,
        "foreground_checkpoint_time_sec",
        "foreground_checkpoint_time_vs_transfer_chunk.png",
        "Foreground Checkpoint Stall vs PCIe Transfer Chunk Size",
        "Foreground checkpoint stall time (sec)",
    )
    plot_metric(
        summaries,
        images_dir,
        "train_steps_per_sec",
        "throughput_vs_transfer_chunk.png",
        "Training Throughput vs PCIe Transfer Chunk Size",
        "Training throughput (steps/sec)",
    )
    plot_metric(
        summaries,
        images_dir,
        "transfer_duration_sec",
        "transfer_duration_vs_transfer_chunk.png",
        "Checkpoint GPU-to-CPU Transfer Duration vs Chunk Size",
        "Transfer duration sum (sec)",
    )
    plot_metric(
        summaries,
        images_dir,
        "forward_begin_sec",
        "forward_begin_vs_transfer_chunk.png",
        "Forward-Begin Stall vs PCIe Transfer Chunk Size",
        "hook.forward_begin total (sec)",
    )

    print(f"summary written to {summary_path}")
    print(f"images written to {images_dir}")
    if any(summary.returncode != 0 for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
