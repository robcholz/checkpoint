#!/usr/bin/env python3
"""Sweep GoCkpt reconstruction ring-buffer size and plot pressure over time."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOOK_TYPES = ("gockpt", "gockpt_o")
HOOK_LABELS = {
    "gockpt": "GoCkpt",
    "gockpt_o": "GoCkpt-O",
}
HOOK_LINESTYLES = {
    "gockpt": "-",
    "gockpt_o": "--",
}
PACKET_COLORS = [
    "#006d77",
    "#ef6c00",
    "#c62828",
    "#5e35b1",
    "#2e7d32",
    "#455a64",
]


@dataclass
class PressureRunSummary:
    inflight_packets: int
    output_dir: str
    command: list[str]
    returncode: int
    report_path: str
    runs: list[dict[str, Any]]


def parse_packet_list(value: str) -> list[int]:
    packets: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        packet_count = int(item)
        if packet_count <= 0:
            raise argparse.ArgumentTypeError("in-flight packet sizes must be positive")
        packets.append(packet_count)
    if not packets:
        raise argparse.ArgumentTypeError("at least one in-flight packet size is required")
    return packets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real GoCkpt ring-buffer pressure benchmark."
    )
    parser.add_argument(
        "--gockpt-inflight-packets",
        type=parse_packet_list,
        required=True,
        help="Comma-separated reconstruction ring-buffer capacities, for example 64,128,256.",
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
        default=Path("benchmark/ringbuffer_pressure_runs"),
        help="Directory for per-capacity benchmark reports.",
    )
    parser.add_argument(
        "--hook-types",
        nargs="+",
        default=list(DEFAULT_HOOK_TYPES),
        choices=DEFAULT_HOOK_TYPES,
        help="Hook implementations to benchmark.",
    )
    parser.add_argument("--seq-len", type=int, default=512, choices=(256, 512))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=20, choices=(10, 20))
    parser.add_argument("--overlap-steps", type=int, default=7)
    parser.add_argument("--gockpt-transfer-chunk-mb", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gradient-checkpointing", action="store_true")
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


def build_command(args: argparse.Namespace, packet_count: int, run_dir: Path) -> list[str]:
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
        str(packet_count),
        "--gockpt-transfer-chunk-mb",
        str(args.gockpt_transfer_chunk_mb),
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
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    return command


def run_packet_size(args: argparse.Namespace, packet_count: int) -> PressureRunSummary:
    run_dir = args.output_dir / f"inflight_{packet_count}"
    if run_dir.exists() and not args.keep_output:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(args, packet_count, run_dir)
    log_path = run_dir / "ringbuffer_pressure_stdout.log"
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
    runs: list[dict[str, Any]] = []
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        raw_runs = report.get("runs", [])
        if isinstance(raw_runs, list):
            runs = raw_runs

    return PressureRunSummary(
        inflight_packets=packet_count,
        output_dir=str(run_dir),
        command=command,
        returncode=completed.returncode,
        report_path=str(report_path),
        runs=runs,
    )


def plot_pressure(summaries: list[PressureRunSummary], images_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    plotted = 0
    training_end_times: list[float] = []

    color_by_packets = {
        packet_count: PACKET_COLORS[index % len(PACKET_COLORS)]
        for index, packet_count in enumerate(summary.inflight_packets for summary in summaries)
    }

    for summary in summaries:
        color = color_by_packets[summary.inflight_packets]
        for run in summary.runs:
            hook_type = run.get("hook_type")
            if hook_type not in DEFAULT_HOOK_TYPES:
                continue
            train_runtime = run.get("train_runtime_sec")
            if isinstance(train_runtime, (int, float)):
                training_end_times.append(float(train_runtime))

            samples = run.get("ringbuffer_pressure_samples")
            if not isinstance(samples, list) or not samples:
                continue
            xs: list[float] = []
            ys: list[float] = []
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                time_sec = sample.get("time_sec")
                pressure_percent = sample.get("pressure_percent")
                if isinstance(time_sec, (int, float)) and isinstance(pressure_percent, (int, float)):
                    xs.append(float(time_sec))
                    ys.append(float(pressure_percent))
            if not xs:
                continue

            ax.step(
                xs,
                ys,
                where="post",
                linewidth=2.0,
                color=color,
                linestyle=HOOK_LINESTYLES.get(str(hook_type), "-"),
                label=f"{HOOK_LABELS.get(str(hook_type), hook_type)} / {summary.inflight_packets} packets",
                alpha=0.9,
            )
            plotted += 1

    if plotted == 0:
        raise ValueError("no ring-buffer pressure samples found in benchmark reports")

    if training_end_times:
        training_end = sum(training_end_times) / len(training_end_times)
        ax.axvline(
            training_end,
            color="#d32f2f",
            linestyle=":",
            linewidth=2.4,
            label="training end (avg)",
        )

    ax.set_title("GoCkpt Ring-Buffer Pressure vs Time")
    ax.set_xlabel("Time from training start (sec)")
    ax.set_ylabel("Ring-buffer pressure (%)")
    ax.set_ylim(-2, 105)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()

    output_path = images_dir / "ringbuffer_pressure.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_summary(summaries: list[PressureRunSummary], args: argparse.Namespace) -> Path:
    args_payload = vars(args).copy()
    args_payload["gockpt_inflight_packets"] = args.gockpt_inflight_packets
    args_payload["images_folder"] = str(args.images_folder)
    args_payload["output_dir"] = str(args.output_dir)
    args_payload["python"] = str(args.python)
    payload = {
        "config": args_payload,
        "runs": [asdict(summary) for summary in summaries],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "ringbuffer_pressure_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return summary_path


def main() -> None:
    args = parse_args()
    args.output_dir = (
        (REPO_ROOT / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir
    )
    images_dir = resolve_images_dir(args.images_folder)
    if images_dir.exists() and not args.keep_output:
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[PressureRunSummary] = []
    for packet_count in args.gockpt_inflight_packets:
        print(f"running in-flight packet size {packet_count}")
        summary = run_packet_size(args, packet_count)
        summaries.append(summary)
        if summary.returncode != 0:
            print(
                f"in-flight packet size {packet_count} failed; "
                f"see {summary.output_dir}/ringbuffer_pressure_stdout.log"
            )
            break

    summary_path = write_summary(summaries, args)
    image_path = plot_pressure(summaries, images_dir)
    print(f"summary written to {summary_path}")
    print(f"image written to {image_path}")
    if any(summary.returncode != 0 for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
