#!/usr/bin/env python3
"""Visualize finetune benchmark results from benchmark/finetune_runs/report.json."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create plots for finetune benchmark runs."
    )
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=Path("benchmark/finetune_runs/report.json"),
        help="Path to the benchmark report JSON.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("benchmark/images/finetune_runs"),
        help="Directory for saved plots.",
    )
    return parser.parse_args()


def load_runs(report_path: Path) -> list[dict[str, Any]]:
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    runs = [run for run in report.get("runs", []) if run.get("returncode") == 0]
    if not runs:
        raise ValueError(f"no successful runs found in {report_path}")
    return runs


def phase_total(run: dict[str, Any], phase: str) -> float:
    phase_data = run.get("phase_summary", {}).get(phase, {})
    return float(phase_data.get("total_sec", 0.0))


def checkpoint_points(run: dict[str, Any]) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    durations: list[float] = []
    for result in run.get("checkpoint_results", []):
        step = result.get("step")
        duration = result.get("duration_sec", result.get("total_duration_sec"))
        if step is None or duration is None:
            continue
        steps.append(int(step))
        durations.append(float(duration))
    return steps, durations


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_throughput_plot(plt, runs: list[dict[str, Any]], output_path: Path) -> None:
    hooks = [run["hook_type"] for run in runs]
    steps_per_sec = [float(run["train_steps_per_sec"]) for run in runs]

    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(hooks, steps_per_sec, color="#2a6f97")
    plt.title("Training Throughput by Checkpoint Algorithm")
    plt.xlabel("Algorithm")
    plt.ylabel("Steps / sec")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, steps_per_sec):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_checkpoint_overhead_plot(
    plt, runs: list[dict[str, Any]], output_path: Path
) -> None:
    hooks = [run["hook_type"] for run in runs]
    totals = [phase_total(run, "hook.save_checkpoint") for run in runs]

    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(hooks, totals, color="#bc4749")
    plt.title("Total Checkpoint Save Time")
    plt.xlabel("Algorithm")
    plt.ylabel("Seconds")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, totals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_tradeoff_plot(plt, runs: list[dict[str, Any]], output_path: Path) -> None:
    checkpoint_totals = [phase_total(run, "hook.save_checkpoint") for run in runs]
    steps_per_sec = [float(run["train_steps_per_sec"]) for run in runs]
    hooks = [run["hook_type"] for run in runs]

    plt.figure(figsize=(8.5, 6))
    plt.scatter(checkpoint_totals, steps_per_sec, s=80, color="#6a994e")
    for hook, x_value, y_value in zip(hooks, checkpoint_totals, steps_per_sec):
        plt.annotate(
            hook, (x_value, y_value), xytext=(6, 6), textcoords="offset points"
        )
    plt.title("Checkpoint Cost vs Training Throughput")
    plt.xlabel("Total checkpoint save time (sec)")
    plt.ylabel("Steps / sec")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_checkpoint_timeline_plot(
    plt, runs: list[dict[str, Any]], output_path: Path
) -> None:
    plt.figure(figsize=(10, 6))
    for run in runs:
        steps, durations = checkpoint_points(run)
        if not steps:
            continue
        plt.plot(steps, durations, marker="o", linewidth=1.8, label=run["hook_type"])
    plt.title("Checkpoint Duration by Save Step")
    plt.xlabel("Training step")
    plt.ylabel("Checkpoint duration (sec)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    runs = load_runs(args.report)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt = setup_matplotlib()
    save_throughput_plot(plt, runs, args.output_dir / "throughput.png")
    save_checkpoint_overhead_plot(
        plt, runs, args.output_dir / "checkpoint_overhead.png"
    )
    save_tradeoff_plot(plt, runs, args.output_dir / "tradeoff.png")
    save_checkpoint_timeline_plot(
        plt, runs, args.output_dir / "checkpoint_timeline.png"
    )

    print(f"saved 4 plots to {args.output_dir}")


if __name__ == "__main__":
    main()
