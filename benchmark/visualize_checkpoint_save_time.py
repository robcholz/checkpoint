#!/usr/bin/env python3
"""Plot foreground checkpoint stall time across checkpoint algorithms.

The input must be a JSON report produced by benchmark/finetune_benchmark.py.

The plotted value is foreground time spent inside checkpoint hook phases. This
is the paper-style metric for visible checkpoint overhead; it intentionally
does not include background persistence/reconstruction work unless the training
thread waits for it in a hook phase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HOOK_ORDER = ("baseline", "async", "async_o", "gockpt", "gockpt_o")
HOOK_LABELS = {
    "baseline": "Baseline",
    "async": "Async",
    "async_o": "Async-O",
    "gockpt": "GoCkpt",
    "gockpt_o": "GoCkpt-O",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize foreground checkpoint stall time across algorithms."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmark/finetune_runs/report.json"),
        help="Benchmark report JSON path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("benchmark/images/foreground_checkpoint_time.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Foreground Checkpoint Stall Time",
        help="Plot title.",
    )
    return parser.parse_args()


def read_report(report_path: Path) -> dict[str, Any]:
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_runs(report: dict[str, Any]) -> list[dict[str, Any]]:
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError(
            "report JSON must be produced by benchmark/finetune_benchmark.py "
            "and contain a top-level 'runs' list"
        )
    return runs


def phase_total(run: dict[str, Any], phase: str) -> float | None:
    phase_data = run.get("phase_summary", {}).get(phase)
    if not isinstance(phase_data, dict):
        return None

    total = phase_data.get("total_sec")
    if total is None:
        return None
    return float(total)


def foreground_phase_names(run: dict[str, Any]) -> list[str]:
    hook_type = run.get("hook_type")
    if hook_type == "baseline":
        return ["hook.save_checkpoint"]
    if hook_type == "async":
        return [
            "hook.save_checkpoint",
            "hook.forward_begin",
            "hook.wait_for_pending_persistence",
        ]
    if hook_type == "async_o":
        return [
            "hook.save_checkpoint",
            "hook.update_begin",
            "hook.wait_for_pending_persistence",
        ]
    if hook_type in {"gockpt", "gockpt_o"}:
        return [
            "hook.save_checkpoint",
            "hook.forward_begin",
            "hook.backward_begin",
            "hook.backward_end",
            "hook.update_begin",
            "hook.update_end",
            "hook.wait_for_pending_persistence",
        ]
    return ["hook.save_checkpoint"]


def collect_foreground_times(report: dict[str, Any]) -> list[tuple[str, float]]:
    by_hook: dict[str, float] = {}
    for run in get_runs(report):
        hook_type = run.get("hook_type")
        if not isinstance(hook_type, str):
            continue

        total = 0.0
        for phase in foreground_phase_names(run):
            total += phase_total(run, phase) or 0.0
        by_hook[hook_type] = total

    ordered = [(hook, by_hook[hook]) for hook in HOOK_ORDER if hook in by_hook]
    if not ordered:
        raise ValueError("no foreground checkpoint timing data found in report")
    return ordered


def plot_foreground_times(
    foreground_times: list[tuple[str, float]],
    output_path: Path,
    title: str,
    report_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [HOOK_LABELS.get(hook, hook) for hook, _ in foreground_times]
    values = [value for _, value in foreground_times]
    colors = ["#263238", "#607d8b", "#00897b", "#ef6c00", "#c62828"][: len(values)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(labels, values, color=colors, width=0.64)

    ax.set_title(title)
    ax.set_ylabel("Foreground checkpoint stall time (sec)")
    ax.set_xlabel("Checkpoint algorithm")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_value = max(values)
    ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1.0)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.text(
        0.01,
        0.01,
        f"source: {report_path}",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    report = read_report(args.report)
    foreground_times = collect_foreground_times(report)
    plot_foreground_times(foreground_times, args.output, args.title, args.report)
    print(f"saved {len(foreground_times)} foreground checkpoint-time bars to {args.output}")


if __name__ == "__main__":
    main()
