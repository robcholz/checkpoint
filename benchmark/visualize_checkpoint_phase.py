#!/usr/bin/env python3
"""Plot average per-step foreground overhead for forward, backward, and update phases.

Input must be a JSON report produced by benchmark/finetune_benchmark.py.

Each algorithm gets its own horizontal track on the y-axis with up to three
stacked bars: forward (yellow), backward (orange), and update (red).
Only checkpoint-induced overhead is shown (not the computation itself).
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

PHASE_KEYS = {
    "forward": "raw_foreground_forward",
    "backward": "raw_foreground_backward",
    "update": "raw_foreground_update",
}
PHASE_COLORS = {
    "forward": "#fdd835",
    "backward": "#fb8c00",
    "update": "#e53935",
}
PHASE_LABELS = {
    "forward": "Forward",
    "backward": "Backward",
    "update": "Update",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize average per-step foreground phase overhead."
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
        default=Path("benchmark/images/checkpoint_phase.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Average Per-Step Foreground Phase Overhead",
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


def phase_avg(run: dict[str, Any], phase_key: str) -> float:
    raw_summary = run.get("raw_foreground_summary", {})
    data = raw_summary.get(phase_key)
    if isinstance(data, dict) and data.get("avg_sec") is not None:
        return float(data["avg_sec"])
    return 0.0


def collect_phase_times(
    report: dict[str, Any],
) -> list[tuple[str, dict[str, float]]]:
    by_hook: dict[str, dict[str, float]] = {}
    for run in get_runs(report):
        hook_type = run.get("hook_type")
        if not isinstance(hook_type, str):
            continue
        if run.get("returncode") != 0:
            continue

        times: dict[str, float] = {}
        for group_name, phase_key in PHASE_KEYS.items():
            times[group_name] = phase_avg(run, phase_key)
        by_hook[hook_type] = times

    ordered = [(hook, by_hook[hook]) for hook in HOOK_ORDER if hook in by_hook]
    if not ordered:
        raise ValueError("no phase timing data found in report")
    return ordered


def plot_phase_times(
    phase_times: list[tuple[str, dict[str, float]]],
    output_path: Path,
    title: str,
    report_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [HOOK_LABELS.get(hook, hook) for hook, _ in phase_times]
    n_hooks = len(labels)
    y_positions = list(range(n_hooks))

    fig, ax = plt.subplots(figsize=(10, 5.8))

    phase_order = ["forward", "backward", "update"]
    for phase_name in phase_order:
        widths = [times.get(phase_name, 0.0) for _, times in phase_times]
        lefts = []
        for _, times in phase_times:
            left = 0.0
            for prev_phase in phase_order:
                if prev_phase == phase_name:
                    break
                left += times.get(prev_phase, 0.0)
            lefts.append(left)

        ax.barh(
            y_positions,
            widths,
            left=lefts,
            height=0.55,
            color=PHASE_COLORS[phase_name],
            label=PHASE_LABELS[phase_name],
            zorder=2,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time (sec)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_max = max(
        sum(times.values()) for _, times in phase_times
    )
    if x_max > 0:
        ax.set_xlim(0, x_max * 1.15)

    ax.legend(frameon=False, loc="lower right")

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
    phase_times = collect_phase_times(report)
    plot_phase_times(phase_times, args.output, args.title, args.report)
    print(f"saved {len(phase_times)} phase-overhead bars to {args.output}")


if __name__ == "__main__":
    main()
