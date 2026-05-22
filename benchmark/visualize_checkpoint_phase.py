#!/usr/bin/env python3
"""Plot average per-step foreground phase times and M+O / gradient transfer times.

Input must be a JSON report produced by benchmark/finetune_benchmark.py.

Each algorithm gets its own track on the y-axis with two sub-rows:
  - Top: raw foreground computation (forward=yellow, backward=orange, update=red)
  - Bottom: transfer timing (M+O foreground, M+O full, gradient foreground, gradient full)
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

TRANSFER_COLORS = {
    "mo_foreground": "#1565c0",
    "mo_full": "#90caf9",
    "gradient_foreground": "#6a1b9a",
    "gradient_full": "#ce93d8",
}
TRANSFER_LABELS = {
    "mo_foreground": "M+O foreground",
    "mo_full": "M+O full",
    "gradient_foreground": "Gradient foreground",
    "gradient_full": "Gradient full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize average per-step foreground phase overhead and transfer times."
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
        default="Average Per-Step Phase & Transfer Overhead",
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


def collect_data(
    report: dict[str, Any],
) -> list[tuple[str, dict[str, float], dict[str, float]]]:
    """Returns (hook_type, phase_times, transfer_times) for each algorithm."""
    by_hook: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for run in get_runs(report):
        hook_type = run.get("hook_type")
        if not isinstance(hook_type, str):
            continue
        if run.get("returncode") != 0:
            continue

        phases: dict[str, float] = {}
        for group_name, phase_key in PHASE_KEYS.items():
            phases[group_name] = phase_avg(run, phase_key)

        tts = run.get("transfer_timing_summary", {})
        transfers: dict[str, float] = {
            "mo_foreground": float(tts.get("mo_foreground_avg_sec") or 0.0),
            "mo_full": float(tts.get("mo_full_avg_sec") or 0.0),
            "gradient_foreground": float(tts.get("gradient_foreground_avg_sec") or 0.0),
            "gradient_full": float(tts.get("gradient_full_avg_sec") or 0.0),
        }
        by_hook[hook_type] = (phases, transfers)

    ordered = [
        (hook, by_hook[hook][0], by_hook[hook][1])
        for hook in HOOK_ORDER
        if hook in by_hook
    ]
    if not ordered:
        raise ValueError("no phase timing data found in report")
    return ordered


def plot_phase_times(
    data: list[tuple[str, dict[str, float], dict[str, float]]],
    output_path: Path,
    title: str,
    report_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_hooks = len(data)
    bar_height = 0.35
    row_spacing = 1.2

    fig, ax = plt.subplots(figsize=(10, 1.5 + n_hooks * row_spacing))

    y_tick_positions = []
    y_tick_labels = []

    phase_order = ["forward", "backward", "update"]
    transfer_order = ["mo_foreground", "mo_full", "gradient_foreground", "gradient_full"]

    legend_handles: dict[str, Any] = {}

    for idx, (hook_type, phases, transfers) in enumerate(data):
        y_top = idx * row_spacing
        y_bottom = y_top + bar_height + 0.05

        y_tick_positions.append(y_top + bar_height / 2 + 0.025)
        y_tick_labels.append(HOOK_LABELS.get(hook_type, hook_type))

        left = 0.0
        for phase_name in phase_order:
            width = phases.get(phase_name, 0.0)
            bar = ax.barh(
                y_top,
                width,
                left=left,
                height=bar_height,
                color=PHASE_COLORS[phase_name],
                zorder=2,
            )
            if phase_name not in legend_handles:
                legend_handles[phase_name] = bar[0]
            left += width

        left = 0.0
        for t_key in transfer_order:
            width = transfers.get(t_key, 0.0)
            bar = ax.barh(
                y_bottom,
                width,
                left=left,
                height=bar_height,
                color=TRANSFER_COLORS[t_key],
                zorder=2,
            )
            if t_key not in legend_handles:
                legend_handles[t_key] = bar[0]
            left += width

    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time (sec)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    all_widths = []
    for _, phases, transfers in data:
        all_widths.append(sum(phases.values()))
        all_widths.append(sum(transfers.values()))
    x_max = max(all_widths) if all_widths else 0.0
    if x_max > 0:
        ax.set_xlim(0, x_max * 1.15)

    all_labels = list(PHASE_LABELS.values()) + list(TRANSFER_LABELS.values())
    all_keys = list(PHASE_LABELS.keys()) + list(TRANSFER_LABELS.keys())
    handles = [legend_handles[k] for k in all_keys if k in legend_handles]
    labels = [
        lbl for k, lbl in zip(all_keys, all_labels) if k in legend_handles
    ]
    ax.legend(handles, labels, frameon=False, loc="lower right", fontsize=8)

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
    data = collect_data(report)
    plot_phase_times(data, args.output, args.title, args.report)
    print(f"saved {len(data)} phase-overhead bars to {args.output}")


if __name__ == "__main__":
    main()
