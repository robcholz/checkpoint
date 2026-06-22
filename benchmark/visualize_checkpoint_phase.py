#!/usr/bin/env python3
"""Plot checkpoint foreground stall time and background work.

Input must be a JSON report produced by benchmark/finetune_benchmark.py.

Each algorithm gets its own track on the y-axis with two sub-rows:
  - Top (stall): total blocking time spent inside checkpoint hook callbacks
  - Bottom (background): work done while training continues (reconstruction + persistence)

Note: Transfer time is NOT shown separately because:
  - For async/async_o: save_hook IS the transfer (blocking)
  - For gockpt/gockpt_o: forward_begin IS the transfer (blocking)
Showing both would double-count the same operation.
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

# Row 1: Foreground stall phases (what blocks training)
# For baseline/async: save_hook contains the blocking transfer
# For gockpt: forward_begin contains the blocking transfer
FOREGROUND_PHASES = {
    "save": "hook.save_checkpoint",
    "forward_begin": "hook.forward_begin",
    "backward_begin": "hook.backward_begin",
    "backward_end": "hook.backward_end",
    "update_begin": "hook.update_begin",
    "update_end": "hook.update_end",
    "final_drain": "hook.wait_for_pending_persistence",
}
FOREGROUND_COLORS = {
    "save": "#263238",           # Dark gray - save/transfer for baseline/async
    "forward_begin": "#90caf9",  # Light blue - transfer for gockpt
    "backward_begin": "#fb8c00",
    "backward_end": "#e53935",
    "update_begin": "#ce93d8",   # Purple - gradient work
    "update_end": "#7b1fa2",
    "final_drain": "#757575",    # Gray - drain/wait
}
FOREGROUND_LABELS = {
    "save": "Save/Transfer (blocking)",
    "forward_begin": "Transfer (forward hook)",
    "backward_begin": "Backward begin",
    "backward_end": "Backward end",
    "update_begin": "Gradient stall (update begin)",
    "update_end": "Update end",
    "final_drain": "Final drain (wait)",
}

# Row 2: Background work (overlapped with training)
BACKGROUND_COLORS = {
    "reconstruction": "#ffcc80",  # Orange
    "persistence": "#a5d6a7",     # Green
    "backpressure": "#bcaaa4",    # Brown
}
BACKGROUND_LABELS = {
    "reconstruction": "Reconstruction",
    "persistence": "Persistence",
    "backpressure": "Backpressure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize checkpoint foreground stall and background work."
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
        default="Checkpoint Stall Time (blocking) vs Background Work (overlapped)",
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


def numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def phase_total(run: dict[str, Any], phase_key: str) -> float:
    phase_summary = run.get("phase_summary", {})
    data = phase_summary.get(phase_key)
    if isinstance(data, dict):
        return numeric(data.get("total_sec"))
    return 0.0


def collect_foreground_phases(run: dict[str, Any]) -> dict[str, float]:
    return {
        group_name: phase_total(run, phase_key)
        for group_name, phase_key in FOREGROUND_PHASES.items()
    }


def collect_background_work(run: dict[str, Any]) -> dict[str, float]:
    """Collect background work: reconstruction + persistence.

    Note: We do NOT include transfer here because transfer is already
    counted in the foreground hooks (save_hook for async, forward_begin for gockpt).
    """
    totals = {key: 0.0 for key in BACKGROUND_LABELS}
    raw_results = run.get("checkpoint_results", [])
    if not isinstance(raw_results, list):
        return totals

    for result in raw_results:
        if not isinstance(result, dict):
            continue

        totals["reconstruction"] += numeric(result.get("reconstruction_duration_sec"))
        totals["backpressure"] += numeric(
            result.get("reconstruction_backpressure_sec")
        )
        totals["persistence"] += numeric(result.get("persistence_duration_sec"))

    return totals


def collect_data(
    report: dict[str, Any],
) -> list[tuple[str, dict[str, float], dict[str, float]]]:
    """Returns (hook_type, foreground_stall, background_work) for each algorithm."""
    by_hook: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for run in get_runs(report):
        hook_type = run.get("hook_type")
        if not isinstance(hook_type, str):
            continue
        if run.get("returncode") != 0:
            continue

        by_hook[hook_type] = (
            collect_foreground_phases(run),
            collect_background_work(run),
        )

    ordered = [
        (hook, by_hook[hook][0], by_hook[hook][1])
        for hook in HOOK_ORDER
        if hook in by_hook
    ]
    if not ordered:
        raise ValueError("no checkpoint phase timing data found in report")
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

    fig, ax = plt.subplots(figsize=(12, 1.5 + n_hooks * row_spacing))

    y_tick_positions = []
    y_tick_labels = []

    foreground_order = list(FOREGROUND_LABELS)
    background_order = list(BACKGROUND_LABELS)

    legend_handles: dict[str, Any] = {}

    for idx, (hook_type, foreground, background) in enumerate(data):
        y_row1 = idx * row_spacing  # Foreground stall
        y_row2 = y_row1 + bar_height + 0.05  # Background work

        y_tick_positions.append(y_row1 + bar_height / 2 + 0.025)
        y_tick_labels.append(HOOK_LABELS.get(hook_type, hook_type))
        ax.text(
            -0.01,
            y_row1,
            "stall",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7,
            color="#555555",
        )
        ax.text(
            -0.01,
            y_row2,
            "background",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7,
            color="#555555",
        )

        # Row 1: Foreground stall (blocking hooks)
        left = 0.0
        for phase_name in foreground_order:
            width = foreground.get(phase_name, 0.0)
            if width < 0.001:  # Skip negligible values
                continue
            bar = ax.barh(
                y_row1,
                width,
                left=left,
                height=bar_height,
                color=FOREGROUND_COLORS[phase_name],
                zorder=2,
            )
            if phase_name not in legend_handles:
                legend_handles[phase_name] = bar[0]
            left += width

        # Row 2: Background work (overlapped)
        left = 0.0
        for work_key in background_order:
            width = background.get(work_key, 0.0)
            if width < 0.001:  # Skip negligible values
                continue
            bar = ax.barh(
                y_row2,
                width,
                left=left,
                height=bar_height,
                color=BACKGROUND_COLORS[work_key],
                zorder=2,
            )
            if work_key not in legend_handles:
                legend_handles[work_key] = bar[0]
            left += width

    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Total time over run (sec)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    all_widths = []
    for _, foreground, background in data:
        all_widths.append(sum(foreground.values()))
        all_widths.append(sum(background.values()))
    x_max = max(all_widths) if all_widths else 0.0
    if x_max > 0:
        ax.set_xlim(0, x_max * 1.15)

    all_labels = (
        list(FOREGROUND_LABELS.values())
        + list(BACKGROUND_LABELS.values())
    )
    all_keys = (
        list(FOREGROUND_LABELS.keys())
        + list(BACKGROUND_LABELS.keys())
    )
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
