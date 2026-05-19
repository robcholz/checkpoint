#!/usr/bin/env python3
"""Plot GPU power over time for each checkpoint algorithm.

Input must be benchmark power data produced by benchmark/finetune_benchmark.py
at <output_dir>/power.json.
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
HOOK_COLORS = {
    "baseline": "#263238",
    "async": "#607d8b",
    "async_o": "#00897b",
    "gockpt": "#ef6c00",
    "gockpt_o": "#c62828",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GPU power draw over time across checkpoint algorithms."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmark/finetune_runs/power.json"),
        help="Power report JSON path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("benchmark/images/checkpoint_power.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="GPU Power Draw Over Time",
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


def collect_power_series(report: dict[str, Any]) -> list[tuple[str, list[tuple[float, float]]]]:
    by_hook: dict[str, list[tuple[float, float]]] = {}
    for run in get_runs(report):
        hook_type = run.get("hook_type")
        if not isinstance(hook_type, str):
            continue
        if run.get("returncode") != 0:
            continue

        raw_samples = run.get("power_samples")
        if not isinstance(raw_samples, list):
            continue

        series: list[tuple[float, float]] = []
        for sample in raw_samples:
            if not isinstance(sample, dict):
                continue
            time_sec = sample.get("time_sec")
            power_w = sample.get("power_w")
            if time_sec is None or power_w is None:
                continue
            series.append((float(time_sec), float(power_w)))

        if series:
            by_hook[hook_type] = series

    ordered = [(hook, by_hook[hook]) for hook in HOOK_ORDER if hook in by_hook]
    if not ordered:
        raise ValueError("no successful runs with power samples were found in the report")
    return ordered


def plot_power_series(
    series_by_hook: list[tuple[str, list[tuple[float, float]]]],
    output_path: Path,
    title: str,
    report_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))

    y_max = 0.0
    x_max = 0.0
    for hook_type, series in series_by_hook:
        xs = [point[0] for point in series]
        ys = [point[1] for point in series]
        if not xs:
            continue

        y_max = max(y_max, max(ys))
        x_max = max(x_max, max(xs))
        ax.plot(
            xs,
            ys,
            label=HOOK_LABELS.get(hook_type, hook_type),
            color=HOOK_COLORS.get(hook_type, "#455a64"),
            linewidth=1.8,
        )

    ax.set_title(title)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("GPU power draw (W)")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_max > 0:
        ax.set_ylim(0, y_max * 1.1)
    if x_max > 0:
        ax.set_xlim(0, x_max)
    ax.legend(frameon=False, loc="best")

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
    power_series = collect_power_series(report)
    plot_power_series(power_series, args.output, args.title, args.report)
    print(f"saved {len(power_series)} power traces to {args.output}")


if __name__ == "__main__":
    main()
