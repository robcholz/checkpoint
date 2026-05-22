#!/usr/bin/env python3
"""Plot host memory over time for each checkpoint algorithm.

Input must be benchmark host-memory data produced by
benchmark/finetune_benchmark.py at <output_dir>/host_memory.json.
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


MemorySeries = tuple[str, int | None, list[tuple[float, float]], list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize host memory usage over time across checkpoint algorithms."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmark/finetune_runs/host_memory.json"),
        help="Host memory report JSON path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("benchmark/images/checkpoint_memory.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Host Memory Usage Over Time",
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


def _memory_value_gb(sample: dict[str, Any]) -> float | None:
    process_tree_rss = sample.get("process_tree_rss_gb")
    if process_tree_rss is not None:
        return float(process_tree_rss)

    cgroup_current = sample.get("cgroup_memory_current_gb")
    if cgroup_current is not None:
        return float(cgroup_current)

    return None


def _oom_kill_events(sample: dict[str, Any]) -> int | None:
    value = sample.get("cgroup_memory_oom_kill_events")
    if value is None:
        return None
    return int(value)


def collect_memory_series(report: dict[str, Any]) -> list[MemorySeries]:
    by_hook: dict[str, MemorySeries] = {}
    for run in get_runs(report):
        hook_type = run.get("hook_type")
        if not isinstance(hook_type, str):
            continue

        raw_samples = run.get("host_memory_samples")
        if not isinstance(raw_samples, list):
            continue

        series: list[tuple[float, float]] = []
        oom_event_times: list[float] = []
        previous_oom_events: int | None = None
        for sample in raw_samples:
            if not isinstance(sample, dict):
                continue
            time_sec = sample.get("time_sec")
            memory_gb = _memory_value_gb(sample)
            if time_sec is None or memory_gb is None:
                continue

            sample_time = float(time_sec)
            series.append((sample_time, memory_gb))

            oom_events = _oom_kill_events(sample)
            if (
                oom_events is not None
                and previous_oom_events is not None
                and oom_events > previous_oom_events
            ):
                oom_event_times.append(sample_time)
            if oom_events is not None:
                previous_oom_events = oom_events

        if series:
            by_hook[hook_type] = (
                hook_type,
                run.get("returncode"),
                series,
                oom_event_times,
            )

    ordered = [by_hook[hook] for hook in HOOK_ORDER if hook in by_hook]
    if not ordered:
        raise ValueError("no runs with host memory samples were found in the report")
    return ordered


def plot_memory_series(
    series_by_hook: list[MemorySeries],
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
    for hook_type, returncode, series, oom_event_times in series_by_hook:
        xs = [point[0] for point in series]
        ys = [point[1] for point in series]
        if not xs:
            continue

        y_max = max(y_max, max(ys))
        x_max = max(x_max, max(xs))
        label = HOOK_LABELS.get(hook_type, hook_type)
        if returncode not in (0, None):
            label = f"{label} (rc={returncode})"

        color = HOOK_COLORS.get(hook_type, "#455a64")
        ax.plot(
            xs,
            ys,
            label=label,
            color=color,
            linewidth=1.8,
            linestyle="--" if returncode not in (0, None) else "-",
        )

        for event_time in oom_event_times:
            ax.axvline(event_time, color=color, alpha=0.18, linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Host memory usage (GiB)")
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
    memory_series = collect_memory_series(report)
    plot_memory_series(memory_series, args.output, args.title, args.report)
    print(f"saved {len(memory_series)} memory traces to {args.output}")


if __name__ == "__main__":
    main()
