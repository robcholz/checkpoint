#!/usr/bin/env python3
"""Plot foreground checkpoint stall time across checkpoint algorithms.

The input must be a JSON report produced by benchmark/finetune_benchmark.py.

The plotted value is foreground time spent inside checkpoint hook phases. This
is the paper-style metric for visible checkpoint overhead; it intentionally
does not include final post-training drain time. For GoCkpt and GoCkpt-O, the
plot also shows a shaded same-color bar that includes final wrapping-up so the
catch-up cost remains visible without mixing it into the foreground metric.
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
WRAPUP_HOOKS = {"gockpt", "gockpt_o"}
FINAL_WRAPUP_PHASE = "hook.wait_for_pending_persistence"


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
        ]
    if hook_type == "async_o":
        return [
            "hook.save_checkpoint",
            "hook.update_begin",
        ]
    if hook_type in {"gockpt", "gockpt_o"}:
        return [
            "hook.save_checkpoint",
            "hook.forward_begin",
            "hook.backward_begin",
            "hook.backward_end",
            "hook.update_begin",
            "hook.update_end",
        ]
    return ["hook.save_checkpoint"]


def collect_foreground_times(
    report: dict[str, Any],
) -> list[tuple[str, float, float | None]]:
    by_hook: dict[str, tuple[float, float | None]] = {}
    for run in get_runs(report):
        hook_type = run.get("hook_type")
        if not isinstance(hook_type, str):
            continue

        total = 0.0
        for phase in foreground_phase_names(run):
            total += phase_total(run, phase) or 0.0

        total_with_final: float | None = None
        if hook_type in WRAPUP_HOOKS:
            total_with_final = total + (phase_total(run, FINAL_WRAPUP_PHASE) or 0.0)

        by_hook[hook_type] = (total, total_with_final)

    ordered = [(hook, by_hook[hook]) for hook in HOOK_ORDER if hook in by_hook]
    if not ordered:
        raise ValueError("no foreground checkpoint timing data found in report")
    return [(hook, values[0], values[1]) for hook, values in ordered]


def plot_foreground_times(
    foreground_times: list[tuple[str, float, float | None]],
    output_path: Path,
    title: str,
    report_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [HOOK_LABELS.get(hook, hook) for hook, _, _ in foreground_times]
    values = [value for _, value, _ in foreground_times]
    wrapup_values = [value for _, _, value in foreground_times]
    colors = [HOOK_COLORS.get(hook, "#455a64") for hook, _, _ in foreground_times]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x_positions = list(range(len(labels)))

    wrapup_bars = []
    for index, (hook, _, total_with_final) in enumerate(foreground_times):
        if total_with_final is None:
            continue
        wrapup_bars.append(
            ax.bar(
                index,
                total_with_final,
                color=HOOK_COLORS.get(hook, "#455a64"),
                alpha=0.28,
                width=0.74,
                label="Including final wrapping-up" if not wrapup_bars else None,
                zorder=1,
            )[0]
        )

    bars = ax.bar(
        x_positions,
        values,
        color=colors,
        width=0.54,
        label="Foreground stall only",
        zorder=2,
    )
    ax.set_xticks(x_positions, labels)

    ax.set_title(title)
    ax.set_ylabel("Foreground checkpoint stall time (sec)")
    ax.set_xlabel("Checkpoint algorithm")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_value = max(values + [value for value in wrapup_values if value is not None])
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

    for bar in wrapup_bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}s incl.",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )

    if wrapup_bars:
        ax.legend(frameon=False, loc="upper left")

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
    print(
        f"saved {len(foreground_times)} foreground checkpoint-time bars to {args.output}"
    )


if __name__ == "__main__":
    main()
