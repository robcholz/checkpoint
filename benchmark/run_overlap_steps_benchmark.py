#!/usr/bin/env python3
"""Run finetune benchmark sequentially across overlap-step values.

Generates a single line chart where:
- x-axis: checkpoint algorithms
- y-axis: foreground checkpoint stall time
- one line per overlap-step setting (different colors)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_TYPES = ("baseline", "async", "async_o", "gockpt", "gockpt_o")
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


def parse_overlap_steps(raw_values: list[str]) -> list[int]:
    values: list[int] = []
    for raw in raw_values:
        for item in raw.split(","):
            token = item.strip()
            if not token:
                continue
            try:
                value = int(token)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"invalid overlap step '{token}' (expected integer)"
                ) from error
            if value <= 0:
                raise argparse.ArgumentTypeError(
                    f"invalid overlap step '{value}' (must be > 0)"
                )
            values.append(value)

    if not values:
        raise argparse.ArgumentTypeError("no overlap steps provided")

    return list(dict.fromkeys(values))


def parse_images_folder(raw_value: str) -> str:
    folder = raw_value.strip()
    if not folder:
        raise argparse.ArgumentTypeError("images folder name cannot be empty")
    if "/" in folder or "\\" in folder:
        raise argparse.ArgumentTypeError(
            "images folder name must be a single folder name (no path separators)"
        )
    return folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finetune benchmark for each overlap step and plot one line graph."
    )
    parser.add_argument(
        "--overlap-steps",
        nargs="+",
        required=True,
        help="Overlap steps list, e.g. 7,8,9,10 or 7 8 9 10.",
    )
    parser.add_argument(
        "--hook-types",
        nargs="+",
        default=list(HOOK_TYPES),
        choices=HOOK_TYPES,
        help="Hook implementations to benchmark.",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument(
        "--gockpt-transfer-chunk-mb",
        type=float,
        default=64.0,
        help=(
            "GoCkpt GPU-to-CPU transfer chunk size in MiB forwarded to "
            "finetune_benchmark.py. Use the measured local best by default."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing (enabled by default).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/overlap_steps_runs"),
        help="Output directory passed to finetune benchmark.",
    )
    parser.add_argument(
        "--images-folder",
        type=parse_images_folder,
        required=True,
        help="Output folder name under benchmark/images for overlap-step plots.",
    )
    parser.add_argument(
        "--title",
        default="Foreground Checkpoint Stall Time vs Algorithms (lines: overlap steps)",
        help="Line chart title.",
    )
    parser.add_argument(
        "--output-image",
        default="checkpoint_time_overlap_steps.png",
        help="Output filename for the combined line chart.",
    )
    parser.add_argument(
        "--conda-env",
        default="checkpoint",
        help="Conda environment name used for benchmark runs.",
    )
    args = parser.parse_args()
    args.overlap_steps = parse_overlap_steps(args.overlap_steps)
    return args


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def build_finetune_command(args: argparse.Namespace, overlap_step: int) -> list[str]:
    command = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "benchmark/finetune_benchmark.py",
        "--hook-types",
        *args.hook_types,
        "--seq-len",
        str(args.seq_len),
        "--max-steps",
        str(args.max_steps),
        "--save-steps",
        str(args.save_steps),
        "--overlap-steps",
        str(overlap_step),
        "--gockpt-transfer-chunk-mb",
        str(args.gockpt_transfer_chunk_mb),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    return command


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


def collect_foreground_times(report: dict[str, Any]) -> dict[str, float]:
    by_hook: dict[str, float] = {}
    for run in get_runs(report):
        hook_type = run.get("hook_type")
        if not isinstance(hook_type, str):
            continue
        total = 0.0
        for phase in foreground_phase_names(run):
            total += phase_total(run, phase) or 0.0
        by_hook[hook_type] = total
    if not by_hook:
        raise ValueError("no foreground checkpoint timing data found in report")
    return by_hook


def plot_overlap_sweep(
    overlap_steps: list[int],
    times_by_step: dict[int, dict[str, float]],
    hooks_in_order: list[str],
    output_path: Path,
    title: str,
    source_report_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))

    x_positions = list(range(len(hooks_in_order)))
    x_labels = [HOOK_LABELS.get(hook, hook) for hook in hooks_in_order]
    color_map = plt.get_cmap("tab10")
    max_value = 0.0
    for color_idx, overlap_step in enumerate(overlap_steps):
        per_hook = times_by_step.get(overlap_step, {})
        xs: list[int] = []
        ys: list[float] = []
        for hook_idx, hook in enumerate(hooks_in_order):
            value = per_hook.get(hook)
            if value is None:
                continue
            xs.append(hook_idx)
            ys.append(value)

        if not ys:
            continue
        max_value = max(max_value, max(ys))
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=color_map(color_idx % 10),
            label=f"overlap={overlap_step}",
        )

    ax.set_title(title)
    ax.set_xlabel("Checkpoint algorithm")
    ax.set_ylabel("Foreground checkpoint stall time (sec)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if max_value > 0:
        ax.set_ylim(0, max_value * 1.16)
    ax.legend(frameon=False)

    fig.text(
        0.01,
        0.01,
        f"source: {source_report_path}",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    images_dir = Path("benchmark/images") / args.images_folder
    images_dir.mkdir(parents=True, exist_ok=True)
    output_image = images_dir / args.output_image

    report_path = args.output_dir / "report.json"
    times_by_step: dict[int, dict[str, float]] = {}

    total = len(args.overlap_steps)
    for index, overlap_step in enumerate(args.overlap_steps, start=1):
        print(f"[{index}/{total}] overlap-steps={overlap_step}: running benchmark")
        run_command(build_finetune_command(args, overlap_step))
        report = read_report(report_path)
        foreground_times = collect_foreground_times(report)
        times_by_step[overlap_step] = {}
        for hook in args.hook_types:
            value = foreground_times.get(hook)
            if value is not None:
                times_by_step[overlap_step][hook] = value

    print(f"generating combined line chart: {output_image}")
    plot_overlap_sweep(
        overlap_steps=args.overlap_steps,
        times_by_step=times_by_step,
        hooks_in_order=args.hook_types,
        output_path=output_image,
        title=args.title,
        source_report_path=report_path,
    )

    print("completed overlap-step sweep in series")
    print(f"saved line chart to {output_image}")


if __name__ == "__main__":
    main()
