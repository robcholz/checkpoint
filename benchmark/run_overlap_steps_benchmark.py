#!/usr/bin/env python3
"""Run finetune benchmark sequentially across overlap-step values."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_TYPES = ("baseline", "async", "async_o", "gockpt", "gockpt_o")


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

    return values


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
        description="Run finetune benchmark + visualization for each overlap step."
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
    parser.add_argument ("--seq-len", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=20)
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
        default="Foreground Checkpoint Stall Time vs Algorithms",
        help="Base title passed to visualization script.",
    )
    parser.add_argument(
        "--conda-env",
        default="checkpoint",
        help="Conda environment name used for both commands.",
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
        "--output-dir",
        str(args.output_dir),
    ]
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    return command


def build_visualize_command(args: argparse.Namespace, output_image: Path) -> list[str]:
    report_path = args.output_dir / "report.json"
    return [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "benchmark/visualize_checkpoint_save_time.py",
        "--report",
        str(report_path),
        "--output",
        str(output_image),
        "--title",
        args.title,
    ]


def main() -> None:
    args = parse_args()
    images_dir = Path("benchmark/images") / args.images_folder
    images_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.overlap_steps)
    for index, overlap_step in enumerate(args.overlap_steps, start=1):
        print(f"[{index}/{total}] overlap-steps={overlap_step}: running benchmark")
        run_command(build_finetune_command(args, overlap_step))

        output_image = images_dir / f"checkpoint_time_overlap_steps={overlap_step}.png"
        print(
            f"[{index}/{total}] overlap-steps={overlap_step}: generating {output_image}"
        )
        run_command(build_visualize_command(args, output_image))

    print("completed overlap-step sweep in series")


if __name__ == "__main__":
    main()
