#!/usr/bin/env python3
"""Plot training loss from a line-oriented log file.

Each non-empty line is expected to contain whitespace-separated key=value pairs, for
example:

    step=198/200 loss=0.8669 lr=0.00000001 steps_per_sec=5.52
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_record(line: str, line_number: int) -> dict[str, Any]:
    """Parse one whitespace-delimited key=value log line."""
    record: dict[str, Any] = {}
    for field in line.split():
        if "=" not in field:
            raise ValueError(f"line {line_number}: malformed field: {field!r}")
        key, value = field.split("=", 1)
        if not key:
            raise ValueError(f"line {line_number}: empty key in field: {field!r}")
        record[key] = value

    return record


def read_losses(log_path: Path) -> list[float]:
    losses: list[float] = []

    with log_path.open("r", encoding="utf-8") as log_file:
        for line_number, raw_line in enumerate(log_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            record = parse_record(line, line_number)
            if "loss" not in record:
                continue

            try:
                losses.append(float(record["loss"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"line {line_number}: loss is not numeric: {record['loss']!r}"
                ) from exc

    if not losses:
        raise ValueError(f"no loss values found in {log_path}")
    return losses


def plot_losses(losses: list[float], output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = range(1, len(losses) + 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.5))
    plt.plot(steps, losses, color="#1f77b4", linewidth=1.8)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize loss values from a one-record-per-line training log."
    )
    parser.add_argument("log_file", type=Path, help="Path to the input log file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("benchmark/images/loss.png"),
        help="Path for the saved plot. Defaults to benchmark/images/loss.png.",
    )
    parser.add_argument(
        "--title",
        default="Training Loss",
        help="Plot title. Defaults to 'Training Loss'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    losses = read_losses(args.log_file)
    plot_losses(losses, args.output, args.title)
    print(f"saved {len(losses)} loss points to {args.output}")


if __name__ == "__main__":
    main()
