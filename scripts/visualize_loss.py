#!/usr/bin/env python3
"""Plot training loss from a line-oriented log file.

Each non-empty line is expected to be either JSON or a Python dict literal, for
example:

    {'loss': '1.069', 'grad_norm': '25.38', 'learning_rate': '0', 'epoch': '1.932e-05'}
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def parse_record(line: str, line_number: int) -> dict[str, Any]:
    """Parse one log line as JSON first, then as a Python literal fallback."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        try:
            record = ast.literal_eval(line)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"line {line_number}: cannot parse record") from exc

    if not isinstance(record, dict):
        raise ValueError(f"line {line_number}: record is not a dict")
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
