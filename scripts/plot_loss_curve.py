from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot train/eval loss from Trainer logs.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to log_history.json, trainer_state.json, or a raw terminal log captured by tee.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/loss_curve.png"),
        help="Output PNG path.",
    )
    return parser


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict) and isinstance(data.get("log_history"), list):
            return [item for item in data["log_history"] if isinstance(item, dict)]
        raise ValueError(f"Unsupported JSON log format: {path}")

    records: list[dict[str, Any]] = []
    for match in re.finditer(r"\{[^{}]+\}", text):
        try:
            item = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def pick_x_axis(records: list[dict[str, Any]]) -> str:
    if any("step" in item for item in records):
        return "step"
    if any("epoch" in item for item in records):
        return "epoch"
    return "index"


def point_x(item: dict[str, Any], x_axis: str, index: int) -> float:
    if x_axis == "index":
        return float(index)
    value = item.get(x_axis)
    if isinstance(value, (int, float)):
        return float(value)
    return float(index)


def main() -> None:
    args = build_parser().parse_args()
    records = load_records(args.input)
    if not records:
        raise ValueError(f"No log records found in {args.input}")

    x_axis = pick_x_axis(records)
    train_points: list[tuple[float, float]] = []
    eval_points: list[tuple[float, float]] = []

    for index, item in enumerate(records):
        x_value = point_x(item, x_axis, index)
        if isinstance(item.get("loss"), (int, float)):
            train_points.append((x_value, float(item["loss"])))
        if isinstance(item.get("eval_loss"), (int, float)):
            eval_points.append((x_value, float(item["eval_loss"])))

    if not train_points and not eval_points:
        raise ValueError(f"No loss/eval_loss values found in {args.input}")

    import matplotlib.pyplot as plt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    if train_points:
        xs, ys = zip(*train_points)
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label="train loss")
    if eval_points:
        xs, ys = zip(*eval_points)
        plt.plot(xs, ys, marker="s", markersize=3, linewidth=1.6, label="eval loss")
    plt.xlabel(x_axis)
    plt.ylabel("loss")
    plt.title("LoRA SFT Loss Curve")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved loss curve to {args.output}")


if __name__ == "__main__":
    main()
