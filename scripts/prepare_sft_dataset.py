from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_codegen.dataset import load_alpaca_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert quant_code.json into local SFT train/validation/test JSONL files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "quant_code.json",
        help="Path to the Alpaca-style source dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Directory for train.jsonl, val.jsonl, and test.jsonl.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_messages(row: dict[str, str]) -> dict[str, object]:
    user = row["instruction"]
    if row["input"]:
        user = f"{user}\n{row['input']}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["output"]},
        ]
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0:
        raise ValueError("val-ratio and test-ratio must be non-negative")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("val-ratio + test-ratio must be less than 1")

    rows = [
        row
        for row in load_alpaca_dataset(args.input)
        if row["instruction"] and row["output"]
    ]

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    total_size = len(rows)
    train_size = int(total_size * (1 - args.val_ratio - args.test_ratio))
    val_size = int(total_size * args.val_ratio)
    train_rows = rows[:train_size]
    val_rows = rows[train_size : train_size + val_size]
    test_rows = rows[train_size + val_size :]

    write_jsonl(args.output_dir / "train.jsonl", [to_messages(row) for row in train_rows])
    write_jsonl(args.output_dir / "val.jsonl", [to_messages(row) for row in val_rows])
    write_jsonl(args.output_dir / "test.jsonl", [to_messages(row) for row in test_rows])

    metadata = {
        "source": str(args.input),
        "total_size": total_size,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "train_ratio": round(len(train_rows) / len(rows), 6) if rows else 0,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "format": "jsonl with messages: user/assistant",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {args.output_dir / 'train.jsonl'} ({len(train_rows)} rows)")
    print(f"Wrote {args.output_dir / 'val.jsonl'} ({len(val_rows)} rows)")
    print(f"Wrote {args.output_dir / 'test.jsonl'} ({len(test_rows)} rows)")
    print(f"Wrote {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
