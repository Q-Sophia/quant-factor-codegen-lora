from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_codegen.dataset import load_alpaca_dataset, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze token lengths with a model tokenizer.")
    parser.add_argument(
        "--model-name-or-path",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Hugging Face model id or local tokenizer directory.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "quant_code.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "token_length_metrics.json",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * pct))
    return ordered[index]


def stats(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "mean": round(sum(values) / len(values)),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": ordered[-1],
    }


def threshold_counts(values: list[int], thresholds: list[int]) -> list[dict[str, float | int]]:
    total = len(values)
    return [
        {
            "threshold": threshold,
            "over_count": sum(1 for value in values if value > threshold),
            "over_rate": round(sum(1 for value in values if value > threshold) / total, 4),
        }
        for threshold in thresholds
    ]


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    rows = load_alpaca_dataset(args.dataset)

    token_lengths: list[int] = []
    char_lengths: list[int] = []
    for row in rows:
        user = row["instruction"]
        if row["input"]:
            user = f"{user}\n{row['input']}"
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        token_lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
        char_lengths.append(len(text))

    report = {
        "model_name_or_path": args.model_name_or_path,
        "rows": len(rows),
        "token_length_stats": stats(token_lengths),
        "char_length_stats": stats(char_lengths),
        "token_thresholds": threshold_counts(token_lengths, [1024, 1536, 2048, 3072, 4096]),
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
