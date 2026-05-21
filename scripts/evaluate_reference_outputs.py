from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_codegen.dataset import load_alpaca_dataset, write_json  # noqa: E402
from quant_codegen.evaluator import evaluate_code_text, summarize_eval_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reference factor code outputs.")
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of rows to execute. Use 0 for all rows.",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Only run AST checks, without executing generated code.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_alpaca_dataset(ROOT / "data" / "quant_code.json")
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    results = [
        evaluate_code_text(row["output"], run_execution=not args.static_only)
        for row in rows
    ]
    summary = summarize_eval_results(results)
    summary["limit"] = args.limit
    summary["static_only"] = args.static_only

    output_path = ROOT / "results" / "reference_eval_metrics.json"
    write_json(output_path, summary)
    print(f"Wrote {output_path}")
    print(summary)


if __name__ == "__main__":
    main()
