from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_codegen.dataset import write_json  # noqa: E402
from quant_codegen.evaluator import (  # noqa: E402
    ContractEvalResult,
    FunctionalEvalResult,
    summarize_functional_results,
    summarize_reference_validity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge chunked held-out evaluation JSON files.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.inputs
    ]
    if not payloads:
        raise ValueError("No input result files provided")
    model_key = payloads[0].get("model_key")
    if not model_key:
        raise ValueError("Input result has no model_key")
    if any(payload.get("model_key") != model_key for payload in payloads):
        raise ValueError("All input result files must evaluate the same model")

    case_map: dict[str, dict] = {}
    for path, payload in zip(args.inputs, payloads):
        for row in payload.get("cases", []):
            case_id = row["id"]
            if case_id in case_map:
                raise ValueError(f"Duplicate case id {case_id} in {path}")
            case_map[case_id] = row

    rows = [case_map[key] for key in sorted(case_map)]
    reference_evals = [ContractEvalResult(**row["reference_eval"]) for row in rows]
    functional_evals = [FunctionalEvalResult(**row["functional_eval"]) for row in rows]

    first = payloads[0]
    merged = {
        "evaluation": "random_held_out_executable_functional_accuracy_merged",
        "model_key": model_key,
        "test_file": first.get("test_file"),
        "test_cases": len(rows),
        "source_files": [str(path) for path in args.inputs],
        "includes_private_case_details": any(
            bool(payload.get("includes_private_case_details")) for payload in payloads
        ),
        "metric_definition": first.get("metric_definition"),
        "models": first.get("models"),
        "summaries": {
            "dataset_audit": summarize_reference_validity(reference_evals),
            model_key: summarize_functional_results(functional_evals),
        },
        "cases": rows,
    }
    write_json(args.output, merged)
    print(json.dumps(merged["summaries"], ensure_ascii=False, indent=2))
    print(f"Saved merged evaluation to {args.output}")


if __name__ == "__main__":
    main()
