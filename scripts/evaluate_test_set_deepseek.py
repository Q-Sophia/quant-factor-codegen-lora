from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_codegen.dataset import write_json  # noqa: E402
from quant_codegen.evaluator import (  # noqa: E402
    evaluate_against_reference,
    evaluate_contract,
    summarize_functional_results,
    summarize_reference_validity,
)
from quant_codegen.mock_data import make_evaluation_panels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a DeepSeek API model against held-out reference factor programs."
    )
    parser.add_argument("--model", default="deepseek-reasoner")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--include-details", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def load_test_cases(path: Path, start_index: int, limit: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if not line.strip() or index < start_index:
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Row {index} does not contain user/assistant messages")
            reference = messages[-1]
            if reference.get("role") != "assistant":
                raise ValueError(f"Row {index} has no final assistant reference answer")
            cases.append(
                {
                    "id": f"test_{index:04d}",
                    "prompt_messages": messages[:-1],
                    "reference_output": str(reference.get("content", "")),
                }
            )
            if limit > 0 and len(cases) >= limit:
                break
    return cases


def generate(client, args: argparse.Namespace, messages: list[dict[str, str]]) -> tuple[str, dict]:
    request_kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "stream": False,
        "extra_body": {"thinking": {"type": args.thinking}},
    }
    if args.thinking == "enabled":
        request_kwargs["reasoning_effort"] = args.reasoning_effort
    else:
        request_kwargs["temperature"] = args.temperature

    response = client.chat.completions.create(**request_kwargs)
    choice = response.choices[0]
    message = choice.message
    output = message.content or ""
    reasoning_content = getattr(message, "reasoning_content", None)
    metadata = {
        "finish_reason": choice.finish_reason,
        "content_chars": len(output),
        "reasoning_content_chars": len(reasoning_content or ""),
        "usage": response.usage.model_dump() if response.usage else None,
    }
    return output, metadata


def main() -> None:
    args = parse_args()
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")

    cases = load_test_cases(args.test_file, args.start_index, args.limit)
    if not cases:
        raise ValueError(f"No test cases found in {args.test_file}")
    panels = make_evaluation_panels()

    if args.audit_only:
        reference_evals = [
            evaluate_contract(case["reference_output"], panels=panels) for case in cases
        ]
        payload = {
            "evaluation": "random_held_out_reference_audit",
            "test_file": str(args.test_file),
            "test_cases": len(cases),
            "panel_count": len(panels),
            "summary": summarize_reference_validity(reference_evals),
        }
        write_json(args.output, payload)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Set {args.api_key_env} first.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    reference_evals = []
    functional_evals = []
    output_rows = []
    for number, case in enumerate(cases, start=1):
        output, metadata = generate(client, args, case["prompt_messages"])
        reference_eval, functional_eval = evaluate_against_reference(
            output,
            case["reference_output"],
            panels=panels,
            rtol=args.rtol,
            atol=args.atol,
        )
        reference_evals.append(reference_eval)
        functional_evals.append(functional_eval)
        row = {
            "id": case["id"],
            "reference_eval": reference_eval.to_dict(),
            "functional_eval": functional_eval.to_dict(),
            "generation_metadata": metadata,
        }
        if args.include_details:
            row.update(
                {
                    "prompt_messages": case["prompt_messages"],
                    "reference_output": case["reference_output"],
                    "output": output,
                }
            )
        output_rows.append(row)
        if number % args.log_every == 0 or number == len(cases):
            print(f"Evaluated {number}/{len(cases)} cases")

    payload = {
        "evaluation": "random_held_out_executable_functional_accuracy",
        "model_key": "deepseek_api",
        "test_file": str(args.test_file),
        "start_index": args.start_index,
        "test_cases": len(cases),
        "includes_private_case_details": args.include_details,
        "metric_definition": {
            "reference_valid_rate": (
                "Held-out reference programs that execute successfully on all deterministic "
                "mock market panels."
            ),
            "executable_functional_accuracy": (
                "Generated programs that expose any callable factor function compatible with "
                "the long-form market DataFrame and whose numeric outputs match the valid "
                "reference outputs on every panel; denominator is reference-valid test cases."
            ),
            "rtol": args.rtol,
            "atol": args.atol,
            "panel_count": len(panels),
        },
        "models": {
            "deepseek_api": {
                "model": args.model,
                "base_url": args.base_url,
                "thinking": args.thinking,
                "reasoning_effort": args.reasoning_effort,
            }
        },
        "summaries": {
            "dataset_audit": summarize_reference_validity(reference_evals),
            "deepseek_api": summarize_functional_results(functional_evals),
        },
        "cases": output_rows,
    }
    write_json(args.output, payload)
    print(json.dumps(payload["summaries"], ensure_ascii=False, indent=2))
    print(f"Saved test evaluation to {args.output}")


if __name__ == "__main__":
    main()
