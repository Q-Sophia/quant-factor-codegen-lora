from __future__ import annotations

import argparse
import json
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
        description="Evaluate a Qwen LoRA adapter against held-out reference factor programs."
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--adapter",
        default=None,
        help="Local adapter directory or Hugging Face repo id; required unless --audit-only is used.",
    )
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts generated together. Try 2 or 4 on an RTX 4090.",
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate held-out reference programs without loading or running a language model.",
    )
    parser.add_argument(
        "--include-details",
        action="store_true",
        help="Include prompts, references, and generated code in output; keep such files private.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate N cases after --start-index; use 0 for all remaining cases.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip cases before this zero-based row index, useful for chunked evaluation.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_test_cases(path: Path, start_index: int, limit: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if not line.strip():
                continue
            if index < start_index:
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


def generate_batch(
    model,
    tokenizer,
    conversations: list[list[dict[str, str]]],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    import torch

    prompt_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in conversations
    ]
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
    with torch.no_grad():
        output_ids = model.generate(**inputs, **kwargs)
    prompt_width = inputs["input_ids"].shape[-1]
    return [
        tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
        for row in output_ids
    ]


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
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
            "cases": [
                {"id": case["id"], "reference_eval": result.to_dict()}
                for case, result in zip(cases, reference_evals)
            ],
        }
        write_json(args.output, payload)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        print(f"Saved reference audit to {args.output}")
        return

    if not args.adapter:
        raise ValueError("--adapter is required unless --audit-only is used")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
    )
    tuned_model = PeftModel.from_pretrained(base_model, args.adapter)
    tuned_model.eval()

    reference_evals = []
    lora_functional_evals = []
    output_rows = []

    for batch_start in range(0, len(cases), args.batch_size):
        batch = cases[batch_start : batch_start + args.batch_size]
        conversations = [case["prompt_messages"] for case in batch]
        lora_outputs = generate_batch(
            tuned_model,
            tokenizer,
            conversations,
            args.max_new_tokens,
            args.temperature,
        )

        for case, lora_output in zip(batch, lora_outputs):
            reference_eval, lora_functional = evaluate_against_reference(
                lora_output,
                case["reference_output"],
                panels=panels,
                rtol=args.rtol,
                atol=args.atol,
            )

            reference_evals.append(reference_eval)
            lora_functional_evals.append(lora_functional)
            row = {
                "id": case["id"],
                "reference_eval": reference_eval.to_dict(),
                "functional_eval": lora_functional.to_dict(),
            }
            if args.include_details:
                row.update(
                    {
                        "prompt_messages": case["prompt_messages"],
                        "reference_output": case["reference_output"],
                        "output": lora_output,
                    }
                )
            output_rows.append(row)

        number = min(batch_start + len(batch), len(cases))
        if number % args.log_every == 0 or number == len(cases):
            print(f"Evaluated {number}/{len(cases)} cases")

    payload = {
        "evaluation": "random_held_out_executable_functional_accuracy",
        "model_key": "qwen_lora",
        "test_file": str(args.test_file),
        "start_index": args.start_index,
        "test_cases": len(cases),
        "includes_private_case_details": args.include_details,
        "metric_definition": {
            "reference_valid_rate": (
                "Held-out reference programs that satisfy the factor(df) execution contract "
                "on all deterministic mock market panels."
            ),
            "executable_functional_accuracy": (
                "Generated programs that expose any callable factor function compatible with "
                "the long-form market DataFrame and whose numeric outputs match the valid "
                "reference outputs on every panel; denominator is reference-valid test cases."
            ),
            "rtol": args.rtol,
            "atol": args.atol,
            "panel_count": len(panels),
            "generation_batch_size": args.batch_size,
        },
        "models": {
            "qwen_lora": {
                "base_model": args.base_model,
                "adapter": args.adapter,
            },
        },
        "summaries": {
            "dataset_audit": summarize_reference_validity(reference_evals),
            "qwen_lora": summarize_functional_results(lora_functional_evals),
        },
        "cases": output_rows,
    }
    write_json(args.output, payload)
    print(json.dumps(payload["summaries"], ensure_ascii=False, indent=2))
    print(f"Saved test evaluation to {args.output}")


if __name__ == "__main__":
    main()
