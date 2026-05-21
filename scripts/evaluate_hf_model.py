from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_codegen.dataset import write_json  # noqa: E402
from quant_codegen.evaluator import evaluate_code_text, summarize_eval_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one Hugging Face causal LM on factor-code cases.")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local model directory.")
    parser.add_argument(
        "--eval-cases",
        type=Path,
        default=ROOT / "examples" / "eval_cases.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "hf_model_eval.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_eval_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def resolve_dtype(value: str):
    import torch

    if value == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[value]


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    rows = []
    evals = []
    for case in load_eval_cases(args.eval_cases):
        title = case.get("title", case["id"])
        print("=" * 70)
        print(title)
        print("=" * 70)
        output = generate(
            model,
            tokenizer,
            case["prompt"],
            args.max_new_tokens,
            args.temperature,
        )
        print(output)
        result = evaluate_code_text(output)
        evals.append(result)
        rows.append(
            {
                "id": case["id"],
                "title": title,
                "prompt": case["prompt"],
                "output": output,
                "eval": result.to_dict(),
            }
        )

    payload = {
        "model": args.model,
        "summary": summarize_eval_results(evals),
        "cases": rows,
    }
    write_json(args.output, payload)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
