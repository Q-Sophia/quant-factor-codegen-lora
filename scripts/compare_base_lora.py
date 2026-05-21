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
    parser = argparse.ArgumentParser(description="Compare base model and fine-tuned LoRA outputs.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("outputs/qwen3-4b-quant-lora"),
        help="Path to the trained LoRA adapter.",
    )
    parser.add_argument(
        "--eval-cases",
        type=Path,
        default=ROOT / "examples" / "eval_cases.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "base_vs_lora_outputs.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_eval_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


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

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
    )
    tuned_model = PeftModel.from_pretrained(base_model, args.adapter)
    tuned_model.eval()

    rows = []
    base_evals = []
    tuned_evals = []
    for case in load_eval_cases(args.eval_cases):
        title = case.get("title", case["id"])
        print("=" * 70)
        print(title)
        print("=" * 70)

        print("\n>>> 基座模型输出:")
        print("-" * 50)
        with tuned_model.disable_adapter():
            base_output = generate(
                tuned_model,
                tokenizer,
                case["prompt"],
                args.max_new_tokens,
                args.temperature,
            )
        print(base_output)

        print("\n>>> 微调模型输出:")
        print("-" * 50)
        tuned_output = generate(
            tuned_model,
            tokenizer,
            case["prompt"],
            args.max_new_tokens,
            args.temperature,
        )
        print(tuned_output)

        base_eval = evaluate_code_text(base_output)
        tuned_eval = evaluate_code_text(tuned_output)
        base_evals.append(base_eval)
        tuned_evals.append(tuned_eval)
        rows.append(
            {
                "id": case["id"],
                "title": title,
                "prompt": case["prompt"],
                "base_output": base_output,
                "tuned_output": tuned_output,
                "base_eval": base_eval.to_dict(),
                "tuned_eval": tuned_eval.to_dict(),
            }
        )

    payload = {
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "base_summary": summarize_eval_results(base_evals),
        "tuned_summary": summarize_eval_results(tuned_evals),
        "cases": rows,
    }
    write_json(args.output, payload)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
