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
from quant_codegen.evaluator import evaluate_code_text, summarize_eval_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Qwen base, Qwen LoRA, and DeepSeek API on the same light prompts."
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--adapter",
        default="outputs/qwen3-4b-quant-lora",
        help="Local adapter directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--deepseek-model",
        default="deepseek-reasoner",
        help="DeepSeek API model name for the R1/reasoning baseline.",
    )
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com")
    parser.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--skip-deepseek", action="store_true")
    parser.add_argument(
        "--eval-cases",
        type=Path,
        default=ROOT / "examples" / "eval_cases.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "base_lora_deepseek_light.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--deepseek-max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--deepseek-reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument("--deepseek-thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def generate_local(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
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


def build_deepseek_client(args: argparse.Namespace):
    if args.skip_deepseek:
        return None
    api_key = os.environ.get(args.deepseek_api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Set {args.deepseek_api_key_env} first.")

    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=args.deepseek_base_url)


def generate_deepseek(
    client,
    args: argparse.Namespace,
    prompt: str,
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    messages = [{"role": "user", "content": prompt}]
    request_kwargs: dict[str, Any] = {
        "model": args.deepseek_model,
        "messages": messages,
        "max_tokens": args.deepseek_max_tokens,
        "stream": False,
        "extra_body": {"thinking": {"type": args.deepseek_thinking}},
    }
    if args.deepseek_thinking == "enabled":
        request_kwargs["reasoning_effort"] = args.deepseek_reasoning_effort
    else:
        request_kwargs["temperature"] = args.temperature

    response = client.chat.completions.create(**request_kwargs)
    choice = response.choices[0]
    message = choice.message
    output = message.content or ""
    usage = response.usage.model_dump() if response.usage else None
    reasoning_content = getattr(message, "reasoning_content", None)
    meta = {
        "finish_reason": choice.finish_reason,
        "content_chars": len(output),
        "reasoning_content_chars": len(reasoning_content or ""),
    }
    return output, usage, meta


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
    deepseek_client = build_deepseek_client(args)

    base_evals = []
    lora_evals = []
    deepseek_evals = []
    rows = []

    for case in load_eval_cases(args.eval_cases):
        title = case.get("title", case["id"])
        print("=" * 70)
        print(title)
        print("=" * 70)

        print("\n>>> Qwen base output:")
        print("-" * 50)
        with tuned_model.disable_adapter():
            base_output = generate_local(
                tuned_model,
                tokenizer,
                case["prompt"],
                args.max_new_tokens,
                args.temperature,
            )
        print(base_output)

        print("\n>>> Qwen LoRA output:")
        print("-" * 50)
        lora_output = generate_local(
            tuned_model,
            tokenizer,
            case["prompt"],
            args.max_new_tokens,
            args.temperature,
        )
        print(lora_output)

        deepseek_output = None
        deepseek_usage = None
        deepseek_meta = None
        deepseek_eval = None
        if deepseek_client is not None:
            print("\n>>> DeepSeek R1 API output:")
            print("-" * 50)
            deepseek_output, deepseek_usage, deepseek_meta = generate_deepseek(
                deepseek_client,
                args,
                case["prompt"],
            )
            print(deepseek_output)
            if not deepseek_output.strip():
                print(f"[warning] DeepSeek returned empty content: {deepseek_meta}")
            deepseek_eval = evaluate_code_text(deepseek_output)
            deepseek_evals.append(deepseek_eval)

        base_eval = evaluate_code_text(base_output)
        lora_eval = evaluate_code_text(lora_output)
        base_evals.append(base_eval)
        lora_evals.append(lora_eval)

        rows.append(
            {
                "id": case["id"],
                "title": title,
                "prompt": case["prompt"],
                "base_output": base_output,
                "lora_output": lora_output,
                "deepseek_output": deepseek_output,
                "base_eval": base_eval.to_dict(),
                "lora_eval": lora_eval.to_dict(),
                "deepseek_eval": deepseek_eval.to_dict() if deepseek_eval else None,
                "deepseek_usage": deepseek_usage,
                "deepseek_meta": deepseek_meta,
            }
        )

    summaries: dict[str, Any] = {
        "qwen_base": summarize_eval_results(base_evals),
        "qwen_lora": summarize_eval_results(lora_evals),
    }
    if deepseek_client is not None:
        summaries["deepseek_r1_api"] = summarize_eval_results(deepseek_evals)

    payload = {
        "eval_mode": "light_prompt",
        "models": {
            "qwen_base": args.base_model,
            "qwen_lora_adapter": str(args.adapter),
            "deepseek_r1_api": None if args.skip_deepseek else args.deepseek_model,
        },
        "deepseek_base_url": None if args.skip_deepseek else args.deepseek_base_url,
        "summaries": summaries,
        "cases": rows,
    }
    write_json(args.output, payload)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
