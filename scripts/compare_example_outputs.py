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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate example outputs from Qwen base, Qwen LoRA, and optional "
            "DeepSeek-V4 API models. This script is for qualitative comparison only."
        )
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--adapter",
        required=True,
        help="Local LoRA adapter directory or Hugging Face repo id.",
    )
    parser.add_argument("--eval-cases", type=Path, default=ROOT / "examples" / "eval_cases.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "example_outputs_base_lora_deepseek_v4.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-deepseek", action="store_true")
    parser.add_argument("--deepseek-model", default="deepseek-reasoner")
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com")
    parser.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--deepseek-max-tokens", type=int, default=4096)
    parser.add_argument("--deepseek-thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--deepseek-reasoning-effort", choices=["high", "max"], default="high")
    return parser.parse_args()


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def generate_local(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    import torch

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **kwargs)
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


def generate_deepseek(client, args: argparse.Namespace, prompt: str) -> tuple[str, dict[str, Any]]:
    request_kwargs: dict[str, Any] = {
        "model": args.deepseek_model,
        "messages": [{"role": "user", "content": prompt}],
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
    reasoning_content = getattr(message, "reasoning_content", None)
    output = message.content or ""
    metadata = {
        "finish_reason": choice.finish_reason,
        "content_chars": len(output),
        "reasoning_content_chars": len(reasoning_content or ""),
        "usage": response.usage.model_dump() if response.usage else None,
    }
    return output, metadata


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
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()
    deepseek_client = build_deepseek_client(args)

    rows = []
    for case in load_eval_cases(args.eval_cases):
        title = case.get("title", case["id"])
        print("=" * 70)
        print(title)
        print("=" * 70)

        print("\n>>> Qwen base output:")
        print("-" * 50)
        with model.disable_adapter():
            base_output = generate_local(
                model,
                tokenizer,
                case["prompt"],
                args.max_new_tokens,
                args.temperature,
            )
        print(base_output)

        print("\n>>> Qwen LoRA output:")
        print("-" * 50)
        lora_output = generate_local(
            model,
            tokenizer,
            case["prompt"],
            args.max_new_tokens,
            args.temperature,
        )
        print(lora_output)

        deepseek_output = None
        deepseek_metadata = None
        if deepseek_client is not None:
            print("\n>>> DeepSeek-V4 API output:")
            print("-" * 50)
            deepseek_output, deepseek_metadata = generate_deepseek(
                deepseek_client,
                args,
                case["prompt"],
            )
            print(deepseek_output)

        rows.append(
            {
                "id": case["id"],
                "title": title,
                "prompt": case["prompt"],
                "qwen_base_output": base_output,
                "qwen_lora_output": lora_output,
                "deepseek_output": deepseek_output,
                "deepseek_metadata": deepseek_metadata,
            }
        )

    payload = {
        "comparison_type": "qualitative_example_outputs",
        "metric_note": (
            "These examples do not include reference code, so they are for qualitative "
            "output-style comparison only. Do not report them as accuracy metrics."
        ),
        "models": {
            "qwen_base": args.base_model,
            "qwen_lora_adapter": args.adapter,
            "deepseek_v4_api": None if args.skip_deepseek else args.deepseek_model,
        },
        "cases": rows,
    }
    write_json(args.output, payload)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
