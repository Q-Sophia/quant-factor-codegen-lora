from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_codegen.dataset import write_json  # noqa: E402
from quant_codegen.evaluator import evaluate_code_text, summarize_eval_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepSeek API on factor-code cases.")
    parser.add_argument(
        "--model",
        default="deepseek-reasoner",
        help="DeepSeek reasoning model name. The official R1 API historically uses deepseek-reasoner.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="DeepSeek OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Environment variable name that stores the API key.",
    )
    parser.add_argument(
        "--eval-cases",
        type=Path,
        default=ROOT / "examples" / "eval_cases.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "deepseek_api_outputs.json",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default="enabled",
        help="DeepSeek thinking mode. Keep enabled for an R1-style reasoning baseline.",
    )
    return parser.parse_args()


def load_eval_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Set {args.api_key_env} first.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    rows = []
    evals = []
    for case in load_eval_cases(args.eval_cases):
        title = case.get("title", case["id"])
        print("=" * 70)
        print(title)
        print("=" * 70)

        messages = [{"role": "user", "content": case["prompt"]}]

        request_kwargs = {
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
        output = response.choices[0].message.content or ""
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
                "usage": response.usage.model_dump() if response.usage else None,
            }
        )

    payload = {
        "provider": "deepseek",
        "model": args.model,
        "base_url": args.base_url,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort,
        "summary": summarize_eval_results(evals),
        "cases": rows,
    }
    write_json(args.output, payload)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
