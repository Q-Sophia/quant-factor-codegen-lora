from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PYTHON_FENCE_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def load_alpaca_dataset(path: str | Path) -> list[dict[str, str]]:
    """Load the Alpaca-style dataset and normalize fields to strings."""
    dataset_path = Path(path)
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list, got {type(raw).__name__}")

    rows: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Row {index} is not an object")
        rows.append(
            {
                "instruction": str(item.get("instruction", "")).strip(),
                "input": str(item.get("input", "")).strip(),
                "output": str(item.get("output", "")).strip(),
            }
        )
    return rows


def to_chat_examples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert Alpaca rows into user/assistant examples."""
    examples: list[dict[str, str]] = []
    for row in rows:
        instruction = row["instruction"]
        input_text = row["input"]
        output_text = row["output"]
        if not instruction or not output_text:
            continue
        user = instruction if not input_text else f"{instruction}\n{input_text}"
        examples.append({"user": user, "assistant": output_text})
    return examples


def extract_python_code(text: str) -> str:
    """Extract Python code from a fenced markdown block if present."""
    match = PYTHON_FENCE_RE.search(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
