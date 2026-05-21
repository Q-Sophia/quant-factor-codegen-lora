from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .dataset import load_alpaca_dataset


OPERATIONS = [
    "rank",
    "rolling",
    "groupby",
    "corr",
    "std",
    "mean",
    "sum",
    "log",
    "diff",
    "shift",
    "clip",
    "pct_change",
]


def _length_stats(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    total = len(ordered)

    def percentile(pct: float) -> int:
        if total == 0:
            return 0
        index = min(total - 1, int((total - 1) * pct))
        return ordered[index]

    if not ordered:
        return {"min": 0, "p25": 0, "median": 0, "mean": 0, "p75": 0, "p95": 0, "max": 0}

    return {
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "mean": round(mean(ordered)),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def analyze_dataset(dataset_path: str | Path) -> dict[str, Any]:
    rows = load_alpaca_dataset(dataset_path)
    total = len(rows)

    input_lengths = [len(row["input"]) for row in rows]
    output_lengths = [len(row["output"]) for row in rows]
    total_lengths = [len(row["input"]) + len(row["output"]) for row in rows]

    op_counts = {
        op: sum(1 for row in rows if op in (row["input"] + row["output"]).lower())
        for op in OPERATIONS
    }
    instruction_counts = Counter(row["instruction"] for row in rows)

    return {
        "rows": total,
        "schema_ok": all(
            set(row.keys()) == {"instruction", "input", "output"} for row in rows
        ),
        "non_empty_rows": sum(
            1 for row in rows if row["instruction"] and row["input"] and row["output"]
        ),
        "unique_instructions": len(instruction_counts),
        "unique_inputs": len({row["input"] for row in rows}),
        "unique_outputs": len({row["output"] for row in rows}),
        "duplicate_inputs": total - len({row["input"] for row in rows}),
        "duplicate_outputs": total - len({row["output"] for row in rows}),
        "python_fence_count": sum(
            1 for row in rows if row["output"].strip().lower().startswith("```python")
        ),
        "factor_function_count": sum(1 for row in rows if "def factor" in row["output"]),
        "pandas_import_count": sum(1 for row in rows if "import pandas as pd" in row["output"]),
        "numpy_import_count": sum(1 for row in rows if "import numpy as np" in row["output"]),
        "length_stats": {
            "input": _length_stats(input_lengths),
            "output": _length_stats(output_lengths),
            "total": _length_stats(total_lengths),
        },
        "operation_counts": op_counts,
    }


def write_markdown_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = report["rows"]
    lines = [
        "# 数据质量报告",
        "",
        "## 概览",
        "",
        f"- 样本数：{rows}",
        f"- 非空样本数：{report['non_empty_rows']}",
        f"- 字段结构完整：{report['schema_ok']}",
        f"- 唯一 instruction 数：{report['unique_instructions']}",
        f"- 唯一 input 数：{report['unique_inputs']}",
        f"- 唯一 output 数：{report['unique_outputs']}",
        f"- 重复 input 数：{report['duplicate_inputs']}",
        f"- 重复 output 数：{report['duplicate_outputs']}",
        "",
        "## 代码格式",
        "",
        f"- Python fenced code block：{report['python_fence_count']} / {rows}",
        f"- 包含 `def factor`：{report['factor_function_count']} / {rows}",
        f"- 导入 Pandas：{report['pandas_import_count']} / {rows}",
        f"- 导入 NumPy：{report['numpy_import_count']} / {rows}",
        "",
        "## 长度分布",
        "",
        "| 字段 | min | p25 | median | mean | p75 | p95 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for key, stats in report["length_stats"].items():
        lines.append(
            f"| {key} | {stats['min']} | {stats['p25']} | {stats['median']} | "
            f"{stats['mean']} | {stats['p75']} | {stats['p95']} | {stats['max']} |"
        )

    lines.extend(
        [
            "",
            "## 常见操作统计",
            "",
            "| 操作 | 样本数 |",
            "|---|---:|",
        ]
    )
    for op, count in sorted(
        report["operation_counts"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"| `{op}` | {count} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
