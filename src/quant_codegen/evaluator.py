from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Any

from .dataset import extract_python_code
from .mock_data import make_mock_ohlcv_frame


@dataclass
class CodeEvalResult:
    syntax_ok: bool
    has_factor_function: bool
    imports_pandas: bool
    imports_numpy: bool
    execution_ok: bool | None
    returns_series: bool | None
    finite_values: bool | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _has_import(tree: ast.AST, module_name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == module_name for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module == module_name:
            return True
    return False


def _has_factor_function(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "factor":
            if not node.args.args:
                return False
            return node.args.args[0].arg == "df"
    return False


def evaluate_code_text(text: str, run_execution: bool = True) -> CodeEvalResult:
    """Evaluate generated code with static checks and optional local execution."""
    code = extract_python_code(text)

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return CodeEvalResult(
            syntax_ok=False,
            has_factor_function=False,
            imports_pandas=False,
            imports_numpy=False,
            execution_ok=False if run_execution else None,
            returns_series=False if run_execution else None,
            finite_values=False if run_execution else None,
            error=f"SyntaxError: {exc}",
        )

    has_factor = _has_factor_function(tree)
    imports_pandas = _has_import(tree, "pandas")
    imports_numpy = _has_import(tree, "numpy")

    if not run_execution:
        return CodeEvalResult(
            syntax_ok=True,
            has_factor_function=has_factor,
            imports_pandas=imports_pandas,
            imports_numpy=imports_numpy,
            execution_ok=None,
            returns_series=None,
            finite_values=None,
        )

    try:
        import numpy as np
        import pandas as pd

        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "np": np,
            "pd": pd,
        }
        exec(compile(tree, "<generated_factor>", "exec"), namespace)
        factor = namespace.get("factor")
        if not callable(factor):
            raise ValueError("factor is not callable")

        df = make_mock_ohlcv_frame()
        result = factor(df.copy())
        returns_series = isinstance(result, pd.Series)
        finite_values = False
        if returns_series:
            values = pd.to_numeric(result, errors="coerce")
            finite_values = bool(np.isfinite(values.dropna()).all())
            if values.dropna().empty:
                finite_values = True

        return CodeEvalResult(
            syntax_ok=True,
            has_factor_function=has_factor,
            imports_pandas=imports_pandas,
            imports_numpy=imports_numpy,
            execution_ok=True,
            returns_series=returns_series,
            finite_values=finite_values,
        )
    except Exception as exc:  # noqa: BLE001 - report generated-code failures.
        error = f"{type(exc).__name__}: {exc}"
        return CodeEvalResult(
            syntax_ok=True,
            has_factor_function=has_factor,
            imports_pandas=imports_pandas,
            imports_numpy=imports_numpy,
            execution_ok=False,
            returns_series=False,
            finite_values=False,
            error=error,
        )


def summarize_eval_results(results: list[CodeEvalResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0}

    def rate(name: str) -> float:
        ok = sum(1 for item in results if getattr(item, name) is True)
        return round(ok / total, 4)

    return {
        "total": total,
        "syntax_pass_rate": rate("syntax_ok"),
        "signature_pass_rate": rate("has_factor_function"),
        "pandas_import_rate": rate("imports_pandas"),
        "numpy_import_rate": rate("imports_numpy"),
        "execution_pass_rate": rate("execution_ok"),
        "series_return_rate": rate("returns_series"),
        "finite_value_rate": rate("finite_values"),
        "error_count": sum(1 for item in results if item.error),
        "top_errors": _top_errors(results),
    }


def _top_errors(results: list[CodeEvalResult], limit: int = 10) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in results:
        if not item.error:
            continue
        key = item.error.splitlines()[0][:160]
        counts[key] = counts.get(key, 0) + 1
    return [
        {"error": error, "count": count}
        for error, count in sorted(counts.items(), key=lambda pair: pair[1], reverse=True)[:limit]
    ]
