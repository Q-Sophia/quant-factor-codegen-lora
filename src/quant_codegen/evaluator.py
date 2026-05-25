from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Any

from .dataset import extract_python_code
from .mock_data import make_evaluation_panels, make_mock_ohlcv_frame


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


@dataclass
class ContractEvalResult:
    syntax_ok: bool
    has_factor_function: bool
    execution_ok: bool
    returns_series: bool
    index_aligned: bool
    no_infinite_values: bool
    has_finite_values: bool
    contract_pass: bool
    panels_passed: int
    panels_total: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FunctionalEvalResult:
    reference_valid: bool
    candidate_executable: bool
    functional_pass: bool | None
    selected_function: str | None
    candidate_functions: list[str]
    output_kind: str | None
    panels_matched: int
    panels_total: int
    compared_points: int
    valid_mask_match: bool | None
    max_absolute_error: float | None
    diagnostic_mean_correlation: float | None
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


def _error_text(message: str, limit: int = 240) -> str:
    return message if len(message) <= limit else f"{message[: limit - 3]}..."


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
            error=_error_text(f"SyntaxError: {exc}"),
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
        error = _error_text(f"{type(exc).__name__}: {exc}")
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


def _execute_contract(
    text: str,
    panels: list[Any],
) -> tuple[ContractEvalResult, list[Any]]:
    import numpy as np
    import pandas as pd

    try:
        tree = ast.parse(extract_python_code(text))
    except SyntaxError as exc:
        return (
            ContractEvalResult(
                syntax_ok=False,
                has_factor_function=False,
                execution_ok=False,
                returns_series=False,
                index_aligned=False,
                no_infinite_values=False,
                has_finite_values=False,
                contract_pass=False,
                panels_passed=0,
                panels_total=len(panels),
                error=_error_text(f"SyntaxError: {exc}"),
            ),
            [],
        )

    has_factor = _has_factor_function(tree)
    try:
        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "np": np,
            "pd": pd,
        }
        exec(compile(tree, "<factor_contract>", "exec"), namespace)
        factor = namespace.get("factor")
        if not callable(factor):
            raise ValueError("factor is not callable")
    except Exception as exc:  # noqa: BLE001 - report generated-code failures.
        return (
            ContractEvalResult(
                syntax_ok=True,
                has_factor_function=has_factor,
                execution_ok=False,
                returns_series=False,
                index_aligned=False,
                no_infinite_values=False,
                has_finite_values=False,
                contract_pass=False,
                panels_passed=0,
                panels_total=len(panels),
                error=_error_text(f"{type(exc).__name__}: {exc}"),
            ),
            [],
        )

    outputs: list[Any] = []
    panels_passed = 0
    for panel_index, panel in enumerate(panels, start=1):
        try:
            output = factor(panel.copy(deep=True))
        except Exception as exc:  # noqa: BLE001 - report generated-code failures.
            return (
                ContractEvalResult(
                    syntax_ok=True,
                    has_factor_function=has_factor,
                    execution_ok=False,
                    returns_series=False,
                    index_aligned=False,
                    no_infinite_values=False,
                    has_finite_values=False,
                    contract_pass=False,
                    panels_passed=panels_passed,
                    panels_total=len(panels),
                    error=_error_text(f"Panel {panel_index} {type(exc).__name__}: {exc}"),
                ),
                outputs,
            )
        if not isinstance(output, pd.Series):
            return (
                ContractEvalResult(
                    syntax_ok=True,
                    has_factor_function=has_factor,
                    execution_ok=True,
                    returns_series=False,
                    index_aligned=False,
                    no_infinite_values=False,
                    has_finite_values=False,
                    contract_pass=False,
                    panels_passed=panels_passed,
                    panels_total=len(panels),
                    error=f"Panel {panel_index} TypeError: factor output is not pd.Series",
                ),
                outputs,
            )
        if not output.index.equals(panel.index):
            return (
                ContractEvalResult(
                    syntax_ok=True,
                    has_factor_function=has_factor,
                    execution_ok=True,
                    returns_series=True,
                    index_aligned=False,
                    no_infinite_values=False,
                    has_finite_values=False,
                    contract_pass=False,
                    panels_passed=panels_passed,
                    panels_total=len(panels),
                    error=(
                        f"Panel {panel_index} ValueError: "
                        "factor output index is not aligned with input index"
                    ),
                ),
                outputs,
            )

        numeric = pd.to_numeric(output, errors="coerce")
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
        if bool(np.isinf(values).any()):
            return (
                ContractEvalResult(
                    syntax_ok=True,
                    has_factor_function=has_factor,
                    execution_ok=True,
                    returns_series=True,
                    index_aligned=True,
                    no_infinite_values=False,
                    has_finite_values=bool(np.isfinite(values).any()),
                    contract_pass=False,
                    panels_passed=panels_passed,
                    panels_total=len(panels),
                    error=f"Panel {panel_index} ValueError: factor output contains infinite values",
                ),
                outputs,
            )
        if not bool(np.isfinite(values).any()):
            return (
                ContractEvalResult(
                    syntax_ok=True,
                    has_factor_function=has_factor,
                    execution_ok=True,
                    returns_series=True,
                    index_aligned=True,
                    no_infinite_values=True,
                    has_finite_values=False,
                    contract_pass=False,
                    panels_passed=panels_passed,
                    panels_total=len(panels),
                    error=f"Panel {panel_index} ValueError: factor output has no finite values",
                ),
                outputs,
            )

        outputs.append(numeric)
        panels_passed += 1

    passed = has_factor and panels_passed == len(panels)
    return (
        ContractEvalResult(
            syntax_ok=True,
            has_factor_function=has_factor,
            execution_ok=True,
            returns_series=True,
            index_aligned=True,
            no_infinite_values=True,
            has_finite_values=True,
            contract_pass=passed,
            panels_passed=panels_passed,
            panels_total=len(panels),
            error=None if passed else "Required factor(df) function is missing",
        ),
        outputs,
    )


def evaluate_contract(
    text: str,
    panels: list[Any] | None = None,
) -> ContractEvalResult:
    """Check whether code can enter the standard factor execution pipeline."""
    result, _ = _execute_contract(text, panels or make_evaluation_panels())
    return result


def _candidate_function_names(tree: ast.AST) -> list[str]:
    names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.args.args
        and not node.name.startswith("_")
    ]
    if "factor" in names:
        names.remove("factor")
        names.insert(0, "factor")
    return names


def _normalize_candidate_output(output: Any, panel: Any) -> tuple[Any, str]:
    import numpy as np
    import pandas as pd

    if isinstance(output, pd.Series):
        series = output
        output_kind = "series"
    elif isinstance(output, pd.DataFrame) and "factor" in output.columns:
        series = output["factor"]
        output_kind = "dataframe_factor_column"
    elif isinstance(output, pd.DataFrame) and output.shape[1] == 1:
        series = output.iloc[:, 0]
        output_kind = "single_column_dataframe"
    else:
        raise TypeError("output is not a comparable Series or single factor column")

    if not series.index.equals(panel.index):
        raise ValueError("output index is not aligned with long-form input index")

    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if bool(np.isinf(values).any()):
        raise ValueError("output contains infinite values")
    if not bool(np.isfinite(values).any()):
        raise ValueError("output has no finite values")
    return numeric, output_kind


def _execute_candidates(
    text: str,
    panels: list[Any],
) -> tuple[list[str], list[tuple[str, str, list[Any]]], str | None]:
    import numpy as np
    import pandas as pd

    try:
        tree = ast.parse(extract_python_code(text))
    except SyntaxError as exc:
        return [], [], _error_text(f"SyntaxError: {exc}")

    function_names = _candidate_function_names(tree)
    if not function_names:
        return [], [], "No candidate function found"

    try:
        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "np": np,
            "pd": pd,
        }
        exec(compile(tree, "<candidate_factor>", "exec"), namespace)
    except Exception as exc:  # noqa: BLE001 - report generated-code failures.
        return function_names, [], _error_text(f"{type(exc).__name__}: {exc}")

    executable: list[tuple[str, str, list[Any]]] = []
    failures: list[str] = []
    for function_name in function_names:
        function = namespace.get(function_name)
        if not callable(function):
            continue
        outputs: list[Any] = []
        output_kind = None
        try:
            for panel in panels:
                raw_output = function(panel.copy(deep=True))
                normalized, output_kind = _normalize_candidate_output(raw_output, panel)
                outputs.append(normalized)
            executable.append((function_name, str(output_kind), outputs))
        except Exception as exc:  # noqa: BLE001 - report candidate failures.
            failures.append(f"{function_name}: {type(exc).__name__}: {exc}")

    if executable:
        return function_names, executable, None
    error = "; ".join(failures) if failures else "No callable candidate function found"
    return function_names, [], _error_text(error)


def _compare_output_lists(
    reference_outputs: list[Any],
    candidate_outputs: list[Any],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    import numpy as np

    panels_matched = 0
    compared_points = 0
    mask_match = True
    absolute_errors: list[float] = []
    correlations: list[float] = []
    for reference, candidate in zip(reference_outputs, candidate_outputs):
        reference_values = reference.to_numpy(dtype=float, na_value=np.nan)
        candidate_values = candidate.to_numpy(dtype=float, na_value=np.nan)
        reference_mask = np.isfinite(reference_values)
        candidate_mask = np.isfinite(candidate_values)
        mask_match = mask_match and bool(np.array_equal(reference_mask, candidate_mask))
        if not bool(np.isfinite(candidate_values[reference_mask]).all()):
            continue

        expected = reference_values[reference_mask]
        actual = candidate_values[reference_mask]
        compared_points += len(expected)
        absolute_errors.extend(np.abs(actual - expected).tolist())
        if bool(np.allclose(actual, expected, rtol=rtol, atol=atol)):
            panels_matched += 1

        if len(expected) >= 2 and np.std(expected) > 0 and np.std(actual) > 0:
            corr = float(np.corrcoef(expected, actual)[0, 1])
            if np.isfinite(corr):
                correlations.append(corr)

    return {
        "functional_pass": panels_matched == len(reference_outputs),
        "panels_matched": panels_matched,
        "compared_points": compared_points,
        "valid_mask_match": mask_match,
        "max_absolute_error": max(absolute_errors) if absolute_errors else None,
        "diagnostic_mean_correlation": (
            float(sum(correlations) / len(correlations)) if correlations else None
        ),
    }


def evaluate_against_reference(
    candidate_text: str,
    reference_text: str,
    panels: list[Any] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> tuple[ContractEvalResult, FunctionalEvalResult]:
    """Evaluate generated code against the held-out reference implementation.

    The reference output is expected to follow the training-data convention
    ``factor(df)``. Candidate outputs may use another function name, but must
    accept the same long-form market DataFrame and return a directly comparable
    factor result. Correctness is output agreement on all finite reference
    values across deterministic panels.
    """
    evaluation_panels = panels or make_evaluation_panels()
    reference_contract, reference_outputs = _execute_contract(reference_text, evaluation_panels)

    if not reference_contract.contract_pass:
        return (
            reference_contract,
            FunctionalEvalResult(
                reference_valid=False,
                candidate_executable=False,
                functional_pass=None,
                selected_function=None,
                candidate_functions=[],
                output_kind=None,
                panels_matched=0,
                panels_total=len(evaluation_panels),
                compared_points=0,
                valid_mask_match=None,
                max_absolute_error=None,
                diagnostic_mean_correlation=None,
                error=f"Invalid reference: {reference_contract.error}",
            ),
        )

    function_names, executable_candidates, execution_error = _execute_candidates(
        candidate_text,
        evaluation_panels,
    )
    if not executable_candidates:
        return (
            reference_contract,
            FunctionalEvalResult(
                reference_valid=True,
                candidate_executable=False,
                functional_pass=False,
                selected_function=None,
                candidate_functions=function_names,
                output_kind=None,
                panels_matched=0,
                panels_total=len(evaluation_panels),
                compared_points=0,
                valid_mask_match=None,
                max_absolute_error=None,
                diagnostic_mean_correlation=None,
                error=f"Candidate is not executable on task input: {execution_error}",
            ),
        )

    compared = [
        (function_name, output_kind, _compare_output_lists(reference_outputs, outputs, rtol, atol))
        for function_name, output_kind, outputs in executable_candidates
    ]
    selected_name, selected_kind, metrics = max(
        compared,
        key=lambda row: (
            bool(row[2]["functional_pass"]),
            int(row[2]["panels_matched"]),
            int(row[2]["compared_points"]),
        ),
    )
    return (
        reference_contract,
        FunctionalEvalResult(
            reference_valid=True,
            candidate_executable=True,
            functional_pass=bool(metrics["functional_pass"]),
            selected_function=selected_name,
            candidate_functions=function_names,
            output_kind=selected_kind,
            panels_matched=int(metrics["panels_matched"]),
            panels_total=len(evaluation_panels),
            compared_points=int(metrics["compared_points"]),
            valid_mask_match=bool(metrics["valid_mask_match"]),
            max_absolute_error=metrics["max_absolute_error"],
            diagnostic_mean_correlation=metrics["diagnostic_mean_correlation"],
            error=(
                None if metrics["functional_pass"]
                else "Executable candidate output differs from reference output"
            ),
        ),
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


def summarize_contract_results(results: list[ContractEvalResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0}

    return {
        "total": total,
        "contract_pass_count": sum(1 for item in results if item.contract_pass),
        "contract_pass_rate": round(
            sum(1 for item in results if item.contract_pass) / total, 4
        ),
        "error_count": sum(1 for item in results if item.error),
        "top_errors": _top_contract_errors(results),
    }


def summarize_reference_validity(results: list[ContractEvalResult]) -> dict[str, Any]:
    summary = summarize_contract_results(results)
    if summary["total"] == 0:
        return summary
    return {
        "total": summary["total"],
        "reference_valid_count": summary["contract_pass_count"],
        "reference_valid_rate": summary["contract_pass_rate"],
        "error_count": summary["error_count"],
        "top_errors": summary["top_errors"],
    }


def summarize_functional_results(results: list[FunctionalEvalResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0}

    valid = [item for item in results if item.reference_valid]
    executable = sum(1 for item in valid if item.candidate_executable)
    passed = sum(1 for item in valid if item.functional_pass is True)
    return {
        "total": total,
        "reference_valid_count": len(valid),
        "executable_count": executable,
        "executable_rate": round(executable / len(valid), 4) if valid else None,
        "executable_functional_correct_count": passed,
        "executable_functional_accuracy": round(passed / len(valid), 4) if valid else None,
        "error_count": sum(1 for item in valid if item.error),
        "top_errors": _top_functional_errors(valid),
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


def _top_contract_errors(
    results: list[ContractEvalResult],
    limit: int = 10,
) -> list[dict[str, Any]]:
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


def _top_functional_errors(
    results: list[FunctionalEvalResult],
    limit: int = 10,
) -> list[dict[str, Any]]:
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
