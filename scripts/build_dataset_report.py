from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_codegen.dataset import write_json  # noqa: E402
from quant_codegen.quality import analyze_dataset, write_markdown_report  # noqa: E402


def main() -> None:
    dataset_path = ROOT / "data" / "quant_code.json"
    report_path = ROOT / "results" / "dataset_report.md"
    metrics_path = ROOT / "results" / "dataset_metrics.json"

    report = analyze_dataset(dataset_path)
    write_markdown_report(report, report_path)
    write_json(metrics_path, report)

    print(f"Wrote {report_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
