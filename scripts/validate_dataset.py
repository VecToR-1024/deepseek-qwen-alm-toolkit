"""Validate a raw or normalized DeepSeek teacher JSONL file."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from deepseek_distill.validate import validate_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL file to validate")
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    parser.add_argument(
        "--fail-on-api-error",
        action="store_true",
        help="exit unsuccessfully when terminal API errors are present",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="exit unsuccessfully when byte reconstruction warnings are present",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_jsonl(args.input)
    rendered = json.dumps(asdict(report), ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    if report.invalid_records:
        return 1
    if args.fail_on_api_error and report.api_errors:
        return 1
    if args.fail_on_warning and report.warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
