#!/usr/bin/env python3
"""Build earliest-pass MBPP datasets and a deterministic first-200 subset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from deepseek_distill.mbpp import MBPP_REVISION
from deepseek_distill.rejection_sampling import (
    build_rejection_sampling_datasets,
    validate_campaign_tasks,
    write_rejection_sampling_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tasks = _read_jsonl(args.run_dir / "selected_tasks_300.jsonl")
    validate_campaign_tasks(
        tasks,
        expected_count=300,
        expected_revision=MBPP_REVISION,
    )
    datasets = build_rejection_sampling_datasets(
        selected_tasks=tasks,
        raw_records=_read_jsonl(args.run_dir / "raw_attempts.jsonl"),
        normalized_records=_read_jsonl(args.run_dir / "normalized_attempts.jsonl"),
        verifier_records=_read_jsonl(args.run_dir / "verifier_attempts.jsonl"),
        target=args.target,
    )
    output_statuses = write_rejection_sampling_outputs(args.run_dir, datasets)
    summary = asdict(datasets.summary)
    print(
        json.dumps(
            {"summary": summary, "output_statuses": output_statuses},
            ensure_ascii=False,
        )
    )
    if summary["pending_attempt_slots"]:
        return 3
    if not summary["target_met"]:
        return 2
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


if __name__ == "__main__":
    raise SystemExit(main())
