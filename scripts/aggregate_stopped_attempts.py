#!/usr/bin/env python3
"""Aggregate the already-verified prefix of a stopped attempt campaign."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from deepseek_distill.breadth_aggregation import aggregate_attempt_campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selected-tasks", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument(
        "--max-attempts-per-task",
        type=int,
        choices=(1, 2, 3),
        default=3,
    )
    return parser


def aggregate_stopped_attempts(
    *,
    run_dir: Path,
    selected_tasks_path: Path,
    target: int,
    max_attempts_per_task: int,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    selected_tasks_path = Path(selected_tasks_path)
    if not selected_tasks_path.is_file():
        raise FileNotFoundError(f"selected task file not found: {selected_tasks_path}")
    for name in (
        "raw_attempts.jsonl",
        "normalized_attempts.jsonl",
        "verifier_attempts.jsonl",
    ):
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"append-only campaign input not found: {path}")
    if target <= 0:
        raise ValueError("target must be positive")

    aggregate = aggregate_attempt_campaign(
        run_dir=run_dir,
        selected_tasks_path=selected_tasks_path,
        target=target,
        max_attempts_per_task=max_attempts_per_task,
    )
    return {
        "schema_version": "coding.collection.partial_aggregation.v1",
        "run_dir": str(run_dir),
        "cutoff_policy": "already_fsynced_verifier_prefix",
        "api_requests": 0,
        "normalization_started": False,
        "verification_started": False,
        "aggregate": aggregate,
        "training_started": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = aggregate_stopped_attempts(
        run_dir=args.run_dir,
        selected_tasks_path=args.selected_tasks,
        target=args.target,
        max_attempts_per_task=args.max_attempts_per_task,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
