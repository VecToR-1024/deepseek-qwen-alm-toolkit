#!/usr/bin/env python3
"""Finalize already-collected attempt queues without making API requests."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deepseek_distill.breadth_aggregation import aggregate_attempt_campaign
from deepseek_distill.code_verifier import verify_jsonl
from deepseek_distill.normalize import normalize_jsonl_append
from deepseek_distill.rejection_sampling import (
    _append_normalization_failures_to_verifier as append_normalization_failures_to_verifier,
)


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
    parser.add_argument("--verifier-workers", type=int, default=8)
    parser.add_argument("--phase-timeout", type=float, default=12.0)
    parser.add_argument("--max-output-characters", type=int, default=131_072)
    return parser


def finalize_stopped_attempts(
    *,
    run_dir: Path,
    selected_tasks_path: Path,
    target: int,
    max_attempts_per_task: int,
    verifier_workers: int,
    phase_timeout_seconds: float,
    max_output_characters: int,
) -> dict[str, Any]:
    """Drain raw queues, verify pending traces, and publish a partial aggregate."""
    run_dir = Path(run_dir)
    selected_tasks_path = Path(selected_tasks_path)
    if not selected_tasks_path.is_file():
        raise FileNotFoundError(f"selected task file does not exist: {selected_tasks_path}")
    raw_path = run_dir / "raw_attempts.jsonl"
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw attempt file does not exist: {raw_path}")
    if target <= 0:
        raise ValueError("target must be positive")
    if verifier_workers <= 0:
        raise ValueError("verifier_workers must be positive")

    normalized_path = run_dir / "normalized_attempts.jsonl"
    normalization_errors_path = run_dir / "normalization_errors.jsonl"
    verifier_path = run_dir / "verifier_attempts.jsonl"
    normalization = normalize_jsonl_append(
        raw_path,
        normalized_path,
        error_output_path=normalization_errors_path,
    )
    projected_failures = append_normalization_failures_to_verifier(
        raw_path=raw_path,
        normalization_errors_path=normalization_errors_path,
        verifier_path=verifier_path,
    )
    verification = verify_jsonl(
        input_path=normalized_path,
        output_path=verifier_path,
        phase_timeout_seconds=phase_timeout_seconds,
        max_output_characters=max_output_characters,
        max_workers=verifier_workers,
    )
    aggregate = aggregate_attempt_campaign(
        run_dir=run_dir,
        selected_tasks_path=selected_tasks_path,
        target=target,
        max_attempts_per_task=max_attempts_per_task,
    )
    return {
        "schema_version": "coding.collection.offline_finalization.v1",
        "run_dir": str(run_dir),
        "api_requests": 0,
        "normalization": asdict(normalization),
        "normalization_failures_projected": projected_failures,
        "verification": asdict(verification),
        "aggregate": aggregate,
        "training_started": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = finalize_stopped_attempts(
        run_dir=args.run_dir,
        selected_tasks_path=args.selected_tasks,
        target=args.target,
        max_attempts_per_task=args.max_attempts_per_task,
        verifier_workers=args.verifier_workers,
        phase_timeout_seconds=args.phase_timeout,
        max_output_characters=args.max_output_characters,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
