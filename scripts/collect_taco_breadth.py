#!/usr/bin/env python3
"""Collect a one-attempt breadth-first TACO campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deepseek_distill.api import DEFAULT_BASE_URL, DeepSeekClient, GenerationConfig
from deepseek_distill.breadth_aggregation import aggregate_single_attempt_campaign
from deepseek_distill.rejection_sampling import (
    publish_json_once,
    publish_jsonl_once,
    run_rejection_sampling,
)
from deepseek_distill.taco import TACO_DATASET_ID, TACO_REVISION, TACO_TRAIN_SHARD
from deepseek_distill.taco_breadth import (
    TACO_BREADTH_EXCLUDED_SOURCES,
    TACO_BREADTH_SEED,
    TACO_BREADTH_SELECTION_SCOPE,
    TACO_BREADTH_TASK_COUNT,
    validate_taco_breadth_tasks,
)


TACO_BREADTH_MAX_TOKENS = 4096


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--prior-tasks", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=TACO_BREADTH_TASK_COUNT)
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N tasks from the validated 1,000-task manifest",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    mode.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect, normalize, and verify without loading all records for aggregation",
    )
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-minute", type=float, default=120)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=TACO_BREADTH_MAX_TOKENS)
    parser.add_argument("--phase-timeout", type=float, default=8.0)
    parser.add_argument("--max-output-characters", type=int, default=65_536)
    parser.add_argument(
        "--verifier-workers",
        type=int,
        default=4,
        help="Bounded parallel isolated verifier workers; output order stays deterministic",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_tokens != TACO_BREADTH_MAX_TOKENS:
        raise ValueError(
            f"TACO breadth v2 requires --max-tokens {TACO_BREADTH_MAX_TOKENS}"
        )
    all_tasks = _read_jsonl(args.tasks)
    prior_tasks = _read_jsonl(args.prior_tasks)
    validate_taco_breadth_tasks(
        all_tasks,
        prior_tasks=prior_tasks,
        expected_count=args.expected_tasks,
    )
    tasks = all_tasks
    if args.limit is not None:
        if args.limit <= 0 or args.limit > len(all_tasks):
            raise ValueError("--limit must be between 1 and --expected-tasks")
        tasks = all_tasks[: args.limit]

    args.run_dir.mkdir(parents=True, exist_ok=True)
    campaign_tasks = args.run_dir / f"selected_tasks_{len(tasks)}.jsonl"
    task_status = publish_jsonl_once(campaign_tasks, tasks)
    config = GenerationConfig(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_logprobs=args.top_logprobs,
        max_tokens=args.max_tokens,
    )
    manifest = build_campaign_manifest(
        all_tasks=all_tasks,
        run_tasks=tasks,
        tasks_path=args.tasks,
        prior_tasks_path=args.prior_tasks,
        prior_task_count=len(prior_tasks),
        config=config,
        base_url=args.base_url,
        phase_timeout=args.phase_timeout,
    )
    publish_json_once(args.run_dir / "campaign_manifest.json", manifest)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "event": "breadth_prepared",
                    "task_status": task_status,
                    "run_tasks": len(tasks),
                    "run_dir": args.run_dir.as_posix(),
                },
                ensure_ascii=False,
            )
        )
        return 0

    run_summary = None
    if not args.aggregate_only:
        client = DeepSeekClient(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        run_summary = run_rejection_sampling(
            selected_tasks_path=campaign_tasks,
            run_dir=args.run_dir,
            client=client,
            config=config,
            max_workers=args.workers,
            requests_per_minute=args.requests_per_minute,
            provider={"name": "DeepSeek", "base_url": args.base_url},
            phase_timeout_seconds=args.phase_timeout,
            max_output_characters=args.max_output_characters,
            verification_workers=args.verifier_workers,
            max_attempts_per_task=1,
        )
    if args.collect_only:
        assert run_summary is not None
        collection_complete = run_summary.raw_attempts == len(tasks)
        print(
            json.dumps(
                {
                    "event": "breadth_collection_complete",
                    "collection_complete": collection_complete,
                    "run": asdict(run_summary),
                },
                ensure_ascii=False,
            )
        )
        return 0 if collection_complete else 2

    summary = aggregate_single_attempt_campaign(
        run_dir=args.run_dir,
        selected_tasks_path=campaign_tasks,
        target=len(tasks),
    )
    print(
        json.dumps(
            {
                "event": "breadth_complete",
                "operation": (
                    "aggregate_only" if args.aggregate_only else "collect"
                ),
                "run": asdict(run_summary) if run_summary is not None else None,
                **summary,
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["collection_complete"] else 2


def build_campaign_manifest(
    *,
    all_tasks: list[dict],
    run_tasks: list[dict],
    tasks_path: Path,
    prior_tasks_path: Path,
    prior_task_count: int,
    config: GenerationConfig,
    base_url: str,
    phase_timeout: float,
) -> dict[str, Any]:
    return {
        "schema_version": "coding.collection.taco.breadth.v2",
        "dataset": {
            "dataset": TACO_DATASET_ID,
            "split": "train",
            "revision": TACO_REVISION,
            "shard": TACO_TRAIN_SHARD,
            "selection": "random_after_exclusions",
            "seed": TACO_BREADTH_SEED,
            "selection_scope": TACO_BREADTH_SELECTION_SCOPE,
            "excluded_sources": sorted(TACO_BREADTH_EXCLUDED_SOURCES),
            "excluded_prior_tasks": prior_task_count,
            "full_selected_tasks": len(all_tasks),
            "run_tasks": len(run_tasks),
            "tasks_path": tasks_path.as_posix(),
            "tasks_sha256": _sha256(tasks_path),
            "prior_tasks_path": prior_tasks_path.as_posix(),
            "prior_tasks_sha256": _sha256(prior_tasks_path),
            "ordered_full_task_ids_sha256": _ordered_id_hash(all_tasks),
            "ordered_run_task_ids_sha256": _ordered_id_hash(run_tasks),
        },
        "generation": {"model": config.model, **config.as_metadata()},
        "sampling": {
            "max_attempts_per_task": 1,
            "blind": True,
            "verifier_feedback": False,
        },
        "provider": {"name": "DeepSeek", "base_url": base_url},
        "verification": {
            "interface": "stdin_stdout",
            "phase_timeout_seconds": phase_timeout,
            "output_comparison": "strip_outer_whitespace_exact_v1",
            "host_security_boundary": "child_process_not_security_sandbox",
        },
    }


def _ordered_id_hash(tasks: list[dict]) -> str:
    return hashlib.sha256(
        "\n".join(task["id"] for task in tasks).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(record)
    return records


if __name__ == "__main__":
    raise SystemExit(main())
