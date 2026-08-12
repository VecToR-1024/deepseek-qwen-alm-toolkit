#!/usr/bin/env python3
"""Run the 100-task, three-attempt blind TACO stdin/stdout pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from deepseek_distill.api import DEFAULT_BASE_URL, DeepSeekClient, GenerationConfig
from deepseek_distill.rejection_sampling import (
    build_rejection_sampling_datasets,
    publish_json_once,
    publish_jsonl_once,
    run_rejection_sampling,
    validate_campaign_tasks,
    write_rejection_sampling_outputs,
)
from deepseek_distill.taco import (
    TACO_DATASET_ID,
    TACO_PILOT_SEED,
    TACO_REVISION,
    TACO_SPLIT,
    TACO_TASK_SCHEMA_VERSION,
    TACO_TRAIN_SHARD,
)


EXPECTED_TASKS = 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=EXPECTED_TASKS)
    parser.add_argument(
        "--limit",
        type=int,
        help="Use only the first N tasks from the immutable input (for smoke runs)",
    )
    parser.add_argument("--target", type=int, default=EXPECTED_TASKS)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-minute", type=float, default=120)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--phase-timeout", type=float, default=5.0)
    parser.add_argument("--max-output-characters", type=int, default=65_536)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip API waves and rebuild outputs from durable attempt files",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tasks = _read_jsonl(args.tasks)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        tasks = tasks[: args.limit]
    validate_campaign_tasks(
        tasks,
        expected_count=args.expected_tasks,
        expected_revision=TACO_REVISION,
        expected_schema_version=TACO_TASK_SCHEMA_VERSION,
        expected_dataset=TACO_DATASET_ID,
        expected_config=None,
        expected_split=TACO_SPLIT,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    campaign_tasks = args.run_dir / f"selected_tasks_{args.expected_tasks}.jsonl"
    publish_jsonl_once(campaign_tasks, tasks)
    config = GenerationConfig(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_logprobs=args.top_logprobs,
        max_tokens=args.max_tokens,
    )
    manifest = {
        "schema_version": "coding.collection.taco.pilot.v1",
        "dataset": {
            "dataset": TACO_DATASET_ID,
            "split": TACO_SPLIT,
            "revision": TACO_REVISION,
            "shard": TACO_TRAIN_SHARD,
            "selection": "random",
            "seed": TACO_PILOT_SEED,
            "selected_tasks": args.expected_tasks,
            "sampling_frame": "eligible stdin/stdout rows in one pinned train shard",
            "ordered_task_ids_sha256": hashlib.sha256(
                "\n".join(task["id"] for task in tasks).encode("utf-8")
            ).hexdigest(),
        },
        "generation": {"model": config.model, **config.as_metadata()},
        "sampling": {
            "max_attempts_per_task": 3,
            "stop_after_first_pass": True,
            "blind": True,
        },
        "provider": {"name": "DeepSeek", "base_url": args.base_url},
        "verification": {
            "interface": "stdin_stdout",
            "output_comparison": "strip_outer_whitespace_exact_v1",
            "host_security_boundary": "child_process_not_security_sandbox",
        },
    }
    publish_json_once(args.run_dir / "campaign_manifest.json", manifest)
    def report_wave(wave) -> None:
        print(
            json.dumps({"event": "wave_complete", **asdict(wave)}, ensure_ascii=False),
            flush=True,
        )

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
            progress=report_wave,
        )
    raw = _read_jsonl(args.run_dir / "raw_attempts.jsonl")
    normalized = _read_jsonl_if_exists(args.run_dir / "normalized_attempts.jsonl")
    verified = _read_jsonl_if_exists(args.run_dir / "verifier_attempts.jsonl")
    datasets = build_rejection_sampling_datasets(
        selected_tasks=tasks,
        raw_records=raw,
        normalized_records=normalized,
        verifier_records=verified,
        target=args.target,
        embed_rejected_records=False,
    )
    output_statuses = write_rejection_sampling_outputs(args.run_dir, datasets)
    summary = {
        "schema_version": "coding.collection.taco.pilot.summary.v1",
        "run": asdict(run_summary) if run_summary is not None else {
            "aggregate_only": True,
            "selected_tasks": len(tasks),
            "raw_attempts": len(raw),
            "normalized_attempts": len(normalized),
            "verifier_results": len(verified),
        },
        "dataset": asdict(datasets.summary),
        "failure_categories": dict(
            sorted(
                Counter(
                    record.get("failure_category")
                    for record in verified
                    if record.get("failure_category") != "passed"
                ).items()
            )
        ),
        "outputs": output_statuses,
    }
    publish_json_once(args.run_dir / "pilot_summary.json", summary)
    print(json.dumps({"event": "pilot_complete", **summary}, ensure_ascii=False))
    return 0 if datasets.summary.target_met else 2


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(value)
    return records


def _read_jsonl_if_exists(path: Path) -> list[dict]:
    return _read_jsonl(path) if path.exists() else []


if __name__ == "__main__":
    raise SystemExit(main())
