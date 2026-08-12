#!/usr/bin/env python3
"""Run the blind TACO v2 retry for unaccepted v1 length-truncated attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from deepseek_distill.api import DEFAULT_BASE_URL, DeepSeekClient, GenerationConfig
from deepseek_distill.code_verifier import verify_jsonl
from deepseek_distill.collector import collect_records
from deepseek_distill.normalize import normalize_jsonl_append
from deepseek_distill.rejection_sampling import publish_json_once, publish_jsonl_once
from deepseek_distill.taco_retry import (
    TACO_LENGTH_RETRY_MAX_TOKENS,
    build_length_retry_datasets,
    build_length_retry_tasks,
    portable_manifest_path,
    select_first_retry_per_problem,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        help="Use the first N canonical retries for a separate smoke run",
    )
    parser.add_argument(
        "--smoke-problems",
        type=int,
        help="Use the first retry from each of the first N distinct problems",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-minute", type=float, default=120)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=TACO_LENGTH_RETRY_MAX_TOKENS)
    parser.add_argument("--phase-timeout", type=float, default=8.0)
    parser.add_argument("--max-output-characters", type=int, default=65_536)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.prepare_only and args.aggregate_only:
        raise ValueError("--prepare-only and --aggregate-only are mutually exclusive")
    if args.limit is not None and args.smoke_problems is not None:
        raise ValueError("--limit and --smoke-problems are mutually exclusive")
    if args.max_tokens != TACO_LENGTH_RETRY_MAX_TOKENS:
        raise ValueError(
            f"TACO length-retry v2 requires --max-tokens "
            f"{TACO_LENGTH_RETRY_MAX_TOKENS}"
        )
    selected_path = args.v1_run_dir / "selected_tasks_100.jsonl"
    v1_normalized_path = args.v1_run_dir / "normalized_attempts.jsonl"
    v1_accepted_path = args.v1_run_dir / "accepted_unique.jsonl"
    selected_tasks = _read_jsonl(selected_path)
    v1_normalized = _read_jsonl(v1_normalized_path)
    accepted_v1 = _read_jsonl(v1_accepted_path)
    all_retry_tasks = build_length_retry_tasks(
        selected_tasks=selected_tasks,
        normalized_attempts=v1_normalized,
        accepted_v1=accepted_v1,
        max_tokens=args.max_tokens,
    )
    retry_tasks = all_retry_tasks
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        retry_tasks = retry_tasks[: args.limit]
    elif args.smoke_problems is not None:
        retry_tasks = select_first_retry_per_problem(
            retry_tasks,
            problem_limit=args.smoke_problems,
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    retry_tasks_path = args.run_dir / f"retry_tasks_{len(retry_tasks)}.jsonl"
    task_status = publish_jsonl_once(retry_tasks_path, retry_tasks)

    config = GenerationConfig(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_logprobs=args.top_logprobs,
        max_tokens=args.max_tokens,
    )
    manifest = {
        "schema_version": "coding.collection.taco.length_retry.v2",
        "source": {
            "v1_run_dir": portable_manifest_path(args.v1_run_dir),
            "selected_tasks_sha256": _sha256(selected_path),
            "normalized_attempts_sha256": _sha256(v1_normalized_path),
            "accepted_unique_sha256": _sha256(v1_accepted_path),
            "v1_length_attempts": sum(
                record.get("finish_reason") == "length" for record in v1_normalized
            ),
            "v1_unique_length_tasks": len(
                {
                    record["id"].rsplit("__attempt_", 1)[0]
                    for record in v1_normalized
                    if record.get("finish_reason") == "length"
                }
            ),
        },
        "selection": {
            "policy": "length_only_skip_v1_accepted",
            "canonical_retry_attempts": len(all_retry_tasks),
            "run_retry_attempts": len(retry_tasks),
            "limit": args.limit,
            "smoke_problems": args.smoke_problems,
            "teacher_feedback": False,
        },
        "generation": {"model": config.model, **config.as_metadata()},
        "provider": {"name": "DeepSeek", "base_url": args.base_url},
        "verification": {
            "interface": "stdin_stdout",
            "phase_timeout_seconds": args.phase_timeout,
            "output_comparison": "strip_outer_whitespace_exact_v1",
        },
    }
    publish_json_once(args.run_dir / "campaign_manifest.json", manifest)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "task_status": task_status,
                    "retry_attempts": len(retry_tasks),
                    "retry_tasks": str(retry_tasks_path.resolve()),
                },
                ensure_ascii=False,
            )
        )
        return 0

    raw_path = args.run_dir / "raw_retries.jsonl"
    normalized_path = args.run_dir / "normalized_retries.jsonl"
    normalization_errors_path = args.run_dir / "normalization_errors.jsonl"
    verifier_path = args.run_dir / "verifier_retries.jsonl"
    collection = None
    if not args.aggregate_only:
        client = DeepSeekClient(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        collection = collect_records(
            input_path=retry_tasks_path,
            output_path=raw_path,
            client=client,
            config=config,
            max_workers=args.workers,
            requests_per_minute=args.requests_per_minute,
            provider={"name": "DeepSeek", "base_url": args.base_url},
        )
    if not raw_path.exists():
        raise FileNotFoundError(f"retry raw file does not exist: {raw_path}")
    normalization = normalize_jsonl_append(
        raw_path,
        normalized_path,
        error_output_path=normalization_errors_path,
    )
    normalized_path.touch(exist_ok=True)
    verification = verify_jsonl(
        input_path=normalized_path,
        output_path=verifier_path,
        phase_timeout_seconds=args.phase_timeout,
        max_output_characters=args.max_output_characters,
    )
    raw = _read_jsonl(raw_path)
    normalized = _read_jsonl(normalized_path)
    verifier = _read_jsonl(verifier_path)
    datasets = build_length_retry_datasets(
        selected_tasks=selected_tasks,
        accepted_v1=accepted_v1,
        retry_tasks=retry_tasks,
        normalized_retries=normalized,
        verifier_results=verifier,
    )
    output_status = {
        "newly_accepted_unique": publish_jsonl_once(
            args.run_dir / "newly_accepted_unique.jsonl",
            datasets["newly_accepted_unique"],
        ),
        "combined_accepted_unique": publish_jsonl_once(
            args.run_dir / "combined_accepted_unique.jsonl",
            datasets["combined_accepted_unique"],
        ),
        "retry_outcomes": publish_jsonl_once(
            args.run_dir / "retry_outcomes.jsonl",
            datasets["retry_outcomes"],
        ),
    }
    finish_reasons = Counter(
        record.get("finish_reason")
        for record in normalized
        if isinstance(record.get("finish_reason"), str)
    )
    failure_counts = Counter(
        record.get("failure_category")
        for record in verifier
        if record.get("failure_category") != "passed"
    )
    summary = {
        "schema_version": "coding.collection.taco.length_retry.summary.v2",
        "mode": "aggregate_only" if args.aggregate_only else "collect",
        "collection": asdict(collection) if collection is not None else None,
        "normalization": asdict(normalization),
        "verification": asdict(verification),
        "dataset": datasets["summary"],
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "outputs": output_status,
    }
    _publish_retry_summary_once(args.run_dir / "retry_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


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


def _publish_retry_summary_once(path: Path, summary: dict) -> str:
    """Preserve the first run summary while allowing a no-op resumability check."""
    if not path.exists():
        return publish_json_once(path, summary)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(existing, dict):
        raise ValueError(f"{path}: expected JSON object")
    for key in ("schema_version", "dataset", "finish_reasons", "failure_counts"):
        if existing.get(key) != summary.get(key):
            raise FileExistsError(
                f"{path} already exists with a different {key!r} result"
            )
    return "unchanged"


if __name__ == "__main__":
    raise SystemExit(main())
