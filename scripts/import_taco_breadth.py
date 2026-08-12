#!/usr/bin/env python3
"""Import 1,000 new breadth-first tasks from the pinned TACO train shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from deepseek_distill.rejection_sampling import publish_json_once, publish_jsonl_once
from deepseek_distill.taco import (
    TACO_REVISION,
    TACO_TRAIN_SHARD,
    load_taco_tasks,
)
from deepseek_distill.taco_breadth import (
    TACO_BREADTH_EXCLUDED_SOURCES,
    TACO_BREADTH_SEED,
    TACO_BREADTH_SELECTION_SCOPE,
    TACO_BREADTH_TASK_COUNT,
    validate_taco_breadth_tasks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=TACO_BREADTH_TASK_COUNT)
    parser.add_argument("--seed", type=int, default=TACO_BREADTH_SEED)
    parser.add_argument("--cache-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seed != TACO_BREADTH_SEED:
        raise ValueError(
            f"TACO breadth v2 requires --seed {TACO_BREADTH_SEED}"
        )
    prior_tasks = _read_jsonl(args.prior_tasks)
    prior_ids = {task["id"] for task in prior_tasks}
    if len(prior_ids) != len(prior_tasks):
        raise ValueError("prior task file contains duplicate ids")
    tasks = load_taco_tasks(
        limit=args.limit,
        selection="random",
        seed=args.seed,
        revision=TACO_REVISION,
        shard_path=TACO_TRAIN_SHARD,
        cache_dir=args.cache_dir,
        excluded_task_ids=prior_ids,
        excluded_sources=TACO_BREADTH_EXCLUDED_SOURCES,
        selection_scope=TACO_BREADTH_SELECTION_SCOPE,
    )
    validate_taco_breadth_tasks(
        tasks,
        prior_tasks=prior_tasks,
        expected_count=args.limit,
    )
    output_status = publish_jsonl_once(args.output, tasks)
    summary = {
        "schema_version": "coding.import.taco.breadth.v2",
        "status": "ok",
        "output_status": output_status,
        "output": args.output.as_posix(),
        "tasks": len(tasks),
        "selection": "random_after_exclusions",
        "seed": args.seed,
        "revision": TACO_REVISION,
        "shard": TACO_TRAIN_SHARD,
        "selection_scope": TACO_BREADTH_SELECTION_SCOPE,
        "excluded_sources": sorted(TACO_BREADTH_EXCLUDED_SOURCES),
        "excluded_prior_tasks": len(prior_tasks),
        "prior_tasks_path": args.prior_tasks.as_posix(),
        "prior_tasks_sha256": _sha256(args.prior_tasks),
        "ordered_task_ids_sha256": _ordered_id_hash(tasks),
    }
    publish_json_once(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


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
