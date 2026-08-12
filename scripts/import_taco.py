#!/usr/bin/env python3
"""Import a safe, pinned TACO train-shard pilot into internal JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from deepseek_distill.rejection_sampling import publish_json_once, publish_jsonl_once
from deepseek_distill.taco import (
    TACO_PILOT_SEED,
    TACO_REVISION,
    TACO_TRAIN_SHARD,
    load_taco_tasks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New task JSONL path")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--selection", choices=("first", "random"), default="random"
    )
    parser.add_argument("--seed", type=int, default=TACO_PILOT_SEED)
    parser.add_argument("--revision", default=TACO_REVISION)
    parser.add_argument("--shard", default=TACO_TRAIN_SHARD)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--summary-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tasks = load_taco_tasks(
        limit=args.limit,
        selection=args.selection,
        seed=args.seed,
        revision=args.revision,
        shard_path=args.shard,
        cache_dir=args.cache_dir,
    )
    output_status = publish_jsonl_once(args.output, tasks)
    summary = {
        "schema_version": "coding.import.taco.pilot.v1",
        "status": "ok",
        "output_status": output_status,
        "output": str(args.output.resolve()),
        "tasks": len(tasks),
        "selection": args.selection,
        "seed": args.seed,
        "revision": args.revision,
        "shard": args.shard,
        "sampling_frame": "eligible stdin/stdout rows in one pinned train shard",
        "ordered_task_ids_sha256": hashlib.sha256(
            "\n".join(task["id"] for task in tasks).encode("utf-8")
        ).hexdigest(),
    }
    if args.summary_output is not None:
        publish_json_once(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
