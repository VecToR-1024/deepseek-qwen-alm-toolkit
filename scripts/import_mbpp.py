#!/usr/bin/env python3
"""Import a pinned, train-only MBPP task subset into internal JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from deepseek_distill.mbpp import MBPP_REVISION, load_mbpp_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import official MBPP full/train tasks from a pinned revision."
    )
    parser.add_argument("--output", type=Path, required=True, help="New task JSONL path")
    parser.add_argument("--limit", type=int, default=20, help="Number of tasks (default: 20)")
    parser.add_argument(
        "--selection",
        choices=("first", "random"),
        default="first",
        help="Deterministic selection policy (default: first)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for random selection")
    parser.add_argument(
        "--revision",
        default=MBPP_REVISION,
        help="Pinned Hugging Face dataset revision",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional machine-readable JSON summary path",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional writable Hugging Face cache directory",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if args.summary_output is not None and args.summary_output.exists():
        raise SystemExit(f"refusing to overwrite existing summary: {args.summary_output}")

    tasks = load_mbpp_tasks(
        limit=args.limit,
        selection=args.selection,
        seed=args.seed,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    _write_jsonl_atomic(args.output, tasks)
    summary = {
        "status": "ok",
        "output": str(args.output.resolve()),
        "tasks": len(tasks),
        "selection": args.selection,
        "seed": args.seed,
        "revision": args.revision,
        "first_task_id": tasks[0]["id"] if tasks else None,
        "last_task_id": tasks[-1]["id"] if tasks else None,
    }
    if args.summary_output is not None:
        _write_json_atomic(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    sys.exit(main())
