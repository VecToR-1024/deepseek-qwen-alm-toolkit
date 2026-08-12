#!/usr/bin/env python3
"""Split one immutable multi-source import into ordered collection batches."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from deepseek_distill.rejection_sampling import publish_json_once, publish_jsonl_once


_PREFIX_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--import-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--first-batch-size", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_positive(args.batch_size, "--batch-size")
    if args.first_batch_size is not None:
        _validate_positive(args.first_batch_size, "--first-batch-size")
    if not _PREFIX_RE.fullmatch(args.prefix):
        raise ValueError("--prefix may contain only letters, digits, dot, dash, underscore")

    tasks = _read_jsonl(args.tasks)
    summary = _read_json(args.import_summary)
    _validate_parent(tasks, summary=summary)
    slices = _batch_slices(
        len(tasks),
        first_batch_size=args.first_batch_size,
        batch_size=args.batch_size,
    )
    parent_tasks_sha256 = _sha256(args.tasks)
    parent_summary_sha256 = _sha256(args.import_summary)
    batches: list[dict[str, Any]] = []
    for batch_index, (start, end) in enumerate(slices, start=1):
        batch_tasks = tasks[start:end]
        count = len(batch_tasks)
        stem = f"{args.prefix}_batch_{batch_index:03d}"
        tasks_path = args.output_dir / f"{stem}_tasks_{count}.jsonl"
        summary_path = args.output_dir / f"{stem}_import_{count}.json"
        ordered_ids = [_record_id(task) for task in batch_tasks]
        batch_summary = copy.deepcopy(summary)
        batch_summary["tasks"] = count
        batch_summary["ordered_task_ids"] = ordered_ids
        batch_summary["ordered_task_ids_sha256"] = _ordered_id_hash(ordered_ids)
        batch_summary["derivation"] = {
            "schema_version": "coding.import.multisource.slice.v1",
            "parent_tasks": args.tasks.as_posix(),
            "parent_tasks_sha256": parent_tasks_sha256,
            "parent_import_summary": args.import_summary.as_posix(),
            "parent_import_summary_sha256": parent_summary_sha256,
            "batch_index": batch_index,
            "start_index": start,
            "end_index_exclusive": end,
        }
        publish_jsonl_once(tasks_path, batch_tasks)
        publish_json_once(summary_path, batch_summary)
        batches.append(
            {
                "batch_index": batch_index,
                "start_index": start,
                "end_index_exclusive": end,
                "tasks": count,
                "tasks_path": tasks_path.as_posix(),
                "tasks_sha256": _sha256(tasks_path),
                "import_summary_path": summary_path.as_posix(),
                "import_summary_sha256": _sha256(summary_path),
            }
        )

    manifest = {
        "schema_version": "coding.import.multisource.split.v1",
        "source": summary.get("source"),
        "input": {
            "tasks": args.tasks.as_posix(),
            "tasks_sha256": parent_tasks_sha256,
            "import_summary": args.import_summary.as_posix(),
            "import_summary_sha256": parent_summary_sha256,
        },
        "total_tasks": len(tasks),
        "first_batch_size": args.first_batch_size,
        "batch_size": args.batch_size,
        "duplicate_task_ids": 0,
        "batches": batches,
    }
    manifest_path = args.output_dir / f"{args.prefix}_split_manifest.json"
    manifest_status = publish_json_once(manifest_path, manifest)
    print(
        json.dumps(
            {
                "event": "multisource_import_split_complete",
                "source": summary.get("source"),
                "total_tasks": len(tasks),
                "batches": len(batches),
                "manifest": manifest_path.as_posix(),
                "manifest_status": manifest_status,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _validate_parent(
    tasks: list[dict[str, Any]],
    *,
    summary: Mapping[str, Any],
) -> None:
    if summary.get("schema_version") != "coding.import.multisource.v1":
        raise ValueError("parent import summary schema_version is incompatible")
    if summary.get("status") != "ok" or summary.get("tasks") != len(tasks):
        raise ValueError("parent import summary does not match task count")
    ordered_ids = [_record_id(task) for task in tasks]
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("parent tasks contain duplicate IDs")
    if summary.get("ordered_task_ids") != ordered_ids:
        raise ValueError("parent import summary task order does not match JSONL")
    if summary.get("ordered_task_ids_sha256") != _ordered_id_hash(ordered_ids):
        raise ValueError("parent import summary ordered ID hash is invalid")


def _batch_slices(
    total: int,
    *,
    first_batch_size: int | None,
    batch_size: int,
) -> list[tuple[int, int]]:
    if total <= 0:
        raise ValueError("parent task import must not be empty")
    result: list[tuple[int, int]] = []
    cursor = 0
    if first_batch_size is not None:
        end = min(total, first_batch_size)
        result.append((0, end))
        cursor = end
    while cursor < total:
        end = min(total, cursor + batch_size)
        result.append((cursor, end))
        cursor = end
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _record_id(record: Mapping[str, Any]) -> str:
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("task record has no non-empty string id")
    return record_id


def _ordered_id_hash(ordered_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_positive(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


if __name__ == "__main__":
    raise SystemExit(main())
