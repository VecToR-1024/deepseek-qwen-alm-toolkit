#!/usr/bin/env python3
"""Freeze an ordered target and reserve from audited clean coding records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from deepseek_distill.rejection_sampling import publish_json_once, publish_jsonl_once


_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Audited clean_accepted JSONL in deterministic merge order.",
    )
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if isinstance(args.target, bool) or args.target <= 0:
        raise ValueError("--target must be a positive integer")
    inputs = [_parse_input(value) for value in args.input]
    labels = [label for label, _ in inputs]
    if len(labels) != len(set(labels)):
        raise ValueError("--input labels must be unique")

    merged: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    problem_ids: set[str] = set()
    dataset_counts: Counter[str] = Counter()
    input_metadata: list[dict[str, Any]] = []
    for label, path in inputs:
        records = _read_jsonl(path)
        for record in records:
            record_id = _required_string(record.get("id"), "record id")
            problem_id = _problem_id(record)
            if record_id in record_ids:
                raise ValueError(f"duplicate record id {record_id!r}")
            if problem_id in problem_ids:
                raise ValueError(f"duplicate problem id {problem_id!r}")
            _validate_clean_record(record, record_id=record_id)
            record_ids.add(record_id)
            problem_ids.add(problem_id)
            dataset_counts[_dataset_name(record)] += 1
            merged.append(record)
        input_metadata.append(
            {
                "label": label,
                "path": path.as_posix(),
                "records": len(records),
                "sha256": _sha256(path),
            }
        )

    if len(merged) < args.target:
        raise RuntimeError(
            f"clean target shortfall: have {len(merged)}, need {args.target}"
        )
    selected = merged[: args.target]
    reserve = merged[args.target :]
    all_path = args.output_dir / "all_new_clean.jsonl"
    target_path = args.output_dir / f"accepted_first_{args.target}.jsonl"
    reserve_path = args.output_dir / "reserve_clean.jsonl"
    publish_jsonl_once(all_path, merged)
    publish_jsonl_once(target_path, selected)
    publish_jsonl_once(reserve_path, reserve)
    manifest = {
        "schema_version": "offline_alm.multisource_clean_target.v1",
        "input_order": labels,
        "inputs": input_metadata,
        "counts": {
            "all_new_clean": len(merged),
            "target": args.target,
            "accepted_first_target": len(selected),
            "reserve": len(reserve),
            "shortfall": 0,
        },
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "duplicates": {
            "record_ids": 0,
            "problem_ids": 0,
        },
        "selection": (
            "first_unique_clean_records_in_declared_input_and_jsonl_order"
        ),
        "outputs": {
            "all_new_clean": _output_metadata(all_path, len(merged)),
            "accepted_first_target": _output_metadata(
                target_path,
                len(selected),
            ),
            "reserve_clean": _output_metadata(reserve_path, len(reserve)),
        },
        "training_started": False,
    }
    manifest_path = args.output_dir / "clean_target_manifest.json"
    status = publish_json_once(manifest_path, manifest)
    print(
        json.dumps(
            {
                "event": "clean_multisource_target_complete",
                "all_new_clean": len(merged),
                "target": args.target,
                "reserve": len(reserve),
                "manifest": manifest_path.as_posix(),
                "manifest_status": status,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--input must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not _LABEL_RE.fullmatch(label):
        raise ValueError("--input label contains unsupported characters")
    if not raw_path:
        raise ValueError("--input path must not be empty")
    return label, Path(raw_path)


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


def _problem_id(record: Mapping[str, Any]) -> str:
    sampling = record.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("clean record has no sampling metadata")
    return _required_string(sampling.get("problem_id"), "sampling.problem_id")


def _validate_clean_record(record: Mapping[str, Any], *, record_id: str) -> None:
    if record.get("finish_reason") != "stop":
        raise ValueError(f"{record_id}: clean record finish_reason is not stop")
    verification = record.get("coding_verification")
    if (
        not isinstance(verification, Mapping)
        or verification.get("failure_category") != "passed"
    ):
        raise ValueError(f"{record_id}: clean record did not pass verification")


def _dataset_name(record: Mapping[str, Any]) -> str:
    task = record.get("task")
    source = task.get("source") if isinstance(task, Mapping) else None
    dataset = source.get("dataset") if isinstance(source, Mapping) else None
    return str(dataset) if dataset else "unknown"


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _output_metadata(path: Path, records: int) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
