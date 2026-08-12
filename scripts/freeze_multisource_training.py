#!/usr/bin/env python3
"""Freeze a deterministic, source-interleaved coding training dataset."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from deepseek_distill.rejection_sampling import publish_json_once, publish_jsonl_once
from deepseek_distill.training_data_audit import normalize_problem_text


_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_FENCE_RE = re.compile(r"```", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Audited clean JSONL in deterministic precedence order.",
    )
    parser.add_argument(
        "--target",
        type=_parse_target,
        required=True,
        help="Positive record count, or 'all' to retain every unique record.",
    )
    parser.add_argument("--near-threshold", type=float, default=0.9)
    parser.add_argument("--near-top-k", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target != "all" and (
        isinstance(args.target, bool) or args.target <= 0
    ):
        raise ValueError("--target must be a positive integer or 'all'")
    if not 0.0 <= args.near_threshold <= 1.0:
        raise ValueError("--near-threshold must be between zero and one")
    if isinstance(args.near_top_k, bool) or args.near_top_k < 0:
        raise ValueError("--near-top-k must be a non-negative integer")

    inputs = [_parse_input(value) for value in args.input]
    labels = [label for label, _ in inputs]
    if len(labels) != len(set(labels)):
        raise ValueError("--input labels must be unique")

    records: list[dict[str, Any]] = []
    input_metadata: list[dict[str, Any]] = []
    for label, path in inputs:
        source_records = _read_jsonl(path)
        for record in source_records:
            _validate_clean_record(record)
            records.append(record)
        input_metadata.append(
            {
                "label": label,
                "path": path.as_posix(),
                "records": len(source_records),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    unique, excluded = _deduplicate(records)
    interleaved, source_order = _round_robin_sources(unique)
    target_count = len(interleaved) if args.target == "all" else args.target
    if len(interleaved) < target_count:
        raise RuntimeError(
            f"clean target shortfall after exact deduplication: "
            f"have {len(interleaved)}, need {target_count}"
        )
    selected = interleaved[:target_count]
    reserve = interleaved[target_count:]
    near = _near_duplicate_report(
        selected,
        threshold=args.near_threshold,
        top_k=args.near_top_k,
    )

    output_dir = args.output_dir
    training_path = output_dir / "training_records.jsonl"
    reserve_path = output_dir / "reserve_records.jsonl"
    excluded_path = output_dir / "excluded_records.jsonl"
    publish_jsonl_once(training_path, selected)
    publish_jsonl_once(reserve_path, reserve)
    publish_jsonl_once(excluded_path, excluded)

    manifest = {
        "schema_version": "offline_alm.frozen_multisource_training.v1",
        "selection": {
            "target": args.target,
            "policy": "round_robin_by_first_seen_dataset_preserving_source_order",
            "source_order": source_order,
            "exact_deduplication": "normalized_problem_text_keep_first",
            "near_duplicates": "report_only_not_removed",
        },
        "inputs": input_metadata,
        "counts": {
            "input_records": len(records),
            "unique_after_exact_deduplication": len(unique),
            "exact_duplicates_excluded": len(excluded),
            "training_records": len(selected),
            "reserve_records": len(reserve),
        },
        "source_counts": {
            "input": _source_counts(records),
            "unique": _source_counts(unique),
            "training": _source_counts(selected),
            "reserve": _source_counts(reserve),
        },
        "near_duplicates": near,
        "outputs": {
            "training_records": _output_metadata(training_path, len(selected)),
            "reserve_records": _output_metadata(reserve_path, len(reserve)),
            "excluded_records": _output_metadata(excluded_path, len(excluded)),
        },
        "training_started": False,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_status = publish_json_once(manifest_path, manifest)
    print(
        json.dumps(
            {
                "event": "multisource_training_dataset_frozen",
                "training_records": len(selected),
                "reserve_records": len(reserve),
                "exact_duplicates_excluded": len(excluded),
                "near_duplicate_pairs": near["reported_pairs"],
                "manifest": manifest_path.as_posix(),
                "manifest_status": manifest_status,
            },
            ensure_ascii=False,
        ),
        flush=True,
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


def _parse_target(value: str) -> int | str:
    if value.lower() == "all":
        return "all"
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "target must be a positive integer or 'all'"
        ) from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def _validate_clean_record(record: Mapping[str, Any]) -> None:
    record_id = _required_string(record.get("id"), "record id")
    if record.get("schema_version") != "deepseek.teacher.normalized.v1":
        raise ValueError(f"{record_id}: unsupported normalized schema")
    if record.get("finish_reason") != "stop":
        raise ValueError(f"{record_id}: finish_reason must be stop")
    verification = record.get("coding_verification")
    if not isinstance(verification, Mapping) or verification.get(
        "failure_category"
    ) != "passed":
        raise ValueError(f"{record_id}: official verification did not pass")

    response_text = record.get("response_text")
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError(f"{record_id}: response_text must be non-empty")
    if _FENCE_RE.search(response_text):
        raise ValueError(f"{record_id}: response_text contains a Markdown fence")
    try:
        ast.parse(response_text)
    except SyntaxError as error:
        raise ValueError(f"{record_id}: response_text is not valid Python") from error

    validation = record.get("validation")
    if not isinstance(validation, Mapping) or validation.get(
        "content_bytes_match"
    ) is not True:
        raise ValueError(f"{record_id}: trace byte validation is not true")
    content_tokens = record.get("content_tokens")
    if not isinstance(content_tokens, list):
        raise ValueError(f"{record_id}: content_tokens must be a list")
    try:
        reconstructed = b"".join(bytes(row["bytes"]) for row in content_tokens)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{record_id}: invalid actual-token byte arrays") from error
    if reconstructed != response_text.encode("utf-8"):
        raise ValueError(f"{record_id}: actual-token bytes do not reconstruct response")

    task = record.get("task")
    if not isinstance(task, Mapping):
        raise ValueError(f"{record_id}: task must be an object")
    _required_string(task.get("problem_text"), f"{record_id}: task.problem_text")
    source = task.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"{record_id}: task.source must be an object")
    _required_string(source.get("dataset"), f"{record_id}: task.source.dataset")
    sampling = record.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError(f"{record_id}: sampling must be an object")
    _required_string(
        sampling.get("problem_id"), f"{record_id}: sampling.problem_id"
    )


def _deduplicate(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_record_ids: dict[str, str] = {}
    seen_problem_ids: dict[str, str] = {}
    seen_problem_texts: dict[str, str] = {}
    for record in records:
        record_id = str(record["id"])
        problem_id = str(record["sampling"]["problem_id"])
        fingerprint = normalize_problem_text(record["task"]["problem_text"])
        duplicate_of: str | None = None
        reason: str | None = None
        if record_id in seen_record_ids:
            duplicate_of, reason = seen_record_ids[record_id], "duplicate_record_id"
        elif problem_id in seen_problem_ids:
            duplicate_of, reason = seen_problem_ids[problem_id], "duplicate_problem_id"
        elif fingerprint in seen_problem_texts:
            duplicate_of = seen_problem_texts[fingerprint]
            reason = "exact_normalized_problem_text"
        if reason is not None:
            excluded_record = dict(record)
            excluded_record["freeze_exclusion"] = {
                "reason": reason,
                "duplicate_of": duplicate_of,
            }
            excluded.append(excluded_record)
            continue
        seen_record_ids[record_id] = record_id
        seen_problem_ids[problem_id] = record_id
        seen_problem_texts[fingerprint] = record_id
        unique.append(record)
    return unique, excluded


def _round_robin_sources(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    groups: dict[str, deque[dict[str, Any]]] = {}
    for record in records:
        dataset = _dataset(record)
        groups.setdefault(dataset, deque()).append(record)
    source_order = list(groups)
    ordered: list[dict[str, Any]] = []
    while any(groups.values()):
        for dataset in source_order:
            if groups[dataset]:
                ordered.append(groups[dataset].popleft())
    return ordered, source_order


def _near_duplicate_report(
    records: Sequence[dict[str, Any]],
    *,
    threshold: float,
    top_k: int,
) -> dict[str, Any]:
    normalized = [normalize_problem_text(row["task"]["problem_text"]) for row in records]
    terms = [set(text.split()) for text in normalized]
    pairs: list[dict[str, Any]] = []
    reported = 0
    for first_index, first in enumerate(records):
        first_terms = terms[first_index]
        for second_index in range(first_index + 1, len(records)):
            second_terms = terms[second_index]
            shorter = min(len(first_terms), len(second_terms))
            longer = max(len(first_terms), len(second_terms))
            if longer and shorter / longer < 0.6:
                continue
            union = first_terms | second_terms
            jaccard = len(first_terms & second_terms) / len(union) if union else 1.0
            if jaccard < max(0.0, threshold - 0.2):
                continue
            sequence_ratio = SequenceMatcher(
                None,
                normalized[first_index],
                normalized[second_index],
                autojunk=False,
            ).ratio()
            similarity = max(jaccard, sequence_ratio)
            if similarity < threshold:
                continue
            reported += 1
            pairs.append(
                {
                    "first_id": first["id"],
                    "second_id": records[second_index]["id"],
                    "first_source": _dataset(first),
                    "second_source": _dataset(records[second_index]),
                    "token_jaccard": jaccard,
                    "sequence_ratio": sequence_ratio,
                    "similarity": similarity,
                }
            )
    pairs.sort(
        key=lambda row: (
            -row["similarity"],
            row["first_id"],
            row["second_id"],
        )
    )
    return {
        "threshold": threshold,
        "reported_pairs": reported,
        "stored_top_k": min(top_k, len(pairs)),
        "pairs": pairs[:top_k],
    }


def _dataset(record: Mapping[str, Any]) -> str:
    return str(record["task"]["source"]["dataset"])


def _source_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(_dataset(record) for record in records).items()))


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
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
