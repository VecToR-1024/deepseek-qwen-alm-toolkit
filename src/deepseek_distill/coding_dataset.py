"""Build accepted and rejected coding candidates from durable pipeline stages."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .offline_teacher import OfflineTeacherTraceProvider, TeacherTraceError


REJECTED_SCHEMA_VERSION = "coding.candidate.rejected.v1"


@dataclass(frozen=True, slots=True)
class CandidateDatasetSummary:
    total: int
    accepted: int
    rejected: int
    failure_counts: dict[str, int]


def build_candidate_datasets(
    *,
    raw_path: Path,
    normalized_path: Path,
    verifier_path: Path,
    accepted_path: Path,
    rejected_path: Path,
) -> CandidateDatasetSummary:
    """Join stages by ID and publish passed-only and rejected JSONL atomically."""
    if accepted_path.exists():
        raise FileExistsError(accepted_path)
    if rejected_path.exists():
        raise FileExistsError(rejected_path)
    raw_records = _unique_records(raw_path)
    normalized_records = _unique_records(normalized_path)
    verifier_records = _unique_records(verifier_path)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    trace_provider = OfflineTeacherTraceProvider()
    for record_id, raw in raw_records.items():
        normalized = normalized_records.get(record_id)
        verification = verifier_records.get(record_id)
        category = _candidate_category(raw, normalized, verification)
        if category == "passed" and normalized is not None and verification is not None:
            try:
                trace_provider.get_trace(normalized)
            except (TeacherTraceError, ValueError):
                category = "malformed_trace"
            else:
                candidate = copy.deepcopy(normalized)
                candidate["coding_verification"] = copy.deepcopy(verification)
                accepted.append(candidate)
        if category != "passed":
            failure_counts[category] += 1
            rejected.append(
                {
                    "schema_version": REJECTED_SCHEMA_VERSION,
                    "id": record_id,
                    "failure_category": category,
                    "raw_record": copy.deepcopy(raw),
                    "normalized_record": copy.deepcopy(normalized),
                    "verification": copy.deepcopy(verification),
                }
            )

    extra_normalized = sorted(set(normalized_records) - set(raw_records))
    extra_verifier = sorted(set(verifier_records) - set(raw_records))
    if extra_normalized or extra_verifier:
        raise ValueError(
            "normalized/verifier IDs without raw attempts: "
            + ", ".join([*extra_normalized, *extra_verifier])
        )
    _write_jsonl_pair_atomic(
        accepted_path=accepted_path,
        accepted=accepted,
        rejected_path=rejected_path,
        rejected=rejected,
    )
    return CandidateDatasetSummary(
        total=len(raw_records),
        accepted=len(accepted),
        rejected=len(rejected),
        failure_counts=dict(sorted(failure_counts.items())),
    )


def _candidate_category(
    raw: Mapping[str, Any],
    normalized: Mapping[str, Any] | None,
    verification: Mapping[str, Any] | None,
) -> str:
    if raw.get("status") == "error":
        return "api_error"
    if raw.get("status") != "ok" or normalized is None:
        return "malformed_trace"
    validation = normalized.get("validation")
    if not isinstance(validation, Mapping) or validation.get("content_bytes_match") is not True:
        return "malformed_trace"
    if verification is None:
        return "malformed_trace"
    category = verification.get("failure_category")
    if not isinstance(category, str) or not category:
        return "malformed_trace"
    return category


def _unique_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            record_id = value.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
            if record_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id!r}")
            records[record_id] = value
    return records


def _write_jsonl_pair_atomic(
    *,
    accepted_path: Path,
    accepted: list[dict[str, Any]],
    rejected_path: Path,
    rejected: list[dict[str, Any]],
) -> None:
    accepted_temp = _write_jsonl_temp(accepted_path, accepted)
    try:
        rejected_temp = _write_jsonl_temp(rejected_path, rejected)
    except BaseException:
        os.unlink(accepted_temp)
        raise
    try:
        os.replace(accepted_temp, accepted_path)
        os.replace(rejected_temp, rejected_path)
    except BaseException:
        for temporary in (accepted_temp, rejected_temp):
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def _write_jsonl_temp(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return temporary
