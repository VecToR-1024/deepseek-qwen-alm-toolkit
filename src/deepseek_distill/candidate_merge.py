"""Deterministic, memory-bounded merging of verified training candidates."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .offline_teacher import OfflineTeacherTraceProvider, TeacherTraceError
from .multisource_tasks import (
    MULTISOURCE_TASK_SCHEMA_VERSION,
    multisource_dataset_slug,
)
from .records import NORMALIZED_SCHEMA_VERSION
from .rejection_sampling import publish_json_once


MERGE_SCHEMA_VERSION = "coding.training_candidates.merged.v3"
EXCLUSION_SCHEMA_VERSION = "coding.training_candidate.exclusion.v1"
_SUPPORTED_TASK_SCHEMAS = frozenset(
    {
        "coding.task.mbpp.v1",
        "coding.task.taco.v1",
        "coding.task.taco.length_retry.v2",
        MULTISOURCE_TASK_SCHEMA_VERSION,
    }
)


@dataclass(frozen=True, slots=True)
class CandidateSource:
    """One immutable accepted-candidate artifact in merge order."""

    label: str
    path: Path
    expected_records: int


def build_merged_candidate_outputs(
    *,
    sources: Sequence[CandidateSource],
    output_dir: Path,
    alm_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge verified records and publish an explicit max-length subset."""
    checked_sources = _validate_sources(sources)
    max_length = alm_diagnostics.get("max_length")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("ALM diagnostics max_length must be a positive integer")

    record_meta: dict[str, dict[str, Any]] = {}
    problem_ids: set[str] = set()
    ordered_ids: list[str] = []
    source_summaries: list[dict[str, Any]] = []
    benchmark_counts: dict[str, int] = {}
    trace_provider = OfflineTeacherTraceProvider()

    for source in checked_sources:
        source_count = 0
        source_ids: list[str] = []
        for line_number, record in _read_jsonl_stream(source.path):
            record_id, problem_id, benchmark = _validate_candidate_record(
                record,
                source=source,
                line_number=line_number,
                trace_provider=trace_provider,
            )
            if record_id in record_meta:
                raise ValueError(f"duplicate record id {record_id!r}")
            if problem_id in problem_ids:
                raise ValueError(f"duplicate problem id {problem_id!r}")
            problem_ids.add(problem_id)
            record_meta[record_id] = {
                "problem_id": problem_id,
                "source_label": source.label,
                "benchmark": benchmark,
            }
            ordered_ids.append(record_id)
            source_ids.append(record_id)
            source_count += 1
            benchmark_counts[benchmark] = benchmark_counts.get(benchmark, 0) + 1
        if source_count != source.expected_records:
            raise ValueError(
                f"{source.label}: expected {source.expected_records} records, "
                f"found {source_count}"
            )
        source_summaries.append(
            {
                "label": source.label,
                "path": source.path.as_posix(),
                "records": source_count,
                "sha256": _sha256(source.path),
                "ordered_record_ids_sha256": _ordered_ids_hash(source_ids),
            }
        )

    exclusions, example_by_id = _build_exclusions(
        alm_diagnostics=alm_diagnostics,
        ordered_ids=ordered_ids,
        max_length=max_length,
    )
    excluded_ids = set(exclusions)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_path = output_dir / "all_candidates.jsonl"
    trainable_path = output_dir / f"trainable_max{max_length}.jsonl"
    excluded_path = output_dir / "excluded_records.jsonl"
    all_status = _publish_jsonl_stream_once(
        all_path,
        iter_candidate_records(checked_sources),
    )
    trainable_status = _publish_jsonl_stream_once(
        trainable_path,
        (
            record
            for record in iter_candidate_records(checked_sources)
            if record["id"] not in excluded_ids
        ),
    )
    excluded_status = _publish_jsonl_stream_once(
        excluded_path,
        (
            _exclusion_record(
                record_id=record_id,
                meta=record_meta[record_id],
                reasons=exclusions[record_id],
                example=example_by_id.get(record_id),
            )
            for record_id in ordered_ids
            if record_id in excluded_ids
        ),
    )

    trainable_count = len(ordered_ids) - len(excluded_ids)
    trainable_key = f"trainable_max{max_length}"
    summary = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "source_order": [source.label for source in checked_sources],
        "sources": source_summaries,
        "student_tokenizer": alm_diagnostics.get("student_tokenizer"),
        "student_revision": alm_diagnostics.get("student_revision"),
        "max_length": max_length,
        "counts": {
            "all_candidates": len(ordered_ids),
            trainable_key: trainable_count,
            "excluded": len(excluded_ids),
            **benchmark_counts,
        },
        "duplicates": {"record_ids": 0, "problem_ids": 0},
        "exclusion_counts": _exclusion_counts(exclusions),
        "ordered_record_ids_sha256": _ordered_ids_hash(ordered_ids),
        "outputs": {
            "all_candidates": {
                "path": all_path.name,
                "records": len(ordered_ids),
                "sha256": _sha256(all_path),
            },
            trainable_key: {
                "path": trainable_path.name,
                "records": trainable_count,
                "sha256": _sha256(trainable_path),
            },
            "excluded_records": {
                "path": excluded_path.name,
                "records": len(excluded_ids),
                "sha256": _sha256(excluded_path),
            },
            "alm_diagnostics": {"path": "alm_diagnostics.json"},
        },
        "alm": {
            "preprocessing_errors": len(
                alm_diagnostics.get("preprocessing_errors") or []
            ),
            "zero_valid_chunks": len(
                alm_diagnostics.get("examples_with_zero_valid_chunks") or []
            ),
            "records_exceeding_max_length": len(
                alm_diagnostics.get("records_exceeding_max_length") or []
            ),
            "sequence_length_distribution": alm_diagnostics.get(
                "sequence_length_distribution"
            ),
            "chunks_per_example_distribution": alm_diagnostics.get(
                "chunks_per_example_distribution"
            ),
            "group_counts": alm_diagnostics.get("group_counts"),
            "prompt_completion_boundary_drops": alm_diagnostics.get(
                "prompt_completion_boundary_drops"
            ),
        },
    }
    publish_json_once(output_dir / "alm_diagnostics.json", alm_diagnostics)
    publish_json_once(output_dir / "merge_manifest.json", summary)
    return {
        **summary,
        "write_status": {
            "all_candidates": all_status,
            trainable_key: trainable_status,
            "excluded_records": excluded_status,
        },
    }


def _validate_sources(sources: Sequence[CandidateSource]) -> tuple[CandidateSource, ...]:
    if not sources:
        raise ValueError("at least one candidate source is required")
    labels: set[str] = set()
    checked: list[CandidateSource] = []
    for position, source in enumerate(sources):
        if not isinstance(source, CandidateSource):
            raise ValueError(f"candidate source {position} has an invalid type")
        label = source.label.strip()
        if not label:
            raise ValueError(f"candidate source {position} has an empty label")
        if label in labels:
            raise ValueError(f"duplicate candidate source label {label!r}")
        if (
            isinstance(source.expected_records, bool)
            or not isinstance(source.expected_records, int)
            or source.expected_records <= 0
        ):
            raise ValueError(f"{label}: expected_records must be a positive integer")
        path = Path(source.path)
        if not path.is_file():
            raise FileNotFoundError(path)
        labels.add(label)
        checked.append(CandidateSource(label, path, source.expected_records))
    return tuple(checked)


def _validate_candidate_record(
    record: Mapping[str, Any],
    *,
    source: CandidateSource,
    line_number: int,
    trace_provider: OfflineTeacherTraceProvider,
) -> tuple[str, str, str]:
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"{source.path}:{line_number}: record id is missing")
    if record.get("schema_version") != NORMALIZED_SCHEMA_VERSION:
        raise ValueError(
            f"{source.path}:{line_number}: unsupported normalized schema"
        )
    validation = record.get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("content_bytes_match") is not True
    ):
        raise ValueError(
            f"{source.path}:{line_number}: trace reconstruction is not valid"
        )
    try:
        trace_provider.get_trace(record)
    except TeacherTraceError as error:
        raise ValueError(f"{source.path}:{line_number}: {error}") from error

    task = record.get("task")
    if not isinstance(task, Mapping):
        raise ValueError(f"{source.path}:{line_number}: task is missing")
    task_schema = task.get("schema_version")
    if task_schema not in _SUPPORTED_TASK_SCHEMAS:
        raise ValueError(
            f"{source.path}:{line_number}: unsupported task schema {task_schema!r}"
        )
    sampling = record.get("sampling")
    problem_id = (
        sampling.get("problem_id")
        if isinstance(sampling, Mapping)
        else task.get("id")
    )
    if not isinstance(problem_id, str) or not problem_id:
        problem_id = task.get("id")
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError(f"{source.path}:{line_number}: problem id is missing")
    if task_schema == "coding.task.mbpp.v1":
        benchmark = "mbpp"
    elif task_schema == MULTISOURCE_TASK_SCHEMA_VERSION:
        try:
            benchmark = multisource_dataset_slug(task)
        except ValueError as error:
            raise ValueError(f"{source.path}:{line_number}: {error}") from error
    else:
        benchmark = "taco"
    return record_id, problem_id, benchmark


def _build_exclusions(
    *,
    alm_diagnostics: Mapping[str, Any],
    ordered_ids: Sequence[str],
    max_length: int,
) -> tuple[dict[str, list[str]], dict[str, Mapping[str, Any]]]:
    candidate_ids = set(ordered_ids)
    example_by_id: dict[str, Mapping[str, Any]] = {}
    for position, example in enumerate(alm_diagnostics.get("examples") or []):
        if not isinstance(example, Mapping):
            raise ValueError(f"ALM example {position} must be an object")
        record_id = example.get("id")
        if not isinstance(record_id, str) or record_id not in candidate_ids:
            raise ValueError(f"ALM example {position} has an unknown id")
        if record_id in example_by_id:
            raise ValueError(f"duplicate ALM example id {record_id!r}")
        example_by_id[record_id] = example

    error_ids: set[str] = set()
    for position, error in enumerate(alm_diagnostics.get("preprocessing_errors") or []):
        if not isinstance(error, Mapping):
            raise ValueError(f"ALM preprocessing error {position} must be an object")
        record_id = error.get("id")
        if not isinstance(record_id, str) or record_id not in candidate_ids:
            raise ValueError(f"ALM preprocessing error {position} has an unknown id")
        error_ids.add(record_id)

    if set(example_by_id) & error_ids:
        raise ValueError("an ALM id cannot be both successful and failed")
    covered_ids = set(example_by_id) | error_ids
    if covered_ids != candidate_ids:
        missing = sorted(candidate_ids - covered_ids)
        extra = sorted(covered_ids - candidate_ids)
        raise ValueError(
            f"ALM diagnostics do not cover candidate ids; missing={missing!r}, "
            f"extra={extra!r}"
        )

    zero_ids = _diagnostic_id_set(
        alm_diagnostics.get("examples_with_zero_valid_chunks"),
        label="zero-chunk",
        candidate_ids=candidate_ids,
    )
    over_limit_ids = _diagnostic_id_set(
        alm_diagnostics.get("records_exceeding_max_length"),
        label="over-limit",
        candidate_ids=candidate_ids,
    )
    for record_id in over_limit_ids:
        sequence_length = example_by_id.get(record_id, {}).get("sequence_length")
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or sequence_length <= max_length
        ):
            raise ValueError(
                f"over-limit record {record_id!r} has invalid sequence length"
            )

    exclusions: dict[str, list[str]] = {}
    for record_id in ordered_ids:
        reasons: list[str] = []
        if record_id in error_ids:
            reasons.append("alm_preprocessing_error")
        if record_id in zero_ids:
            reasons.append("zero_valid_alm_chunks")
        if record_id in over_limit_ids:
            reasons.append(f"sequence_length_exceeds_{max_length}")
        if reasons:
            exclusions[record_id] = reasons
    return exclusions, example_by_id


def _diagnostic_id_set(
    values: Any,
    *,
    label: str,
    candidate_ids: set[str],
) -> set[str]:
    if values is None:
        return set()
    if not isinstance(values, list):
        raise ValueError(f"ALM {label} ids must be a list")
    identifiers: set[str] = set()
    for value in values:
        if not isinstance(value, str) or value not in candidate_ids:
            raise ValueError(f"ALM {label} list contains an unknown id")
        if value in identifiers:
            raise ValueError(f"ALM {label} list contains duplicate id {value!r}")
        identifiers.add(value)
    return identifiers


def iter_candidate_records(
    sources: Sequence[CandidateSource],
) -> Iterator[dict[str, Any]]:
    """Yield candidate records in the declared deterministic source order."""
    for source in sources:
        for _, record in _read_jsonl_stream(source.path):
            yield record


def _exclusion_record(
    *,
    record_id: str,
    meta: Mapping[str, Any],
    reasons: list[str],
    example: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "id": record_id,
        "problem_id": meta["problem_id"],
        "source_label": meta["source_label"],
        "reasons": list(reasons),
        "sequence_length": example.get("sequence_length") if example else None,
        "valid_alm_chunks": example.get("valid_alm_chunks") if example else None,
    }


def _exclusion_counts(exclusions: Mapping[str, Sequence[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reasons in exclusions.values():
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _read_jsonl_stream(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
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
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield line_number, value


def _publish_jsonl_stream_once(
    path: Path,
    records: Iterable[Mapping[str, Any]],
) -> str:
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as existing:
            for record in records:
                expected = _serialize_jsonl(record)
                actual = existing.readline()
                if actual != expected:
                    raise FileExistsError(
                        f"{path} already exists with different content"
                    )
            if existing.read():
                raise FileExistsError(f"{path} already contains extra records")
        return "unchanged"

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(_serialize_jsonl(record))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return "created"


def _serialize_jsonl(record: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(record),
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"


def _ordered_ids_hash(identifiers: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
