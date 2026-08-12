"""Atomic normalization of successful DeepSeek teacher records."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .durable_io import open_text_append
from .records import (
    NORMALIZED_SCHEMA_VERSION,
    RAW_SCHEMA_VERSION,
    RecordValidationError,
    normalize_raw_record,
)

NORMALIZATION_ERROR_SCHEMA_VERSION = "deepseek.normalization.error.v1"


@dataclass(frozen=True, slots=True)
class NormalizeSummary:
    total: int
    normalized: int
    api_errors: int
    warnings: int


@dataclass(frozen=True, slots=True)
class AppendNormalizeSummary:
    total: int
    skipped: int
    normalized: int
    api_errors: int
    malformed: int
    warnings: int


def normalize_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> NormalizeSummary:
    """Normalize successful records and atomically publish one JSONL file.

    Provider error records remain in the raw audit log and are counted, but are
    intentionally omitted from the normalized output.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    total = 0
    normalized_count = 0
    api_errors = 0
    warnings = 0
    seen_ids: set[str] = set()

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_handle:
            temporary_path = Path(output_handle.name)
            with input_path.open("r", encoding="utf-8") as input_handle:
                for line_number, line in enumerate(input_handle, start=1):
                    if not line.strip():
                        continue
                    total += 1
                    record = _parse_object(input_path, line_number, line)
                    record_id = _record_id(input_path, line_number, record)
                    if record_id in seen_ids:
                        raise ValueError(f"{input_path}:{line_number}: duplicate id {record_id!r}")
                    seen_ids.add(record_id)

                    if record.get("schema_version") != RAW_SCHEMA_VERSION:
                        raise ValueError(
                            f"{input_path}:{line_number}: schema_version must be "
                            f"{RAW_SCHEMA_VERSION!r}"
                        )
                    status = record.get("status")
                    if status == "error":
                        _validate_error_record(record, input_path, line_number)
                        api_errors += 1
                        continue
                    if status != "ok":
                        raise ValueError(
                            f"{input_path}:{line_number}: status must be 'ok' or 'error'"
                        )

                    try:
                        normalized = normalize_raw_record(record)
                    except RecordValidationError as error:
                        raise ValueError(f"{input_path}:{line_number}: {error}") from error
                    output_handle.write(
                        json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    normalized_count += 1
                    warnings += len(normalized["validation"]["warnings"])

            output_handle.flush()
            os.fsync(output_handle.fileno())

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return NormalizeSummary(
        total=total,
        normalized=normalized_count,
        api_errors=api_errors,
        warnings=warnings,
    )


def normalize_jsonl_append(
    input_path: Path,
    output_path: Path,
    *,
    error_output_path: Path | None = None,
) -> AppendNormalizeSummary:
    """Append newly successful raw IDs while preserving existing normalized rows."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    error_output_path = Path(error_output_path) if error_output_path is not None else None
    raw_ids: set[str] = set()
    total = 0
    with input_path.open("r", encoding="utf-8") as input_handle:
        for line_number, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue
            record = _parse_object(input_path, line_number, line)
            record_id = _record_id(input_path, line_number, record)
            if record_id in raw_ids:
                raise ValueError(f"{input_path}:{line_number}: duplicate id {record_id!r}")
            raw_ids.add(record_id)
            if record.get("schema_version") != RAW_SCHEMA_VERSION:
                raise ValueError(
                    f"{input_path}:{line_number}: schema_version must be {RAW_SCHEMA_VERSION!r}"
                )
            status = record.get("status")
            if status == "error":
                _validate_error_record(record, input_path, line_number)
            elif status != "ok":
                raise ValueError(
                    f"{input_path}:{line_number}: status must be 'ok' or 'error'"
                )
            total += 1

    existing_ids: set[str] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as output_handle:
            for line_number, line in enumerate(output_handle, start=1):
                if not line.strip():
                    continue
                record = _parse_object(output_path, line_number, line)
                record_id = _record_id(output_path, line_number, record)
                if record_id in existing_ids:
                    raise ValueError(
                        f"{output_path}:{line_number}: duplicate id {record_id!r}"
                    )
                if record.get("schema_version") != NORMALIZED_SCHEMA_VERSION:
                    raise ValueError(
                        f"{output_path}:{line_number}: schema_version must be "
                        f"{NORMALIZED_SCHEMA_VERSION!r}"
                    )
                if record_id not in raw_ids:
                    raise ValueError(
                        f"{output_path}:{line_number}: normalized id has no raw record"
                    )
                existing_ids.add(record_id)

    error_ids: set[str] = set()
    if error_output_path is not None and error_output_path.exists():
        with error_output_path.open("r", encoding="utf-8") as error_handle:
            for line_number, line in enumerate(error_handle, start=1):
                if not line.strip():
                    continue
                record = _parse_object(error_output_path, line_number, line)
                record_id = _record_id(error_output_path, line_number, record)
                if record_id in error_ids:
                    raise ValueError(
                        f"{error_output_path}:{line_number}: duplicate id {record_id!r}"
                    )
                if record.get("schema_version") != NORMALIZATION_ERROR_SCHEMA_VERSION:
                    raise ValueError(
                        f"{error_output_path}:{line_number}: invalid schema_version"
                    )
                if record_id not in raw_ids:
                    raise ValueError(
                        f"{error_output_path}:{line_number}: error id has no raw record"
                    )
                if record_id in existing_ids:
                    raise ValueError(
                        f"{error_output_path}:{line_number}: id is both normalized and failed"
                    )
                error_ids.add(record_id)

    normalized_count = 0
    skipped = 0
    api_errors = 0
    malformed = 0
    warnings = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids: set[str] = set()
    with input_path.open("r", encoding="utf-8") as input_handle:
        for line_number, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue
            record = _parse_object(input_path, line_number, line)
            record_id = _record_id(input_path, line_number, record)
            if record_id not in raw_ids or record_id in processed_ids:
                raise ValueError(
                    f"{input_path}:{line_number}: input changed during normalization"
                )
            processed_ids.add(record_id)
            if record["status"] == "error":
                api_errors += 1
                continue
            if record_id in existing_ids or record_id in error_ids:
                skipped += 1
                continue
            try:
                normalized = normalize_raw_record(record)
            except RecordValidationError as error:
                if error_output_path is None:
                    raise ValueError(f"{input_path}:{line_number}: {error}") from error
                failure = {
                    "schema_version": NORMALIZATION_ERROR_SCHEMA_VERSION,
                    "id": record_id,
                    "failure_category": "malformed_trace",
                    "raw_line_number": line_number,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
                _append_jsonl(error_output_path, failure)
                error_ids.add(record_id)
                malformed += 1
                continue
            with output_path.open("a", encoding="utf-8", newline="\n") as output_handle:
                output_handle.write(
                    json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                output_handle.flush()
                os.fsync(output_handle.fileno())
            existing_ids.add(record_id)
            normalized_count += 1
            warnings += len(normalized["validation"]["warnings"])
    if processed_ids != raw_ids:
        raise ValueError(f"{input_path}: input changed during normalization")
    return AppendNormalizeSummary(
        total=total,
        skipped=skipped,
        normalized=normalized_count,
        api_errors=api_errors,
        malformed=malformed,
        warnings=warnings,
    )


def _parse_object(path: Path, line_number: int, line: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}:{line_number}: each JSONL value must be an object")
    return dict(value)


def _record_id(path: Path, line_number: int, record: Mapping[str, Any]) -> str:
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
    return record_id


def _validate_error_record(record: Mapping[str, Any], path: Path, line_number: int) -> None:
    request = record.get("request")
    error = record.get("error")
    if not isinstance(request, Mapping):
        raise ValueError(f"{path}:{line_number}: error record request must be an object")
    if not isinstance(request.get("messages"), list) or not request["messages"]:
        raise ValueError(f"{path}:{line_number}: request.messages must be a non-empty list")
    if not isinstance(request.get("generation_config"), Mapping):
        raise ValueError(f"{path}:{line_number}: request.generation_config must be an object")
    if not isinstance(error, Mapping):
        raise ValueError(f"{path}:{line_number}: error record error must be an object")
    if not isinstance(error.get("type"), str) or not error["type"]:
        raise ValueError(f"{path}:{line_number}: error.type must be a non-empty string")
    if not isinstance(error.get("message"), str):
        raise ValueError(f"{path}:{line_number}: error.message must be a string")


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with open_text_append(path) as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
