"""Streaming structural validation for raw and normalized teacher JSONL."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .alignment_pipeline import ALIGNED_SCHEMA_VERSION
from .normalize import _validate_error_record
from .records import (
    NORMALIZED_SCHEMA_VERSION,
    RAW_SCHEMA_VERSION,
    RecordValidationError,
    normalize_raw_record,
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    line: int
    code: str
    message: str
    record_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    total_lines: int
    valid_records: int
    invalid_records: int
    api_errors: int
    warnings: int
    schema_versions: dict[str, int]
    issues: list[ValidationIssue]


def validate_jsonl(path: Path) -> ValidationReport:
    """Validate every non-blank JSONL record without stopping at the first error."""
    path = Path(path)
    total_lines = 0
    valid_records = 0
    invalid_records = 0
    api_errors = 0
    warnings = 0
    schema_versions: dict[str, int] = {}
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_lines += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                invalid_records += 1
                issues.append(
                    ValidationIssue(line_number, "invalid_json", error.msg)
                )
                continue
            if not isinstance(value, Mapping):
                invalid_records += 1
                issues.append(
                    ValidationIssue(line_number, "non_object", "JSONL value must be an object")
                )
                continue

            record = dict(value)
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                invalid_records += 1
                issues.append(
                    ValidationIssue(line_number, "missing_id", "id must be a non-empty string")
                )
                continue
            if record_id in seen_ids:
                invalid_records += 1
                issues.append(
                    ValidationIssue(
                        line_number,
                        "duplicate_id",
                        f"duplicate id {record_id!r}",
                        record_id,
                    )
                )
                continue
            seen_ids.add(record_id)

            schema_version = record.get("schema_version")
            schema_key = schema_version if isinstance(schema_version, str) else "<missing>"
            schema_versions[schema_key] = schema_versions.get(schema_key, 0) + 1

            try:
                record_warnings, is_api_error = _validate_record(
                    record,
                    path,
                    line_number,
                )
            except (RecordValidationError, ValueError) as error:
                invalid_records += 1
                issues.append(
                    ValidationIssue(line_number, "invalid_record", str(error), record_id)
                )
                continue

            valid_records += 1
            warnings += record_warnings
            api_errors += is_api_error

    return ValidationReport(
        total_lines=total_lines,
        valid_records=valid_records,
        invalid_records=invalid_records,
        api_errors=api_errors,
        warnings=warnings,
        schema_versions=schema_versions,
        issues=issues,
    )


def _validate_record(
    record: Mapping[str, Any],
    path: Path,
    line_number: int,
) -> tuple[int, int]:
    schema_version = record.get("schema_version")
    if schema_version == RAW_SCHEMA_VERSION:
        status = record.get("status")
        if status == "error":
            _validate_error_record(record, path, line_number)
            return 0, 1
        if status != "ok":
            raise RecordValidationError("status must be 'ok' or 'error'")
        normalized = normalize_raw_record(record)
        return len(normalized["validation"]["warnings"]), 0
    if schema_version == NORMALIZED_SCHEMA_VERSION:
        return _validate_normalized_record(record), 0
    if schema_version == ALIGNED_SCHEMA_VERSION:
        return _validate_aligned_record(record), 0
    raise RecordValidationError(f"unsupported schema_version {schema_version!r}")


def _validate_normalized_record(record: Mapping[str, Any]) -> int:
    if not isinstance(record.get("response_text"), str):
        raise RecordValidationError("response_text must be a string")
    if not isinstance(record.get("content_tokens"), list):
        raise RecordValidationError("content_tokens must be a list")
    validation = record.get("validation")
    if not isinstance(validation, Mapping):
        raise RecordValidationError("validation must be an object")
    warnings = validation.get("warnings")
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise RecordValidationError("validation.warnings must be a list of strings")
    return len(warnings)


def _validate_aligned_record(record: Mapping[str, Any]) -> int:
    warnings = _validate_normalized_record(record)
    if record.get("source_normalized_schema_version") != NORMALIZED_SCHEMA_VERSION:
        raise RecordValidationError(
            f"source_normalized_schema_version must be {NORMALIZED_SCHEMA_VERSION!r}"
        )
    if not isinstance(record.get("student_model"), str) or not record["student_model"]:
        raise RecordValidationError("student_model must be a non-empty string")
    tokenizer = record.get("student_tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise RecordValidationError("student_tokenizer must be an object")
    if not isinstance(tokenizer.get("name_or_path"), str) or not tokenizer["name_or_path"]:
        raise RecordValidationError("student_tokenizer.name_or_path must be a non-empty string")
    if tokenizer.get("revision") is not None and not isinstance(tokenizer["revision"], str):
        raise RecordValidationError("student_tokenizer.revision must be a string or null")

    context_ids = _non_negative_int_list(
        record.get("student_generation_context_ids"),
        "student_generation_context_ids",
    )
    input_ids = _non_negative_int_list(record.get("student_input_ids"), "student_input_ids")
    if not context_ids:
        raise RecordValidationError("student_generation_context_ids must not be empty")
    if len(input_ids) < 2:
        raise RecordValidationError("student_input_ids must contain at least two tokens")

    soft_positions = record.get("soft_positions")
    if not isinstance(soft_positions, list):
        raise RecordValidationError("soft_positions must be a list")
    teacher_positions: list[int] = []
    logit_positions: list[int] = []
    mapped_candidate_count = 0
    content_tokens = record["content_tokens"]
    for index, value in enumerate(soft_positions):
        context = f"soft_positions[{index}]"
        if not isinstance(value, Mapping):
            raise RecordValidationError(f"{context} must be an object")
        teacher_position = _non_negative_int(value.get("teacher_position"), f"{context}.teacher_position")
        logit_position = _non_negative_int(
            value.get("student_logit_position"),
            f"{context}.student_logit_position",
        )
        if teacher_position >= len(content_tokens):
            raise RecordValidationError(f"{context}.teacher_position is outside content_tokens")
        if logit_position >= len(input_ids) - 1:
            raise RecordValidationError(f"{context}.student_logit_position cannot predict a next token")

        token_ids = _non_negative_int_list(
            value.get("mapped_student_token_ids"),
            f"{context}.mapped_student_token_ids",
        )
        if not token_ids or len(set(token_ids)) != len(token_ids):
            raise RecordValidationError(
                f"{context}.mapped_student_token_ids must be non-empty and unique"
            )
        probabilities = value.get("teacher_probs")
        if not isinstance(probabilities, list) or len(probabilities) != len(token_ids):
            raise RecordValidationError(f"{context}.teacher_probs must match mapped token IDs")
        normalized_probabilities = [
            _probability_value(probability, f"{context}.teacher_probs[{probability_index}]")
            for probability_index, probability in enumerate(probabilities)
        ]
        tail = _probability_value(value.get("teacher_tail_prob"), f"{context}.teacher_tail_prob")
        if not math.isclose(math.fsum(normalized_probabilities) + tail, 1.0, abs_tol=1e-6):
            raise RecordValidationError(f"{context} teacher probabilities plus tail must sum to one")
        teacher_tokens = value.get("mapped_teacher_tokens")
        if (
            not isinstance(teacher_tokens, list)
            or len(teacher_tokens) != len(token_ids)
            or any(not isinstance(token, str) for token in teacher_tokens)
        ):
            raise RecordValidationError(f"{context}.mapped_teacher_tokens must match mapped token IDs")

        teacher_positions.append(teacher_position)
        logit_positions.append(logit_position)
        mapped_candidate_count += len(token_ids)

    if teacher_positions != sorted(set(teacher_positions)):
        raise RecordValidationError("soft_positions teacher positions must be unique and increasing")
    if logit_positions != sorted(set(logit_positions)):
        raise RecordValidationError("soft_positions student logit positions must be unique and increasing")

    stats = record.get("alignment_stats")
    if not isinstance(stats, Mapping):
        raise RecordValidationError("alignment_stats must be an object")
    aligned_count = _non_negative_int(stats.get("aligned_positions"), "alignment_stats.aligned_positions")
    mapped_count = _non_negative_int(stats.get("mapped_candidates"), "alignment_stats.mapped_candidates")
    total_teacher_positions = _non_negative_int(
        stats.get("teacher_positions"),
        "alignment_stats.teacher_positions",
    )
    if aligned_count != len(soft_positions):
        raise RecordValidationError("alignment_stats.aligned_positions does not match soft_positions")
    if mapped_count != mapped_candidate_count:
        raise RecordValidationError("alignment_stats.mapped_candidates does not match soft_positions")
    if total_teacher_positions != len(content_tokens):
        raise RecordValidationError("alignment_stats.teacher_positions does not match content_tokens")
    return warnings


def _non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecordValidationError(f"{context} must be a non-negative integer")
    return value


def _non_negative_int_list(value: Any, context: str) -> list[int]:
    if not isinstance(value, list):
        raise RecordValidationError(f"{context} must be a list")
    return [_non_negative_int(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _probability_value(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordValidationError(f"{context} must be a number")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RecordValidationError(f"{context} must be a finite probability between zero and one")
    return probability
