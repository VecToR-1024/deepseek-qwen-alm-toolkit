"""Deterministic eligibility rules for clean offline-ALM coding records."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .offline_teacher import OfflineTeacherTraceProvider, TeacherTraceError
from .records import ACTUAL_ONLY_TRACE_PROFILE
from .training_data_audit import record_source_name, response_style_features


CLEAN_ELIGIBILITY_SCHEMA_VERSION = "offline_alm.clean_eligibility.v1"


@dataclass(frozen=True, slots=True)
class CleanEligibilityPolicy:
    """Version-one formatting and preprocessing thresholds."""

    max_comment_line_ratio: float = 0.2
    max_sequence_length: int = 4096

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_comment_line_ratio <= 1.0:
            raise ValueError("max_comment_line_ratio must be between zero and one")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")


def evaluate_clean_eligibility(
    record: Mapping[str, Any],
    *,
    alm_diagnostic: Mapping[str, Any] | None,
    eos_supervised: bool,
    policy: CleanEligibilityPolicy | None = None,
) -> dict[str, Any]:
    """Return a stable, non-mutating eligibility decision for one record."""

    selected_policy = policy or CleanEligibilityPolicy()
    record_id = record.get("id")
    response_text = record.get("response_text")
    source_text = response_text if isinstance(response_text, str) else ""
    style = response_style_features(source_text)
    reasons: list[str] = []

    trace_valid, actual_positions = _validate_actual_trace(record)
    if not trace_valid:
        reasons.append("malformed_trace")

    trace_profile = _trace_profile(record)
    top_logprobs_complete, expected_top_logprobs = _validate_top_logprobs(record)
    if not top_logprobs_complete:
        reasons.append("incomplete_top_logprobs")

    finish_reason = record.get("finish_reason")
    if finish_reason == "length":
        reasons.append("finish_reason_length")
    elif finish_reason != "stop":
        reasons.append("finish_reason_not_stop")

    source_extraction = _mapping(
        _mapping(record.get("coding_verification")).get("source_extraction")
    )
    if style["has_code_fence"] or source_extraction.get(
        "removed_markdown_fence"
    ) is True:
        reasons.append("markdown_fence")

    syntax_ok, has_docstring, has_prose, has_top_level_assert = _source_contract(
        source_text
    )
    if not syntax_ok:
        reasons.append("syntax_error")
    else:
        if has_docstring:
            reasons.append("docstring")
        if has_prose or _starts_with_comment(source_text):
            reasons.append("prose_outside_code")
        if has_top_level_assert:
            reasons.append("benchmark_test_content")

    if style["comment_line_ratio"] > selected_policy.max_comment_line_ratio:
        reasons.append("comment_ratio_above_limit")

    verification = _mapping(record.get("coding_verification"))
    verification_category = verification.get("failure_category")
    verification_status = verification.get("status")
    if (
        verification_category != "passed"
        or verification_status not in (None, "accepted", "passed")
    ):
        reasons.append("official_test_failure")

    diagnostic = _mapping(alm_diagnostic)
    diagnostic_matches = bool(diagnostic) and diagnostic.get("id") == record_id
    if not diagnostic_matches:
        reasons.append("alm_preprocessing_failure")
        sequence_length = None
        valid_chunks = None
        boundary_drops = None
    else:
        sequence_length = _integer(diagnostic.get("sequence_length"))
        valid_chunks = _integer(diagnostic.get("valid_alm_chunks"))
        boundary_drops = _integer(
            diagnostic.get("prompt_completion_boundary_drops")
        )
        if sequence_length is None or valid_chunks is None or boundary_drops is None:
            reasons.append("alm_preprocessing_failure")
        else:
            if sequence_length > selected_policy.max_sequence_length:
                reasons.append("sequence_over_4096")
            if valid_chunks <= 0:
                reasons.append("zero_alm_chunks")
            if boundary_drops > 0:
                reasons.append("boundary_drop")

    if eos_supervised is not True:
        reasons.append("eos_not_supervised")

    return {
        "schema_version": CLEAN_ELIGIBILITY_SCHEMA_VERSION,
        "id": record_id,
        "source": record_source_name(record),
        "eligible": not reasons,
        "reasons": reasons,
        "raw_trace_preserved": True,
        "policy": {
            "max_comment_line_ratio": selected_policy.max_comment_line_ratio,
            "max_sequence_length": selected_policy.max_sequence_length,
        },
        "trace": {
            "actual_trace_valid": trace_valid,
            "actual_positions": actual_positions,
            "trace_profile": trace_profile,
            "expected_top_logprobs": expected_top_logprobs,
            "top_logprobs_complete": top_logprobs_complete,
        },
        "style": style,
        "verification_status": verification_status,
        "verification_category": verification_category,
        "finish_reason": finish_reason,
        "alm": {
            "sequence_length": sequence_length,
            "valid_alm_chunks": valid_chunks,
            "prompt_completion_boundary_drops": boundary_drops,
        },
        "eos_supervised": eos_supervised is True,
    }


def _validate_actual_trace(record: Mapping[str, Any]) -> tuple[bool, int]:
    rows = record.get("content_tokens")
    actual_positions = len(rows) if isinstance(rows, list) else 0
    try:
        OfflineTeacherTraceProvider().get_trace(record)
    except (TeacherTraceError, ValueError):
        return False, actual_positions
    return True, actual_positions


def _validate_top_logprobs(record: Mapping[str, Any]) -> tuple[bool, int | None]:
    request = _mapping(record.get("request"))
    generation = _mapping(request.get("generation_config"))
    profile = generation.get("trace_profile")
    rows = record.get("content_tokens")
    if profile == ACTUAL_ONLY_TRACE_PROFILE:
        if not isinstance(rows, list) or not rows:
            return False, 0
        return (
            all(
                isinstance(row, Mapping)
                and isinstance(row.get("top_logprobs"), list)
                and not row["top_logprobs"]
                for row in rows
            ),
            0,
        )
    if profile is not None:
        return False, None
    expected = _integer(generation.get("top_logprobs"))
    if expected is None or expected <= 0 or not isinstance(rows, list) or not rows:
        return False, expected
    for row in rows:
        if not isinstance(row, Mapping):
            return False, expected
        candidates = row.get("top_logprobs")
        if not isinstance(candidates, list) or len(candidates) < expected:
            return False, expected
        if not all(_valid_top_candidate(candidate) for candidate in candidates):
            return False, expected
    return True, expected


def _trace_profile(record: Mapping[str, Any]) -> str:
    request = _mapping(record.get("request"))
    generation = _mapping(request.get("generation_config"))
    profile = generation.get("trace_profile")
    if isinstance(profile, str) and profile:
        return profile
    expected = _integer(generation.get("top_logprobs"))
    return "top20" if expected == 20 else "legacy_topk"


def _valid_top_candidate(candidate: Any) -> bool:
    if not isinstance(candidate, Mapping):
        return False
    byte_values = candidate.get("bytes")
    logprob = candidate.get("logprob")
    return (
        isinstance(byte_values, list)
        and bool(byte_values)
        and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= 255
            for value in byte_values
        )
        and isinstance(logprob, (int, float))
        and not isinstance(logprob, bool)
        and math.isfinite(float(logprob))
        and float(logprob) <= 1e-7
    )


def _source_contract(source: str) -> tuple[bool, bool, bool, bool]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False, False, False, False

    has_docstring = any(
        ast.get_docstring(node, clean=False) is not None
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        )
    )
    string_expressions = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    module_docstring = (
        tree.body[0]
        if tree.body and tree.body[0] in string_expressions
        else None
    )
    has_prose = any(node is not module_docstring for node in string_expressions)
    has_top_level_assert = any(isinstance(node, ast.Assert) for node in tree.body)
    return True, has_docstring, has_prose, has_top_level_assert


def _starts_with_comment(source: str) -> bool:
    return next(
        (line.lstrip().startswith("#") for line in source.splitlines() if line.strip()),
        False,
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
