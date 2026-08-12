"""Validation and normalization for persisted DeepSeek chat completions.

DeepSeek response fields follow the official Chat Completion schema:
https://api-docs.deepseek.com/api/create-chat-completion/
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

RAW_SCHEMA_VERSION = "deepseek.teacher.raw.v1"
NORMALIZED_SCHEMA_VERSION = "deepseek.teacher.normalized.v1"
SENTINEL_LOGPROB = -9999.0
ACTUAL_ONLY_TRACE_PROFILE = "actual_only"


class RecordValidationError(ValueError):
    """Raised when a persisted teacher record violates the phase-one contract."""


def normalize_raw_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one successful raw response record into the normalized schema."""
    if raw.get("schema_version") != RAW_SCHEMA_VERSION:
        raise RecordValidationError(f"schema_version must be {RAW_SCHEMA_VERSION!r}")
    if raw.get("status") != "ok":
        raise RecordValidationError("raw record status is not ok")

    record_id = _required_string(raw, "id", "raw record")
    request = _required_mapping(raw, "request", "raw record")
    response = _required_mapping(raw, "response", "raw record")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RecordValidationError("request.messages must be a non-empty list")
    generation_config = _required_mapping(request, "generation_config", "request")
    prompt_contract = request.get("prompt_contract")
    if prompt_contract is not None:
        prompt_contract = _mapping_value(
            prompt_contract,
            "request.prompt_contract",
        )
        for key in ("id", "interface_type"):
            _required_string(prompt_contract, key, "request.prompt_contract")

    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RecordValidationError("response.choices must contain exactly one choice")
    choice = _mapping_value(choices[0], "response.choices[0]")
    message = _required_mapping(choice, "message", "response.choices[0]")
    logprobs = _required_mapping(choice, "logprobs", "response.choices[0]")

    response_text = message.get("content")
    if not isinstance(response_text, str):
        raise RecordValidationError("response message content must be a string")
    reasoning_text = message.get("reasoning_content")
    if reasoning_text is not None and not isinstance(reasoning_text, str):
        raise RecordValidationError("reasoning_content must be a string or null")

    trace_profile = generation_config.get("trace_profile")
    if trace_profile is None:
        requested_top_k = generation_config.get("top_logprobs", 20)
        candidates_optional = False
    elif trace_profile == ACTUAL_ONLY_TRACE_PROFILE:
        if "top_logprobs" in generation_config:
            raise RecordValidationError(
                "actual_only generation_config must omit top_logprobs"
            )
        requested_top_k = 0
        candidates_optional = True
    else:
        raise RecordValidationError(
            f"unsupported generation_config.trace_profile {trace_profile!r}"
        )
    if not isinstance(requested_top_k, int) or not 0 <= requested_top_k <= 20:
        raise RecordValidationError("generation_config.top_logprobs must be between 0 and 20")

    warnings: list[str] = []
    content_tokens = _normalize_token_rows(
        logprobs.get("content"),
        field_name="content",
        requested_top_k=requested_top_k,
        candidates_optional=candidates_optional,
    )
    reasoning_tokens = _normalize_token_rows(
        logprobs.get("reasoning_content"),
        field_name="reasoning_content",
        requested_top_k=requested_top_k,
        candidates_optional=candidates_optional,
        nullable=True,
    )
    content_bytes_match = _compare_token_bytes(content_tokens, response_text, "content", warnings)
    reasoning_bytes_match = _compare_token_bytes(
        reasoning_tokens,
        reasoning_text,
        "reasoning_content",
        warnings,
    )

    normalized = {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "id": record_id,
        "source_raw_schema_version": RAW_SCHEMA_VERSION,
        "collected_at": raw.get("collected_at"),
        "request": {
            "model": request.get("model"),
            "messages": copy.deepcopy(messages),
            "generation_config": copy.deepcopy(dict(generation_config)),
        },
        "api_response_id": response.get("id"),
        "teacher_model": response.get("model") or request.get("model"),
        "system_fingerprint": response.get("system_fingerprint"),
        "finish_reason": choice.get("finish_reason"),
        "response_text": response_text,
        "reasoning_text": reasoning_text,
        "content_tokens": content_tokens,
        "reasoning_tokens": reasoning_tokens,
        "usage": copy.deepcopy(response.get("usage") or {}),
        "validation": {
            "content_bytes_match": content_bytes_match,
            "reasoning_bytes_match": reasoning_bytes_match,
            "warnings": warnings,
        },
    }
    if prompt_contract is not None:
        normalized["request"]["prompt_contract"] = copy.deepcopy(
            dict(prompt_contract)
        )
    for optional_field in ("task", "provider", "metrics"):
        value = raw.get(optional_field)
        if value is not None:
            if not isinstance(value, Mapping):
                raise RecordValidationError(f"raw record.{optional_field} must be an object")
            normalized[optional_field] = copy.deepcopy(dict(value))
    return normalized


def _normalize_token_rows(
    value: Any,
    *,
    field_name: str,
    requested_top_k: int,
    candidates_optional: bool = False,
    nullable: bool = False,
) -> list[dict[str, Any]] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, list):
        raise RecordValidationError(f"logprobs.{field_name} must be a list")

    normalized: list[dict[str, Any]] = []
    for position, row_value in enumerate(value):
        row = _mapping_value(row_value, f"logprobs.{field_name}[{position}]")
        token = _required_string(row, "token", f"logprobs.{field_name}[{position}]")
        token_bytes = _normalize_bytes(row.get("bytes"), f"logprobs.{field_name}[{position}].bytes")
        logprob = _normalize_logprob(row.get("logprob"), f"logprobs.{field_name}[{position}].logprob")
        candidates = row.get("top_logprobs", [] if candidates_optional else None)
        if not isinstance(candidates, list):
            raise RecordValidationError(f"logprobs.{field_name}[{position}].top_logprobs must be a list")
        if len(candidates) > requested_top_k:
            raise RecordValidationError(
                f"logprobs.{field_name}[{position}] contains more than requested top_logprobs"
            )

        normalized_candidates: list[dict[str, Any]] = []
        probability_mass = 0.0
        for candidate_index, candidate_value in enumerate(candidates):
            context = f"logprobs.{field_name}[{position}].top_logprobs[{candidate_index}]"
            candidate = _mapping_value(candidate_value, context)
            candidate_logprob = _normalize_logprob(candidate.get("logprob"), f"{context}.logprob")
            probability_mass += _probability(candidate_logprob)
            normalized_candidates.append(
                {
                    "token": _required_string(candidate, "token", context),
                    "bytes": _normalize_bytes(candidate.get("bytes"), f"{context}.bytes"),
                    "logprob": candidate_logprob,
                }
            )
        if probability_mass > 1.0 + 1e-6:
            raise RecordValidationError(
                f"logprobs.{field_name}[{position}] top probability mass is greater than one"
            )

        normalized.append(
            {
                "token": token,
                "bytes": token_bytes,
                "logprob": logprob,
                "top_logprobs": normalized_candidates,
                "top_probability_mass": min(probability_mass, 1.0),
            }
        )
    return normalized


def _compare_token_bytes(
    rows: list[dict[str, Any]] | None,
    expected_text: str | None,
    field_name: str,
    warnings: list[str],
) -> bool | None:
    if rows is None or expected_text is None:
        return None
    if any(row["bytes"] is None for row in rows):
        warnings.append(f"{field_name} token bytes are incomplete")
        return None
    try:
        reconstructed = b"".join(bytes(row["bytes"]) for row in rows).decode("utf-8")
    except UnicodeDecodeError:
        warnings.append(f"{field_name} token bytes are not valid UTF-8")
        return False
    if reconstructed != expected_text:
        warnings.append(f"{field_name} token bytes do not match response text")
        return False
    return True


def _normalize_logprob(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordValidationError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result) or result > 1e-7:
        raise RecordValidationError(f"{context} must be a finite non-positive number")
    return result


def _probability(logprob: float) -> float:
    return 0.0 if logprob == SENTINEL_LOGPROB else math.exp(logprob)


def _normalize_bytes(value: Any, context: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise RecordValidationError(f"{context} must be a list or null")
    if any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255 for item in value):
        raise RecordValidationError(f"{context} must contain byte integers between 0 and 255")
    return list(value)


def _required_mapping(value: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    return _mapping_value(value.get(key), f"{context}.{key}")


def _mapping_value(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordValidationError(f"{context} must be an object")
    return value


def _required_string(value: Mapping[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise RecordValidationError(f"{context}.{key} must be a non-empty string")
    return result
