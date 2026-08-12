"""Offline access to actual-token DeepSeek teacher traces."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .records import NORMALIZED_SCHEMA_VERSION


class TeacherTraceError(ValueError):
    """Raised when an offline record cannot provide an exact ALM trajectory."""


@dataclass(frozen=True, slots=True)
class OfflineTeacherTrace:
    """The teacher information used by ALM, independent of any teacher model."""

    record_id: str
    response_text: str
    token_bytes: tuple[bytes, ...]
    token_logprobs: tuple[float, ...]


class OfflineTeacherTraceProvider:
    """Read authoritative actual-token data from a normalized DeepSeek record.

    Top-20 alternatives remain in the source record for the strict baseline,
    but ALM consumes only the generated trajectory's bytes and logprobs.
    """

    def get_trace(self, record: Mapping[str, Any]) -> OfflineTeacherTrace:
        if not isinstance(record, Mapping):
            raise TeacherTraceError("teacher record must be an object")
        if record.get("schema_version") != NORMALIZED_SCHEMA_VERSION:
            raise TeacherTraceError(
                f"teacher schema_version must be {NORMALIZED_SCHEMA_VERSION!r}"
            )

        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise TeacherTraceError("teacher record id must be a non-empty string")
        response_text = record.get("response_text")
        if not isinstance(response_text, str):
            raise TeacherTraceError("teacher response_text must be a string")
        rows = record.get("content_tokens")
        if not isinstance(rows, list):
            raise TeacherTraceError("teacher content_tokens must be a list")

        token_bytes: list[bytes] = []
        token_logprobs: list[float] = []
        for position, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TeacherTraceError(f"content_tokens[{position}] must be an object")
            byte_values = row.get("bytes")
            if not isinstance(byte_values, list) or not byte_values:
                raise TeacherTraceError(
                    f"content_tokens[{position}].bytes must be a non-empty byte list"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 255
                for value in byte_values
            ):
                raise TeacherTraceError(
                    f"content_tokens[{position}].bytes contains an invalid byte"
                )
            logprob = row.get("logprob")
            if (
                isinstance(logprob, bool)
                or not isinstance(logprob, (int, float))
                or not math.isfinite(float(logprob))
                or float(logprob) > 1e-7
            ):
                raise TeacherTraceError(
                    f"content_tokens[{position}].logprob must be finite and non-positive"
                )
            token_bytes.append(bytes(byte_values))
            token_logprobs.append(float(logprob))

        response_bytes = response_text.encode("utf-8")
        if b"".join(token_bytes) != response_bytes:
            raise TeacherTraceError("actual teacher token bytes do not reconstruct response_text")

        return OfflineTeacherTrace(
            record_id=record_id,
            response_text=response_text,
            token_bytes=tuple(token_bytes),
            token_logprobs=tuple(token_logprobs),
        )
