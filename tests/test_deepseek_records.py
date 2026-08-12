import math

import pytest

from deepseek_distill.records import (
    RAW_SCHEMA_VERSION,
    RecordValidationError,
    normalize_raw_record,
)
from deepseek_distill.offline_teacher import OfflineTeacherTraceProvider


def make_raw_record(*, content: str = "Hi", content_tokens: list[dict] | None = None) -> dict:
    if content_tokens is None:
        content_tokens = [
            {
                "token": "H",
                "bytes": [72],
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": "H", "bytes": [72], "logprob": -0.1},
                    {"token": "A", "bytes": [65], "logprob": -2.5},
                ],
            },
            {
                "token": "i",
                "bytes": [105],
                "logprob": -0.2,
                "top_logprobs": [
                    {"token": "i", "bytes": [105], "logprob": -0.2},
                ],
            },
        ]
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "id": "problem_0001",
        "status": "ok",
        "collected_at": "2026-07-20T00:00:00Z",
        "request": {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "Say hi"}],
            "generation_config": {
                "thinking": {"type": "disabled"},
                "temperature": 1.0,
                "top_p": 1.0,
                "logprobs": True,
                "top_logprobs": 20,
            },
        },
        "response": {
            "id": "chatcmpl-1",
            "model": "deepseek-v4-pro",
            "system_fingerprint": "fp_test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": None,
                    },
                    "logprobs": {
                        "content": content_tokens,
                        "reasoning_content": None,
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    }


def test_normalize_raw_record_preserves_traceability_and_reconstructs_bytes() -> None:
    raw = make_raw_record()
    raw["request"]["prompt_contract"] = {
        "id": "deepseek.python.clean.v2",
        "interface_type": "function",
    }
    raw["task"] = {"schema_version": "coding.task.mbpp.v1", "id": "mbpp_601"}
    raw["provider"] = {"name": "DeepSeek", "base_url": "https://api.deepseek.com"}
    raw["metrics"] = {"request_duration_seconds": 0.75}

    normalized = normalize_raw_record(raw)

    assert normalized["id"] == "problem_0001"
    assert normalized["teacher_model"] == "deepseek-v4-pro"
    assert normalized["request"]["messages"] == [{"role": "user", "content": "Say hi"}]
    assert normalized["request"]["prompt_contract"] == {
        "id": "deepseek.python.clean.v2",
        "interface_type": "function",
    }
    assert normalized["response_text"] == "Hi"
    assert normalized["content_tokens"][0]["bytes"] == [72]
    assert normalized["content_tokens"][0]["top_probability_mass"] == pytest.approx(
        math.exp(-0.1) + math.exp(-2.5)
    )
    assert normalized["validation"]["content_bytes_match"] is True
    assert normalized["validation"]["warnings"] == []
    assert normalized["task"]["id"] == "mbpp_601"
    assert normalized["provider"]["name"] == "DeepSeek"
    assert normalized["metrics"]["request_duration_seconds"] == 0.75


def test_normalize_raw_record_accepts_null_bytes_and_negative_9999_sentinel() -> None:
    token = {
        "token": "x",
        "bytes": None,
        "logprob": -9999.0,
        "top_logprobs": [
            {"token": "x", "bytes": None, "logprob": -9999.0},
        ],
    }

    normalized = normalize_raw_record(make_raw_record(content="x", content_tokens=[token]))

    assert normalized["content_tokens"][0]["top_probability_mass"] == 0.0
    assert normalized["validation"]["content_bytes_match"] is None
    assert "content token bytes are incomplete" in normalized["validation"]["warnings"]


def test_normalize_raw_record_rejects_top_probability_mass_above_one() -> None:
    token = {
        "token": "x",
        "bytes": [120],
        "logprob": 0.0,
        "top_logprobs": [
            {"token": "x", "bytes": [120], "logprob": 0.0},
            {"token": "y", "bytes": [121], "logprob": 0.0},
        ],
    }

    with pytest.raises(RecordValidationError, match="probability mass"):
        normalize_raw_record(make_raw_record(content="x", content_tokens=[token]))


def test_actual_only_record_accepts_missing_candidate_arrays_for_alm() -> None:
    raw = make_raw_record()
    raw["request"]["generation_config"] = {
        "thinking": {"type": "disabled"},
        "temperature": 1.0,
        "top_p": 1.0,
        "logprobs": True,
        "trace_profile": "actual_only",
    }
    for token in raw["response"]["choices"][0]["logprobs"]["content"]:
        token.pop("top_logprobs")

    normalized = normalize_raw_record(raw)
    trace = OfflineTeacherTraceProvider().get_trace(normalized)

    assert normalized["content_tokens"][0]["top_logprobs"] == []
    assert normalized["content_tokens"][0]["top_probability_mass"] == 0.0
    assert b"".join(trace.token_bytes) == b"Hi"
    assert trace.token_logprobs == (-0.1, -0.2)


def test_top20_record_still_rejects_missing_candidate_arrays() -> None:
    raw = make_raw_record()
    raw["response"]["choices"][0]["logprobs"]["content"][0].pop(
        "top_logprobs"
    )

    with pytest.raises(RecordValidationError, match="top_logprobs must be a list"):
        normalize_raw_record(raw)


def test_normalize_raw_record_rejects_api_error_records() -> None:
    raw = make_raw_record()
    raw["status"] = "error"
    raw["response"] = None
    raw["error"] = {"type": "RateLimitError", "message": "slow down"}

    with pytest.raises(RecordValidationError, match="status is not ok"):
        normalize_raw_record(raw)
