from __future__ import annotations

from copy import deepcopy

from deepseek_distill.clean_eligibility import (
    CleanEligibilityPolicy,
    evaluate_clean_eligibility,
)


def _top_candidates(count: int = 20) -> list[dict]:
    return [
        {
            "token": f"candidate_{index}",
            "bytes": [97 + index % 26],
            "logprob": -float(index + 1),
        }
        for index in range(count)
    ]


def normalized_record(
    response_text: str,
    *,
    record_id: str = "taco_1__attempt_1",
    finish_reason: str = "stop",
    verification_category: str = "passed",
    top_candidate_count: int = 20,
) -> dict:
    return {
        "schema_version": "deepseek.teacher.normalized.v1",
        "id": record_id,
        "request": {
            "generation_config": {
                "top_logprobs": 20,
            }
        },
        "finish_reason": finish_reason,
        "response_text": response_text,
        "content_tokens": [
            {
                "token": response_text,
                "bytes": list(response_text.encode("utf-8")),
                "logprob": -0.1,
                "top_logprobs": _top_candidates(top_candidate_count),
            }
        ],
        "validation": {
            "content_bytes_match": True,
        },
        "task": {
            "source": {
                "dataset": "BAAI/TACO",
            },
            "tests": [],
        },
        "coding_verification": {
            "status": (
                "accepted" if verification_category == "passed" else "rejected"
            ),
            "failure_category": verification_category,
            "source_extraction": {
                "removed_markdown_fence": False,
            },
        },
    }


def alm_diagnostic(
    record_id: str = "taco_1__attempt_1",
    *,
    sequence_length: int = 256,
    valid_chunks: int = 20,
    boundary_drops: int = 0,
) -> dict:
    return {
        "id": record_id,
        "sequence_length": sequence_length,
        "valid_alm_chunks": valid_chunks,
        "prompt_completion_boundary_drops": boundary_drops,
    }


def test_accepts_plain_test_passing_source_with_sparse_code_comment() -> None:
    source = (
        "import sys\n"
        "data = sys.stdin.read().split()\n"
        "values = list(map(int, data))\n"
        "# The empty input has sum zero.\n"
        "print(sum(values))"
    )

    result = evaluate_clean_eligibility(
        normalized_record(source),
        alm_diagnostic=alm_diagnostic(),
        eos_supervised=True,
    )

    assert result["schema_version"] == "offline_alm.clean_eligibility.v1"
    assert result["eligible"] is True
    assert result["reasons"] == []
    assert result["raw_trace_preserved"] is True
    assert result["style"]["comment_line_ratio"] == 0.2
    assert result["trace"]["top_logprobs_complete"] is True


def test_accepts_actual_only_trace_without_top_candidates() -> None:
    record = normalized_record("def solve(x):\n    return x")
    record["request"]["generation_config"] = {
        "logprobs": True,
        "trace_profile": "actual_only",
    }
    record["content_tokens"][0]["top_logprobs"] = []

    result = evaluate_clean_eligibility(
        record,
        alm_diagnostic=alm_diagnostic(),
        eos_supervised=True,
    )

    assert result["eligible"] is True
    assert result["reasons"] == []
    assert result["trace"]["trace_profile"] == "actual_only"
    assert result["trace"]["expected_top_logprobs"] == 0
    assert result["trace"]["top_logprobs_complete"] is True


def test_rejects_fence_docstring_excessive_comments_and_length_finish() -> None:
    source = (
        "```python\n"
        '"""explanation"""\n'
        "# first explanation\n"
        "# second explanation\n"
        "print(1)\n"
        "```"
    )
    record = normalized_record(source, finish_reason="length")
    record["coding_verification"]["source_extraction"][
        "removed_markdown_fence"
    ] = True

    result = evaluate_clean_eligibility(
        record,
        alm_diagnostic=alm_diagnostic(),
        eos_supervised=True,
    )

    assert result["eligible"] is False
    assert result["reasons"] == [
        "finish_reason_length",
        "markdown_fence",
        "syntax_error",
        "comment_ratio_above_limit",
    ]


def test_rejects_python_parseable_explanatory_strings_and_top_level_asserts() -> None:
    record = normalized_record(
        '"This is the solution."\n'
        "def solve(x):\n"
        "    return x + 1\n"
        '"Done."\n'
        "assert solve(1) == 2"
    )

    result = evaluate_clean_eligibility(
        record,
        alm_diagnostic=alm_diagnostic(),
        eos_supervised=True,
    )

    assert result["eligible"] is False
    assert result["reasons"] == [
        "docstring",
        "prose_outside_code",
        "benchmark_test_content",
    ]


def test_rejects_incomplete_trace_failed_verification_and_invalid_alm() -> None:
    record = normalized_record(
        "def solve(x):\n    return x",
        verification_category="assertion_failure",
        top_candidate_count=19,
    )
    record["content_tokens"][0]["bytes"] = [120]

    result = evaluate_clean_eligibility(
        record,
        alm_diagnostic=alm_diagnostic(
            sequence_length=4097,
            valid_chunks=0,
            boundary_drops=1,
        ),
        eos_supervised=False,
    )

    assert result["eligible"] is False
    assert result["reasons"] == [
        "malformed_trace",
        "incomplete_top_logprobs",
        "official_test_failure",
        "sequence_over_4096",
        "zero_alm_chunks",
        "boundary_drop",
        "eos_not_supervised",
    ]


def test_evaluation_does_not_mutate_the_teacher_record() -> None:
    record = normalized_record("def solve(x):\n    return x")
    original = deepcopy(record)

    evaluate_clean_eligibility(
        record,
        alm_diagnostic=alm_diagnostic(),
        eos_supervised=True,
        policy=CleanEligibilityPolicy(max_comment_line_ratio=0.2),
    )

    assert record == original


def test_accepts_legacy_taco_verification_with_passed_category_and_no_status() -> None:
    record = normalized_record("def solve(x):\n    return x")
    del record["coding_verification"]["status"]

    result = evaluate_clean_eligibility(
        record,
        alm_diagnostic=alm_diagnostic(),
        eos_supervised=True,
    )

    assert result["eligible"] is True
    assert result["reasons"] == []
