from __future__ import annotations

from deepseek_distill.audit import AuditPricing
from deepseek_distill.rejection_audit import (
    build_rejection_sampling_audit,
    render_rejection_sampling_markdown,
)


def task(problem_id: str) -> dict:
    return {"id": problem_id, "source": {"original_id": int(problem_id.removeprefix("mbpp_"))}}


def attempt_id(problem_id: str, number: int) -> str:
    return f"{problem_id}__attempt_{number}"


def raw(problem_id: str, number: int) -> dict:
    identifier = attempt_id(problem_id, number)
    return {
        "id": identifier,
        "status": "ok",
        "task": {"id": identifier, "problem_id": problem_id, "attempt_number": number},
        "metrics": {"request_duration_seconds": float(number)},
    }


def normalized(problem_id: str, number: int) -> dict:
    identifier = attempt_id(problem_id, number)
    candidates = [
        {"token": str(index), "bytes": [index], "logprob": -5.0}
        for index in range(20)
    ]
    return {
        "id": identifier,
        "response_text": "def answer():\n    return 1\n",
        "content_tokens": [
            {"token": "code", "bytes": [99], "logprob": -0.1, "top_logprobs": candidates}
        ],
        "validation": {"content_bytes_match": True},
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def verifier(problem_id: str, number: int, category: str) -> dict:
    identifier = attempt_id(problem_id, number)
    extracted = category != "syntax_error"
    phases = []
    if extracted:
        phases.append({"name": "compile", "status": "passed"})
        phases.append(
            {
                "name": "import",
                "status": "import_error" if category == "import_error" else "passed",
            }
        )
        if category != "import_error":
            phases.append({"name": "test", "status": category})
    return {
        "id": identifier,
        "failure_category": category,
        "source_extraction": {"status": "passed" if extracted else "rejected"},
        "phases": phases,
    }


def test_rejection_sampling_audit_uses_attempt_and_unique_task_denominators() -> None:
    tasks = [task("mbpp_601"), task("mbpp_602"), task("mbpp_603")]
    outcomes = [
        ("mbpp_601", 1, "assertion_failure"),
        ("mbpp_601", 2, "passed"),
        ("mbpp_602", 1, "passed"),
        ("mbpp_603", 1, "syntax_error"),
        ("mbpp_603", 2, "import_error"),
        ("mbpp_603", 3, "timeout"),
    ]
    raw_records = [raw(problem_id, number) for problem_id, number, _ in outcomes]
    normalized_records = [
        normalized(problem_id, number) for problem_id, number, _ in outcomes
    ]
    verifier_records = [
        verifier(problem_id, number, category)
        for problem_id, number, category in outcomes
    ]
    accepted = [
        normalized("mbpp_601", 2)
        | {"sampling": {"problem_id": "mbpp_601", "attempt_number": 2}},
        normalized("mbpp_602", 1)
        | {"sampling": {"problem_id": "mbpp_602", "attempt_number": 1}},
    ]

    report = build_rejection_sampling_audit(
        tasks=tasks,
        raw_records=raw_records,
        normalized_records=normalized_records,
        verifier_records=verifier_records,
        accepted_records=accepted,
        first_target_records=accepted[:1],
        pricing=AuditPricing(0.025, 3.0, 6.0),
        alm_all={"examples": [{"id": "a"}, {"id": "b"}], "preprocessing_errors": []},
        alm_first_target={"examples": [{"id": "a"}], "preprocessing_errors": []},
    )

    assert report["sampling"]["pass_at_1"]["rate"] == 1 / 3
    assert report["sampling"]["cumulative_pass_at_2"]["rate"] == 2 / 3
    assert report["sampling"]["cumulative_pass_at_3"]["rate"] == 2 / 3
    assert report["sampling"]["accepted_by_attempt"] == {"1": 1, "2": 1, "3": 0}
    assert report["sampling"]["tasks_failing_all_3"] == 1
    assert report["sampling"]["average_attempts_per_accepted_task"] == 1.5
    assert report["rates"]["api_success"]["rate"] == 1.0
    assert report["rates"]["trace_reconstruction"]["rate"] == 1.0
    assert report["rates"]["source_extraction"]["numerator"] == 5
    assert report["rates"]["syntax_success"]["numerator"] == 5
    assert report["rates"]["import_success"]["numerator"] == 4
    assert report["rates"]["test_pass_per_attempt"]["rate"] == 2 / 6
    assert report["failure_counts_per_attempt"]["1"] == {
        "assertion_failure": 1,
        "passed": 1,
        "syntax_error": 1,
    }
    assert "passed" not in report["failure_counts"]
    assert report["cost_rmb"]["total_estimated"] == 0.00036
    assert report["cost_rmb"]["per_unique_accepted_task"] == 0.00018
    assert report["alm"]["all_unique_accepted"]["preprocessing_success"]["rate"] == 1.0
    assert report["duplicates"]["raw_attempt_ids"] == 0

    markdown = render_rejection_sampling_markdown(report)
    assert "Cumulative pass@3" in markdown
    assert "Failure categories by attempt" in markdown
    assert "All unique accepted" in markdown


def test_rejection_sampling_audit_supports_one_attempt_breadth_campaigns() -> None:
    tasks = [task("mbpp_601"), task("mbpp_602"), task("mbpp_603")]
    outcomes = [
        ("mbpp_601", 1, "passed"),
        ("mbpp_602", 1, "assertion_failure"),
        ("mbpp_603", 1, "timeout"),
    ]
    raw_records = [raw(problem_id, number) for problem_id, number, _ in outcomes]
    normalized_records = [
        normalized(problem_id, number) for problem_id, number, _ in outcomes
    ]
    verifier_records = [
        verifier(problem_id, number, category)
        for problem_id, number, category in outcomes
    ]
    accepted = [
        normalized("mbpp_601", 1)
        | {"sampling": {"problem_id": "mbpp_601", "attempt_number": 1}}
    ]

    report = build_rejection_sampling_audit(
        tasks=tasks,
        raw_records=raw_records,
        normalized_records=normalized_records,
        verifier_records=verifier_records,
        accepted_records=accepted,
        first_target_records=accepted,
        pricing=AuditPricing(0.025, 3.0, 6.0),
        max_attempts_per_task=1,
    )

    assert report["sampling"]["pass_at_1"]["rate"] == 1 / 3
    assert "cumulative_pass_at_2" not in report["sampling"]
    assert report["sampling"]["accepted_by_attempt"] == {"1": 1}
    assert report["sampling"]["tasks_failing_all_attempts"] == 2
    assert report["resumability"]["expected_attempt_slots"] == 3
    markdown = render_rejection_sampling_markdown(report)
    assert "Tasks failing all 1" in markdown
    assert "Cumulative pass@2" not in markdown
