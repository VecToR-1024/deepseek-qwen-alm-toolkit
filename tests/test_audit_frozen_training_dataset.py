from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_frozen_training_dataset import (
    benchmark_problem_text,
    build_overlap_report,
    evaluate_preflight_checks,
    parse_chat_template_kwargs,
)


def test_benchmark_problem_text_extracts_docstring_but_keeps_lcb_question() -> None:
    assert benchmark_problem_text(
        {"prompt": 'def add(a, b):\n    """Return the sum.\n    >>> add(1, 2)\n    3\n    """'}
    ) == "Return the sum. >>> add(1, 2) 3"
    assert benchmark_problem_text({"question_content": "Solve A + B."}) == "Solve A + B."


def test_overlap_report_finds_exact_normalized_matches(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "x", "question_content": "ADD two values!"}),
                json.dumps({"task_id": "y", "question_content": "Different task"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    training = [
        {"id": "train-1", "task": {"problem_text": "add TWO values"}},
        {"id": "train-2", "task": {"problem_text": "Another task"}},
    ]

    report = build_overlap_report(training, {"lcb": benchmark})

    assert report["lcb"]["benchmark_records"] == 2
    assert report["lcb"]["exact_normalized_match_count"] == 1
    assert report["lcb"]["matches"] == [
        {
            "training_id": "train-1",
            "benchmark_id": "x",
            "normalized_problem_text": "add two values",
        }
    ]


def test_preflight_checks_are_hard_gates() -> None:
    passing_contract = {
        "records": 1500,
        "end_token_supervision": {
            "eos_supervised_records": 1500,
            "template_boundary_failure_record_ids": [],
        },
        "alm_preprocessing": {
            "boundary_drops": 0,
            "zero_chunk_records": 0,
        },
        "teacher_response": {
            "records_with_code_fences": 0,
            "distributions": {"qwen_sequence_length": {"max": 4096}},
        },
    }
    overlap = {
        "humaneval": {"exact_normalized_match_count": 0},
        "mbpp": {"exact_normalized_match_count": 0},
    }

    checks = evaluate_preflight_checks(
        passing_contract,
        overlap,
        expected_records=1500,
        max_length=4096,
    )
    assert all(checks.values())

    passing_contract["end_token_supervision"]["eos_supervised_records"] = 1499
    checks = evaluate_preflight_checks(
        passing_contract,
        overlap,
        expected_records=1500,
        max_length=4096,
    )
    assert checks["all_eos_labels_supervised"] is False


def test_parse_chat_template_kwargs_requires_an_object() -> None:
    assert parse_chat_template_kwargs('{"enable_thinking": false}') == {
        "enable_thinking": False
    }

    for value in ("[]", '"text"', "{bad json}"):
        try:
            parse_chat_template_kwargs(value)
        except ValueError as error:
            assert "chat-template-kwargs" in str(error)
        else:
            raise AssertionError(f"accepted invalid value: {value}")
