from __future__ import annotations

import pytest

from deepseek_distill.code_verifier import verify_normalized_record
from deepseek_distill.xcodeeval import (
    XCODEEVAL_REVISION,
    build_teacher_messages,
    import_xcodeeval_rows,
)


def xcodeeval_row(index: int) -> dict:
    return {
        "src_uid": f"uid-{index:03d}",
        "description": f"Print the answer for problem {index}.",
        "input_from": "standard input",
        "output_to": "standard output",
        "time_limit": "2 seconds",
        "memory_limit": "256 MB",
        "input_spec": "The input contains one integer.",
        "output_spec": "Print one integer.",
        "sample_inputs": [f"{index}\n"],
        "sample_outputs": [f"{index}\n"],
        "notes": "",
        "created_at": "0",
        "difficulty": index,
        "tags": ["implementation"],
    }


def unit_tests(index: int) -> list[dict]:
    return [
        {
            "input": f"SECRET-XCODE-{index}\n",
            "output": [f"{index}\n", f" {index} \r\n"],
        }
    ]


def normalized_record(task: dict, response_text: str) -> dict:
    return {
        "schema_version": "deepseek.teacher.normalized.v1",
        "id": task["id"],
        "response_text": response_text,
        "content_tokens": [
            {
                "bytes": list(response_text.encode("utf-8")),
                "logprob": -0.1,
            }
        ],
        "task": task,
    }


def test_xcodeeval_compact_selection_is_deterministic_and_split_transparent() -> None:
    rows = [xcodeeval_row(index) for index in range(10)]
    tests = {row["src_uid"]: unit_tests(index) for index, row in enumerate(rows)}

    tasks = import_xcodeeval_rows(
        rows,
        unit_test_db=tests,
        limit=3,
        selection="random",
        seed=20260731,
    )
    repeated = import_xcodeeval_rows(
        reversed(rows),
        unit_test_db=tests,
        limit=3,
        selection="random",
        seed=20260731,
    )

    assert [task["id"] for task in tasks] == [task["id"] for task in repeated]
    assert all(task["source"]["split"] == "compact" for task in tasks)
    assert all(task["source"]["raw_split"] == "validation" for task in tasks)
    assert all(task["source"]["revision"] == XCODEEVAL_REVISION for task in tasks)
    assert all(task["metadata"]["benchmark_split_role"] == (
        "compact_used_as_training_source_by_explicit_project_decision"
    ) for task in tasks)
    assert all("pinned loader" in task["source"]["license"] for task in tasks)


def test_xcodeeval_prompt_hides_unit_tests_and_verifier_accepts_output_alternatives() -> None:
    row = xcodeeval_row(7)
    task = import_xcodeeval_rows(
        [row],
        unit_test_db={row["src_uid"]: unit_tests(7)},
        limit=1,
        selection="first",
    )[0]

    messages = build_teacher_messages(task)
    result = verify_normalized_record(
        normalized_record(task, "input()\nprint(7)\n"),
        phase_timeout_seconds=3.0,
    )

    assert "SECRET-XCODE-7" not in repr(messages)
    assert row["description"] in messages[1]["content"]
    assert task["tests"][0]["output"] == ["7\n", " 7 \r\n"]
    assert result["failure_category"] == "passed"


def test_xcodeeval_skips_missing_tests_and_rejects_malformed_test_rows() -> None:
    missing = xcodeeval_row(1)
    valid = xcodeeval_row(2)
    tasks = import_xcodeeval_rows(
        [missing, valid],
        unit_test_db={valid["src_uid"]: unit_tests(2)},
        limit=1,
        selection="first",
    )
    assert tasks[0]["id"] == "xcodeeval_compact_uid-002"

    with pytest.raises(ValueError, match="output"):
        import_xcodeeval_rows(
            [valid],
            unit_test_db={
                valid["src_uid"]: [{"input": "1\n", "output": []}]
            },
            limit=1,
            selection="first",
        )
