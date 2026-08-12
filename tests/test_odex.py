from __future__ import annotations

from deepseek_distill.code_verifier import verify_normalized_record
from deepseek_distill.odex import (
    ODEX_REVISION,
    build_teacher_messages,
    import_odex_rows,
)


def odex_row(task_id: int, *, library: list[str] | None = None) -> dict:
    return {
        "task_id": task_id,
        "prompt": f"def f_{task_id}(value, *, increment=1): return",
        "suffix": "",
        "canonical_solution": " value + increment",
        "test_start": "def check(candidate):",
        "test": [
            "\n    assert candidate(1) == 2",
            "\n    assert candidate(2, increment=3) == 5",
        ],
        "entry_point": f"f_{task_id}",
        "intent": "add an increment to `value`",
        "library": [] if library is None else library,
    }


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


def test_odex_test_split_function_task_is_transparent_and_executable() -> None:
    task = import_odex_rows(
        [odex_row(101)],
        limit=1,
        selection="first",
    )[0]
    source = "def f_101(value, *, increment=1):\n    return value + increment"

    result = verify_normalized_record(normalized_record(task, source))

    assert task["source"]["split"] == "test"
    assert task["source"]["revision"] == ODEX_REVISION
    assert task["function_name"] == "f_101"
    assert task["function_signature"] == "f_101(value, *, increment=1)"
    assert task["metadata"]["benchmark_split_role"] == (
        "test_used_as_training_source_by_explicit_project_decision"
    )
    assert result["failure_category"] == "passed"


def test_odex_prompt_leaks_neither_tests_nor_canonical_solution() -> None:
    row = odex_row(102)
    task = import_odex_rows([row], limit=1, selection="first")[0]

    messages = build_teacher_messages(task)

    assert task["problem_text"] in messages[1]["content"]
    assert task["function_signature"] in messages[1]["content"]
    for hidden in [*row["test"], row["canonical_solution"], row["test_start"]]:
        assert hidden not in repr(messages)


def test_odex_filters_external_or_forbidden_library_tasks() -> None:
    tasks = import_odex_rows(
        [
            odex_row(103, library=["requests"]),
            odex_row(104, library=["urllib"]),
            odex_row(105, library=["collections"]),
        ],
        limit=1,
        selection="first",
    )

    assert tasks[0]["id"] == "odex_en_test_105"
    assert tasks[0]["metadata"]["libraries"] == ["collections"]


def test_odex_deduplicates_repeated_original_task_ids() -> None:
    duplicate = odex_row(106)
    duplicate["intent"] = "duplicate wording that must not become another task"

    tasks = import_odex_rows(
        [odex_row(106), duplicate],
        limit=1,
        selection="first",
    )

    assert [task["id"] for task in tasks] == ["odex_en_test_106"]
    assert tasks[0]["problem_text"] == "add an increment to `value`"


def test_odex_extracts_signature_from_header_only_prompt_without_inventing_args() -> None:
    row = odex_row(107)
    row["prompt"] = "def f_107(value, flag=False):\n"
    row["canonical_solution"] = "    return value if flag else value"

    task = import_odex_rows([row], limit=1, selection="first")[0]

    assert task["function_signature"] == "f_107(value, flag=False)"
