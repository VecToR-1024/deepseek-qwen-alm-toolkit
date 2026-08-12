from __future__ import annotations

import pytest

from deepseek_distill.code_contests import (
    CODE_CONTESTS_REVISION,
    build_teacher_messages,
    import_code_contests_rows,
)


def code_contests_row(index: int) -> dict:
    return {
        "name": f"problem-{index}",
        "description": f"Solve contest problem {index}.",
        "public_tests": {
            "input": [f"PUBLIC-{index}\n"],
            "output": [f"PUBLIC-OUT-{index}\n"],
        },
        "private_tests": {
            "input": [f"PRIVATE-{index}\n"],
            "output": [f"PRIVATE-OUT-{index}\n"],
        },
        "generated_tests": {
            "input": [f"GENERATED-{index}\n"],
            "output": [f"GENERATED-OUT-{index}\n"],
        },
        "source": 2,
        "difficulty": 7,
        "solutions": {
            "language": [3],
            "solution": [f"print({index})"],
        },
        "incorrect_solutions": {
            "language": [3],
            "solution": ["pass"],
        },
        "cf_contest_id": 1000 + index,
        "cf_index": "A",
        "cf_points": 500.0,
        "cf_rating": 1200,
        "cf_tags": ["implementation"],
        "is_description_translated": False,
        "untranslated_description": "",
        "time_limit": {"seconds": 2, "nanos": 0},
        "memory_limit_bytes": 268_435_456,
        "input_file": "",
        "output_file": "",
    }


def test_code_contests_train_selection_is_deterministic_and_test_complete() -> None:
    rows = [code_contests_row(index) for index in range(8)]

    tasks = import_code_contests_rows(
        rows,
        limit=3,
        selection="random",
        seed=20260731,
    )
    repeated = import_code_contests_rows(
        reversed(rows),
        limit=3,
        selection="random",
        seed=20260731,
    )

    assert [task["id"] for task in tasks] == [task["id"] for task in repeated]
    assert all(task["source"]["split"] == "train" for task in tasks)
    assert all(task["source"]["revision"] == CODE_CONTESTS_REVISION for task in tasks)
    assert all(len(task["tests"]) == 3 for task in tasks)
    assert all(task["metadata"]["test_counts"] == {
        "public": 1,
        "private": 1,
        "generated": 1,
    } for task in tasks)


def test_code_contests_prompt_does_not_leak_private_tests_or_solutions() -> None:
    row = code_contests_row(9)
    task = import_code_contests_rows([row], limit=1, selection="first")[0]

    messages = build_teacher_messages(task)

    assert task["problem_text"] in messages[1]["content"]
    assert "PRIVATE-9" not in repr(messages)
    assert "GENERATED-9" not in repr(messages)
    assert row["solutions"]["solution"][0] not in repr(messages)
    assert task["metadata"]["reference_solution_count"] == 1
    assert len(task["metadata"]["reference_solutions_sha256"]) == 64


def test_code_contests_skips_file_io_tasks_and_rejects_malformed_tests() -> None:
    file_task = code_contests_row(1)
    file_task["input_file"] = "input.txt"

    with pytest.raises(ValueError, match="only 0 eligible"):
        import_code_contests_rows([file_task], limit=1, selection="first")

    malformed = code_contests_row(2)
    malformed["private_tests"]["output"] = []
    with pytest.raises(ValueError, match="private_tests"):
        import_code_contests_rows([malformed], limit=1, selection="first")


def test_code_contests_skips_rows_without_any_executable_tests() -> None:
    no_tests = code_contests_row(3)
    for label in ("public_tests", "private_tests", "generated_tests"):
        no_tests[label] = {"input": [], "output": []}

    tasks = import_code_contests_rows(
        [no_tests, code_contests_row(4)],
        limit=1,
        selection="first",
    )

    assert tasks[0]["source"]["original_name"] == "problem-4"
