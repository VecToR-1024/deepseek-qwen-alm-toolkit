from __future__ import annotations

import json

import pytest

from deepseek_distill.apps import (
    APPS_REVISION,
    build_teacher_messages,
    import_apps_rows,
)


def apps_row(problem_id: int, *, function_task: bool = False) -> dict:
    tests = {
        "inputs": [f"{problem_id}\n"],
        "outputs": [f"{problem_id}\n"],
    }
    if function_task:
        tests["fn_name"] = "solve"
    return {
        "problem_id": problem_id,
        "question": f"Read the integer {problem_id} and print it.",
        "solutions": json.dumps([f"print({problem_id})"]),
        "input_output": json.dumps(tests),
        "difficulty": "introductory",
        "url": f"https://example.test/apps/{problem_id}",
        "starter_code": "",
    }


def test_apps_import_is_train_only_deterministic_and_preserves_private_tests() -> None:
    tasks = import_apps_rows(
        [apps_row(index) for index in range(10, 15)],
        limit=3,
        selection="random",
        seed=20260731,
    )
    repeated = import_apps_rows(
        [apps_row(index) for index in range(10, 15)],
        limit=3,
        selection="random",
        seed=20260731,
    )

    assert [task["id"] for task in tasks] == [task["id"] for task in repeated]
    assert len({task["id"] for task in tasks}) == 3
    assert all(task["source"]["split"] == "train" for task in tasks)
    assert all(task["source"]["revision"] == APPS_REVISION for task in tasks)
    assert all(task["interface_type"] == "stdin_stdout" for task in tasks)
    assert all(task["tests"] for task in tasks)


def test_apps_import_filters_function_tasks_and_prompt_leaks_no_tests_or_solutions() -> None:
    row = apps_row(20)
    tasks = import_apps_rows(
        [apps_row(19, function_task=True), row],
        limit=1,
        selection="first",
    )
    task = tasks[0]

    messages = build_teacher_messages(task)

    assert task["id"] == "apps_train_000020"
    assert task["problem_text"] in messages[1]["content"]
    assert task["tests"][0]["input"] not in repr(messages)
    assert json.loads(row["solutions"])[0] not in repr(messages)
    assert task["metadata"]["reference_solution_count"] == 1
    assert len(task["metadata"]["reference_solutions_sha256"]) == 64


def test_apps_import_rejects_malformed_test_contract() -> None:
    row = apps_row(30)
    row["input_output"] = json.dumps({"inputs": ["1\n"], "outputs": []})

    with pytest.raises(ValueError, match="only 0 eligible"):
        import_apps_rows([row], limit=1)


def test_apps_skips_starter_code_row_before_parsing_empty_tests() -> None:
    official_function_shape = apps_row(1805, function_task=True)
    official_function_shape["starter_code"] = "class Solution:\n    pass\n"
    official_function_shape["input_output"] = ""

    selected = import_apps_rows(
        [official_function_shape, apps_row(7)],
        limit=1,
        selection="first",
    )

    assert [task["id"] for task in selected] == ["apps_train_000007"]


def test_apps_still_rejects_nonempty_malformed_json_for_candidate_row() -> None:
    malformed = apps_row(31)
    malformed["input_output"] = "not-json"

    with pytest.raises(ValueError, match="must be valid JSON"):
        import_apps_rows([malformed], limit=1)
