from __future__ import annotations

import json
import random

import pytest

from deepseek_distill.taco import (
    TACO_REVISION,
    TACO_TASK_SCHEMA_VERSION,
    build_teacher_messages,
    import_taco_rows,
)


def taco_row(index: int, *, fn_name: str | None = None) -> dict:
    input_output = {
        "inputs": [f"{index}\n", f"{index + 1}\n"],
        "outputs": [f"{index}\n", f"{index + 1}\n"],
    }
    if fn_name is not None:
        input_output["fn_name"] = fn_name
    return {
        "question": f"Read one integer and print it unchanged. Problem {index}.",
        "solutions": json.dumps([f"print(input())  # {index}"]),
        "starter_code": "",
        "input_output": json.dumps(input_output),
        "difficulty": "EASY",
        "url": f"https://example.invalid/problem/{index}",
        "source": "codeforces",
        "picture_num": 0,
        "date": "2023-01-01",
    }


def test_import_taco_preserves_problem_and_separates_tests() -> None:
    task = import_taco_rows([taco_row(0)], limit=1, selection="first")[0]

    assert task["schema_version"] == TACO_TASK_SCHEMA_VERSION
    assert task["id"] == "taco_train_000000"
    assert task["source"]["dataset"] == "BAAI/TACO"
    assert task["source"]["split"] == "train"
    assert task["source"]["revision"] == TACO_REVISION
    assert task["source"]["original_index"] == 0
    assert task["interface_type"] == "stdin_stdout"
    assert task["tests"] == [
        {"input": "0\n", "output": "0\n"},
        {"input": "1\n", "output": "1\n"},
    ]
    assert task["metadata"]["reference_solution_count"] == 1
    assert "reference_solutions" not in task["metadata"]


def test_import_taco_random_selection_is_deterministic_and_keeps_sample_order() -> None:
    rows = [taco_row(index) for index in range(20)]
    expected = random.Random(20260728).sample(list(range(20)), 8)

    first = import_taco_rows(
        rows, limit=8, selection="random", seed=20260728
    )
    repeated = import_taco_rows(
        rows, limit=8, selection="random", seed=20260728
    )

    assert [task["source"]["original_index"] for task in first] == expected
    assert [task["id"] for task in repeated] == [task["id"] for task in first]


def test_import_taco_excludes_prior_tasks_and_bad_sources_before_sampling() -> None:
    rows = [taco_row(index) for index in range(12)]
    rows[2]["source"] = "geeksforgeeks"
    rows[7]["source"] = "GeeksForGeeks"

    tasks = import_taco_rows(
        rows,
        limit=8,
        selection="random",
        seed=20260728,
        excluded_task_ids={"taco_train_000000", "taco_train_000005"},
        excluded_sources={"geeksforgeeks"},
        selection_scope="single_pinned_train_shard_breadth_v2",
    )

    assert len(tasks) == 8
    assert not {
        "taco_train_000000",
        "taco_train_000002",
        "taco_train_000005",
        "taco_train_000007",
    } & {task["id"] for task in tasks}
    assert all(
        task["metadata"]["selection_scope"]
        == "single_pinned_train_shard_breadth_v2"
        for task in tasks
    )
    assert all(
        task["metadata"]["eligibility"]["excluded_sources"] == ["geeksforgeeks"]
        for task in tasks
    )


def test_import_taco_skips_call_based_and_rejects_unsafe_test_json() -> None:
    tasks = import_taco_rows(
        [taco_row(0, fn_name="solve"), taco_row(1)],
        limit=1,
        selection="first",
    )
    assert [task["id"] for task in tasks] == ["taco_train_000001"]

    malicious = taco_row(2)
    malicious["input_output"] = "__import__('os').system('echo unsafe')"
    with pytest.raises(ValueError, match="valid JSON"):
        import_taco_rows([malicious], limit=1)


def test_taco_prompt_is_distinct_and_does_not_leak_tests_or_solutions() -> None:
    row = taco_row(7)
    task = import_taco_rows([row], limit=1)[0]

    messages = build_teacher_messages(task)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert task["problem_text"] in messages[1]["content"]
    assert "standard input" in messages[1]["content"].lower()
    for test in task["tests"]:
        assert test["input"] not in repr(messages)
        assert test["output"] not in repr(messages)
    assert json.loads(row["solutions"])[0] not in repr(messages)


def test_taco_prompt_rejects_missing_problem_before_api_call() -> None:
    task = import_taco_rows([taco_row(0)], limit=1)[0]
    task["problem_text"] = " "

    with pytest.raises(ValueError, match="actual problem statement"):
        build_teacher_messages(task)
