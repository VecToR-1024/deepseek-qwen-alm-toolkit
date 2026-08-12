from __future__ import annotations

import json

from deepseek_distill.multisource_tasks import build_multisource_teacher_messages
from deepseek_distill.source_catalog import SOURCE_SPECS
from deepseek_distill.taco_multishard import import_taco_multishard_rows


def _row(index: int) -> dict:
    return {
        "question": f"Read and print one integer. Problem {index}.",
        "solutions": json.dumps(["print(input())"]),
        "starter_code": "",
        "input_output": json.dumps(
            {"inputs": [f"{index}\n"], "outputs": [f"{index}\n"]}
        ),
        "difficulty": "EASY",
        "url": f"https://example.test/taco/{index}",
        "source": "codeforces",
        "picture_num": 0,
        "date": "2024-01-01",
    }


def test_multishard_taco_ids_are_shard_qualified_and_deterministic() -> None:
    shards = [
        (1, [_row(index) for index in range(4)]),
        (8, [_row(index + 10) for index in range(4)]),
    ]

    first = import_taco_multishard_rows(
        shards,
        limit=5,
        selection="random",
        seed=20260731,
    )
    repeated = import_taco_multishard_rows(
        shards,
        limit=5,
        selection="random",
        seed=20260731,
    )

    assert [task["id"] for task in first] == [task["id"] for task in repeated]
    assert len({task["id"] for task in first}) == 5
    assert all(task["id"].startswith("taco_train_s") for task in first)
    assert {task["source"]["shard_index"] for task in first} <= {1, 8}
    assert all(task["source"]["split"] == "train" for task in first)


def test_multishard_taco_hard_profile_filters_before_selection() -> None:
    easy = _row(1)
    medium = _row(2)
    medium["difficulty"] = "MEDIUM"
    medium_hard = _row(3)
    medium_hard["difficulty"] = "MEDIUM_HARD"
    hard = _row(4)
    hard["difficulty"] = "HARD"

    selected = import_taco_multishard_rows(
        [(1, [easy, medium, medium_hard, hard])],
        limit=2,
        selection="first",
        difficulty_profile="hard-v1",
    )

    assert [task["metadata"]["difficulty"] for task in selected] == [
        "MEDIUM_HARD",
        "HARD",
    ]


def test_multishard_taco_filters_old_or_incompatible_task_shapes() -> None:
    geek = _row(1)
    geek["source"] = "GeeksForGeeks"
    starter = _row(2)
    starter["starter_code"] = "def solve():\n    pass\n"
    function_task = _row(3)
    function_task["input_output"] = json.dumps(
        {"fn_name": "solve", "inputs": [[1]], "outputs": [1]}
    )

    selected = import_taco_multishard_rows(
        [(1, [geek, starter, function_task, _row(4)])],
        limit=1,
        selection="first",
    )

    assert [task["id"] for task in selected] == ["taco_train_s01_r000003"]
    assert selected[0]["metadata"]["eligibility"]["shard_zero_excluded"] is True


def test_multishard_taco_prompt_hides_tests_and_reference_solutions() -> None:
    row = _row(7)
    task = import_taco_multishard_rows(
        [(2, [row])],
        limit=1,
        selection="first",
    )[0]

    messages = build_multisource_teacher_messages(task)

    assert task["problem_text"] in messages[1]["content"]
    assert task["tests"][0]["input"] not in repr(messages)
    assert json.loads(row["solutions"])[0] not in repr(messages)


def test_multishard_taco_is_registered_as_a_pinned_source() -> None:
    spec = SOURCE_SPECS["taco-multishard"]

    assert spec.dataset_id == "BAAI/TACO"
    assert spec.config == "ALL"
    assert spec.split == "train"
