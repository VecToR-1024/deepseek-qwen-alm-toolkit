from __future__ import annotations

import json

import pytest

from deepseek_distill.taco import TACO_REVISION, import_taco_rows
from deepseek_distill.taco_breadth import (
    TACO_BREADTH_EXCLUDED_SOURCES,
    TACO_BREADTH_SELECTION_SCOPE,
    select_taco_breadth_tasks,
    validate_taco_breadth_tasks,
)


def row(index: int, *, source: str = "codeforces") -> dict:
    return {
        "question": f"Read and print an integer. Problem {index}.",
        "solutions": json.dumps(["print(input())"]),
        "input_output": json.dumps(
            {"inputs": [f"{index}\n"], "outputs": [f"{index}\n"]}
        ),
        "difficulty": "EASY",
        "source": source,
        "picture_num": 0,
    }


def prior_task(original_index: int) -> dict:
    return import_taco_rows(
        [row(index) for index in range(original_index + 1)],
        limit=1,
        selection="first",
    )[0] | {
        "id": f"taco_train_{original_index:06d}",
        "source": {
            "dataset": "BAAI/TACO",
            "split": "train",
            "revision": TACO_REVISION,
            "original_index": original_index,
        },
    }


def test_breadth_selection_excludes_prior_tasks_and_pseudo_stdin_sources() -> None:
    rows = [row(index) for index in range(15)]
    rows[3]["source"] = "geeksforgeeks"
    rows[9]["source"] = "GeeksForGeeks"
    prior = [prior_task(1), prior_task(7)]

    tasks = select_taco_breadth_tasks(
        rows,
        prior_tasks=prior,
        limit=10,
        seed=20260728,
    )

    assert len(tasks) == 10
    assert {task["id"] for task in tasks}.isdisjoint(
        {"taco_train_000001", "taco_train_000007"}
    )
    assert all(
        task["source"]["original_source"].lower()
        not in TACO_BREADTH_EXCLUDED_SOURCES
        for task in tasks
    )
    assert all(
        task["metadata"]["selection_scope"] == TACO_BREADTH_SELECTION_SCOPE
        for task in tasks
    )
    validate_taco_breadth_tasks(
        tasks,
        prior_tasks=prior,
        expected_count=10,
    )


def test_breadth_validation_rejects_overlap_with_prior_campaign() -> None:
    prior = [prior_task(1)]
    tasks = select_taco_breadth_tasks(
        [row(index) for index in range(5)],
        prior_tasks=[],
        limit=3,
        seed=20260728,
    )
    tasks[0]["id"] = prior[0]["id"]
    tasks[0]["source"]["original_index"] = 1

    with pytest.raises(ValueError, match="overlap"):
        validate_taco_breadth_tasks(
            tasks,
            prior_tasks=prior,
            expected_count=3,
        )


def test_breadth_validation_rejects_unrecorded_eligibility_policy() -> None:
    tasks = select_taco_breadth_tasks(
        [row(index) for index in range(5)],
        prior_tasks=[],
        limit=3,
        seed=20260728,
    )
    tasks[0]["metadata"]["eligibility"].pop("excluded_sources")

    with pytest.raises(ValueError, match="eligibility"):
        validate_taco_breadth_tasks(
            tasks,
            prior_tasks=[],
            expected_count=3,
        )
