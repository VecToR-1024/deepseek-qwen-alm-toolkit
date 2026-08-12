from __future__ import annotations

from pathlib import Path

import pytest

from deepseek_distill.taco import build_teacher_messages
from deepseek_distill.taco_retry import (
    TACO_LENGTH_RETRY_SCHEMA_VERSION,
    build_length_retry_datasets,
    build_length_retry_tasks,
    portable_manifest_path,
    select_first_retry_per_problem,
)


def task(problem_id: str, original_index: int) -> dict:
    return {
        "schema_version": "coding.task.taco.v1",
        "id": problem_id,
        "source": {
            "dataset": "BAAI/TACO",
            "split": "train",
            "original_index": original_index,
        },
        "problem_text": f"Read one integer and print it for {problem_id}.",
        "interface_type": "stdin_stdout",
        "tests": [{"input": "3\n", "output": "3\n"}],
        "metadata": {"reference_solutions_sha256": "abc"},
    }


def normalized_attempt(
    problem_id: str,
    attempt: int,
    *,
    finish_reason: str,
) -> dict:
    attempt_id = f"{problem_id}__attempt_{attempt}"
    return {
        "id": attempt_id,
        "finish_reason": finish_reason,
        "response_text": "print(input())\n",
        "content_tokens": [],
        "task": {
            "id": attempt_id,
            "problem_id": problem_id,
            "attempt_number": attempt,
        },
    }


def accepted_v1(problem_id: str) -> dict:
    return {
        "id": f"{problem_id}__attempt_1",
        "sampling": {"problem_id": problem_id, "attempt_number": 1},
    }


def test_length_retry_tasks_are_canonical_and_skip_already_accepted_problems() -> None:
    first = task("taco_train_000010", 10)
    second = task("taco_train_000020", 20)
    retries = build_length_retry_tasks(
        selected_tasks=[first, second],
        normalized_attempts=[
            normalized_attempt(second["id"], 2, finish_reason="length"),
            normalized_attempt(first["id"], 3, finish_reason="length"),
            normalized_attempt(first["id"], 1, finish_reason="stop"),
            normalized_attempt(second["id"], 1, finish_reason="length"),
        ],
        accepted_v1=[accepted_v1(first["id"])],
        max_tokens=8192,
    )

    assert [record["id"] for record in retries] == [
        "taco_train_000020__attempt_1__length_retry_v2",
        "taco_train_000020__attempt_2__length_retry_v2",
    ]
    assert all(record["schema_version"] == TACO_LENGTH_RETRY_SCHEMA_VERSION for record in retries)
    assert all(record["problem_id"] == second["id"] for record in retries)
    assert [record["retry"]["source_attempt_number"] for record in retries] == [1, 2]
    assert all(record["retry"]["max_tokens"] == 8192 for record in retries)
    assert all(record["retry"]["teacher_feedback"] is False for record in retries)


def test_length_retry_prompt_is_identical_to_original_blind_prompt() -> None:
    original = task("taco_train_000020", 20)
    retry = build_length_retry_tasks(
        selected_tasks=[original],
        normalized_attempts=[
            normalized_attempt(original["id"], 1, finish_reason="length")
        ],
        accepted_v1=[],
        max_tokens=8192,
    )[0]

    assert build_teacher_messages(retry) == build_teacher_messages(original)
    assert original["tests"][0]["input"] not in repr(build_teacher_messages(retry))
    assert "length" not in repr(build_teacher_messages(retry)).lower()
    assert "retry" not in repr(build_teacher_messages(retry)).lower()


def test_length_retry_dataset_keeps_first_pass_and_original_problem_order() -> None:
    first = task("taco_train_000010", 10)
    second = task("taco_train_000020", 20)
    retries = build_length_retry_tasks(
        selected_tasks=[first, second],
        normalized_attempts=[
            normalized_attempt(first["id"], 1, finish_reason="length"),
            normalized_attempt(second["id"], 1, finish_reason="length"),
            normalized_attempt(second["id"], 2, finish_reason="length"),
        ],
        accepted_v1=[],
        max_tokens=8192,
    )
    normalized_retries = [
        {
            **normalized_attempt(first["id"], 1, finish_reason="stop"),
            "id": retries[0]["id"],
            "task": retries[0],
        },
        {
            **normalized_attempt(second["id"], 1, finish_reason="stop"),
            "id": retries[1]["id"],
            "task": retries[1],
        },
        {
            **normalized_attempt(second["id"], 2, finish_reason="stop"),
            "id": retries[2]["id"],
            "task": retries[2],
        },
    ]
    verifier = [
        {"id": retries[0]["id"], "failure_category": "passed"},
        {"id": retries[1]["id"], "failure_category": "assertion_failure"},
        {"id": retries[2]["id"], "failure_category": "passed"},
    ]

    datasets = build_length_retry_datasets(
        selected_tasks=[first, second],
        accepted_v1=[],
        retry_tasks=retries,
        normalized_retries=normalized_retries,
        verifier_results=verifier,
    )

    assert [
        record["sampling"]["problem_id"] for record in datasets["newly_accepted_unique"]
    ] == [first["id"], second["id"]]
    assert datasets["newly_accepted_unique"][1]["sampling"]["source_attempt_number"] == 2
    assert [
        record["sampling"]["problem_id"] for record in datasets["combined_accepted_unique"]
    ] == [first["id"], second["id"]]


def test_smoke_selection_uses_one_retry_per_distinct_problem() -> None:
    retries = [
        {"id": "a1", "problem_id": "taco_train_000001"},
        {"id": "a2", "problem_id": "taco_train_000001"},
        {"id": "b1", "problem_id": "taco_train_000002"},
        {"id": "c1", "problem_id": "taco_train_000003"},
    ]

    selected = select_first_retry_per_problem(retries, problem_limit=2)

    assert [record["id"] for record in selected] == ["a1", "b1"]


def test_manifest_path_is_stable_and_does_not_resolve_the_local_workspace() -> None:
    path = Path("data") / "taco_pilot_v1" / "run100"

    assert portable_manifest_path(path) == "data/taco_pilot_v1/run100"


def test_manifest_path_relativizes_an_absolute_path_inside_the_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data" / "taco_pilot_v1" / "run100"

    assert portable_manifest_path(path) == "data/taco_pilot_v1/run100"
