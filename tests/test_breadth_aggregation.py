from __future__ import annotations

import json
from pathlib import Path

from deepseek_distill.breadth_aggregation import (
    aggregate_attempt_campaign,
    aggregate_single_attempt_campaign,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def task(problem_id: str, selection_index: int) -> dict:
    return {
        "schema_version": "coding.task.taco.v1",
        "id": problem_id,
        "source": {"original_index": selection_index},
        "problem_text": "Echo one integer.",
        "interface_type": "stdin_stdout",
        "tests": [{"input": "1\n", "output": "1\n"}],
    }


def test_single_attempt_aggregation_streams_accepted_and_rejected_outputs(
    tmp_path: Path,
) -> None:
    selected = [
        task("taco_train_000010", 10),
        task("taco_train_000020", 20),
    ]
    write_jsonl(tmp_path / "selected_tasks_2.jsonl", selected)
    raw = [
        {
            "id": "taco_train_000010__attempt_1",
            "status": "ok",
            "task": {"problem_id": "taco_train_000010"},
        },
        {
            "id": "taco_train_000020__attempt_1",
            "status": "ok",
            "task": {"problem_id": "taco_train_000020"},
        },
    ]
    write_jsonl(tmp_path / "raw_attempts.jsonl", raw)
    normalized = [
        {
            "id": record["id"],
            "response_text": "print(input())\n",
            "content_tokens": [],
            "finish_reason": "stop",
        }
        for record in raw
    ]
    write_jsonl(tmp_path / "normalized_attempts.jsonl", normalized)
    verifier = [
        {
            "id": raw[0]["id"],
            "failure_category": "passed",
            "task": {"id": raw[0]["id"]},
        },
        {
            "id": raw[1]["id"],
            "failure_category": "assertion_failure",
            "task": {"id": raw[1]["id"]},
        },
    ]
    write_jsonl(tmp_path / "verifier_attempts.jsonl", verifier)

    summary = aggregate_single_attempt_campaign(
        run_dir=tmp_path,
        selected_tasks_path=tmp_path / "selected_tasks_2.jsonl",
        target=2,
    )

    accepted = read_jsonl(tmp_path / "accepted_unique.jsonl")
    assert [record["sampling"]["problem_id"] for record in accepted] == [
        "taco_train_000010"
    ]
    assert accepted[0]["coding_verification"]["failure_category"] == "passed"
    rejected = read_jsonl(tmp_path / "rejected_tasks.jsonl")
    assert [record["problem_id"] for record in rejected] == [
        "taco_train_000020"
    ]
    assert rejected[0]["campaign_complete"] is True
    assert summary["dataset"]["pending_attempt_slots"] == 0
    assert summary["counts"]["accepted_unique"] == 1
    assert not (tmp_path / "accepted_first_2.jsonl").exists()

    repeated = aggregate_single_attempt_campaign(
        run_dir=tmp_path,
        selected_tasks_path=tmp_path / "selected_tasks_2.jsonl",
        target=2,
    )
    assert repeated == summary


def test_single_attempt_aggregation_uses_multisource_dataset_slug(
    tmp_path: Path,
) -> None:
    selected = [
        {
            "schema_version": "coding.task.multisource.v1",
            "id": "apps_train_000017",
            "source": {"dataset": "codeparrot/apps"},
            "problem_text": "Echo one integer.",
            "interface_type": "stdin_stdout",
            "tests": [{"input": "1\n", "output": "1\n"}],
        }
    ]
    selected_path = tmp_path / "selected_tasks_1.jsonl"
    write_jsonl(selected_path, selected)
    write_jsonl(
        tmp_path / "raw_attempts.jsonl",
        [{"id": "apps_train_000017__attempt_1", "status": "error"}],
    )
    write_jsonl(tmp_path / "normalized_attempts.jsonl", [])
    write_jsonl(tmp_path / "verifier_attempts.jsonl", [])

    summary = aggregate_single_attempt_campaign(
        run_dir=tmp_path,
        selected_tasks_path=selected_path,
        target=1,
    )

    assert summary["schema_version"] == "coding.collection.apps.breadth.summary.v1"
    assert read_jsonl(tmp_path / "rejected_tasks.jsonl")[0][
        "schema_version"
    ] == "coding.rejected.task.apps.v1"
    assert read_jsonl(tmp_path / "attempt_ledger.jsonl")[0][
        "schema_version"
    ] == "coding.attempt.ledger.apps.v1"


def test_multi_attempt_aggregation_selects_earliest_pass_without_loading_traces(
    tmp_path: Path,
) -> None:
    selected = [
        task("taco_train_000010", 10),
        task("taco_train_000020", 20),
    ]
    selected_path = tmp_path / "selected_tasks_2.jsonl"
    write_jsonl(selected_path, selected)
    attempt_ids = [
        "taco_train_000010__attempt_1",
        "taco_train_000020__attempt_1",
        "taco_train_000010__attempt_2",
        "taco_train_000020__attempt_2",
        "taco_train_000020__attempt_3",
    ]
    write_jsonl(
        tmp_path / "raw_attempts.jsonl",
        [{"id": attempt_id, "status": "ok"} for attempt_id in attempt_ids],
    )
    write_jsonl(
        tmp_path / "normalized_attempts.jsonl",
        [
            {
                "id": attempt_id,
                "response_text": f"# {attempt_id}\n",
                "finish_reason": "stop",
            }
            for attempt_id in reversed(attempt_ids)
        ],
    )
    categories = {
        "taco_train_000010__attempt_1": "assertion_failure",
        "taco_train_000010__attempt_2": "passed",
        "taco_train_000020__attempt_1": "runtime_error",
        "taco_train_000020__attempt_2": "assertion_failure",
        "taco_train_000020__attempt_3": "timeout",
    }
    write_jsonl(
        tmp_path / "verifier_attempts.jsonl",
        [
            {"id": attempt_id, "failure_category": categories[attempt_id]}
            for attempt_id in attempt_ids
        ],
    )

    summary = aggregate_attempt_campaign(
        run_dir=tmp_path,
        selected_tasks_path=selected_path,
        target=2,
        max_attempts_per_task=3,
    )

    accepted = read_jsonl(tmp_path / "accepted_unique.jsonl")
    assert [record["id"] for record in accepted] == [
        "taco_train_000010__attempt_2"
    ]
    assert accepted[0]["sampling"]["attempt_number"] == 2
    ledger = {
        record["id"]: record for record in read_jsonl(tmp_path / "attempt_ledger.jsonl")
    }
    assert ledger["taco_train_000010__attempt_3"]["state"] == "not_requested_after_pass"
    assert summary["dataset"]["actual_attempts"] == 5
    assert summary["dataset"]["pending_attempt_slots"] == 0
    assert summary["counts"]["accepted_by_attempt"] == {"2": 1}

    repeated = aggregate_attempt_campaign(
        run_dir=tmp_path,
        selected_tasks_path=selected_path,
        target=2,
        max_attempts_per_task=3,
    )
    assert repeated == summary
