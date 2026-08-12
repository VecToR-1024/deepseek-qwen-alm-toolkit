from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepseek_distill.candidate_merge import (
    CandidateSource,
    build_merged_candidate_outputs,
)


def normalized_record(
    record_id: str,
    *,
    problem_id: str,
    task_schema: str = "coding.task.taco.v1",
    dataset: str = "fixture",
) -> dict:
    response = "print(1)\n"
    return {
        "schema_version": "deepseek.teacher.normalized.v1",
        "id": record_id,
        "request": {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "problem"},
            ]
        },
        "response_text": response,
        "content_tokens": [
            {
                "token": response,
                "bytes": list(response.encode("utf-8")),
                "logprob": -0.25,
                "top_logprobs": [],
            }
        ],
        "validation": {"content_bytes_match": True, "warnings": []},
        "task": {
            "schema_version": task_schema,
            "id": problem_id,
            "source": {"dataset": dataset, "split": "train"},
        },
        "sampling": {"problem_id": problem_id},
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def diagnostics(records: list[dict], *, over_limit: list[str] | None = None) -> dict:
    over_limit = over_limit or []
    return {
        "student_tokenizer": "fixture-tokenizer",
        "student_revision": "fixture-revision",
        "max_length": 4096,
        "sequence_length_distribution": {},
        "chunks_per_example_distribution": {},
        "group_counts": {"1:1": 3, "1:N": 0, "N:1": 0, "N:M": 0},
        "prompt_completion_boundary_drops": 0,
        "examples_with_zero_valid_chunks": [],
        "records_exceeding_max_length": over_limit,
        "preprocessing_errors": [],
        "examples": [
            {
                "id": record["id"],
                "sequence_length": 5000 if record["id"] in over_limit else 500,
                "valid_alm_chunks": 1,
                "group_counts": {"1:1": 1, "1:N": 0, "N:1": 0, "N:M": 0},
                "prompt_completion_boundary_drops": 0,
            }
            for record in records
        ],
    }


def test_merge_preserves_source_order_and_separates_overlength_records(
    tmp_path: Path,
) -> None:
    mbpp = normalized_record(
        "mbpp_1__attempt_1",
        problem_id="mbpp_1",
        task_schema="coding.task.mbpp.v1",
    )
    taco_old = normalized_record(
        "taco_train_1__attempt_1",
        problem_id="taco_train_1",
    )
    taco_new = normalized_record(
        "taco_train_2__attempt_1",
        problem_id="taco_train_2",
    )
    mbpp_path = tmp_path / "mbpp.jsonl"
    old_path = tmp_path / "taco_old.jsonl"
    new_path = tmp_path / "taco_new.jsonl"
    write_jsonl(mbpp_path, [mbpp])
    write_jsonl(old_path, [taco_old])
    write_jsonl(new_path, [taco_new])
    merged = [mbpp, taco_old, taco_new]

    summary = build_merged_candidate_outputs(
        sources=[
            CandidateSource("mbpp200", mbpp_path, expected_records=1),
            CandidateSource("taco49", old_path, expected_records=1),
            CandidateSource("taco412", new_path, expected_records=1),
        ],
        output_dir=tmp_path / "merged",
        alm_diagnostics=diagnostics(
            merged,
            over_limit=["taco_train_1__attempt_1"],
        ),
    )

    assert [record["id"] for record in read_jsonl(tmp_path / "merged/all_candidates.jsonl")] == [
        "mbpp_1__attempt_1",
        "taco_train_1__attempt_1",
        "taco_train_2__attempt_1",
    ]
    assert [
        record["id"]
        for record in read_jsonl(tmp_path / "merged/trainable_max4096.jsonl")
    ] == [
        "mbpp_1__attempt_1",
        "taco_train_2__attempt_1",
    ]
    assert read_jsonl(tmp_path / "merged/excluded_records.jsonl") == [
        {
            "schema_version": "coding.training_candidate.exclusion.v1",
            "id": "taco_train_1__attempt_1",
            "problem_id": "taco_train_1",
            "source_label": "taco49",
            "reasons": ["sequence_length_exceeds_4096"],
            "sequence_length": 5000,
            "valid_alm_chunks": 1,
        }
    ]
    assert summary["counts"] == {
        "all_candidates": 3,
        "trainable_max4096": 2,
        "excluded": 1,
        "mbpp": 1,
        "taco": 2,
    }
    assert summary["source_order"] == ["mbpp200", "taco49", "taco412"]
    assert summary["duplicates"] == {"record_ids": 0, "problem_ids": 0}


def test_merge_labels_multisource_candidates_by_dataset(tmp_path: Path) -> None:
    apps = normalized_record(
        "apps_train_000017__attempt_1",
        problem_id="apps_train_000017",
        task_schema="coding.task.multisource.v1",
        dataset="codeparrot/apps",
    )
    source_path = tmp_path / "apps.jsonl"
    write_jsonl(source_path, [apps])

    summary = build_merged_candidate_outputs(
        sources=[CandidateSource("apps", source_path, expected_records=1)],
        output_dir=tmp_path / "merged",
        alm_diagnostics=diagnostics([apps]),
    )

    assert summary["counts"] == {
        "all_candidates": 1,
        "trainable_max4096": 1,
        "excluded": 0,
        "apps": 1,
    }


def test_merge_rejects_duplicate_problem_ids_across_sources(tmp_path: Path) -> None:
    first = normalized_record("taco_train_1__attempt_1", problem_id="taco_train_1")
    second = normalized_record("taco_train_1__attempt_2", problem_id="taco_train_1")
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    write_jsonl(first_path, [first])
    write_jsonl(second_path, [second])

    with pytest.raises(ValueError, match="duplicate problem id"):
        build_merged_candidate_outputs(
            sources=[
                CandidateSource("first", first_path, expected_records=1),
                CandidateSource("second", second_path, expected_records=1),
            ],
            output_dir=tmp_path / "merged",
            alm_diagnostics=diagnostics([first, second]),
        )


def test_merge_rejects_trace_bytes_that_do_not_reconstruct_response(
    tmp_path: Path,
) -> None:
    record = normalized_record("taco_train_1__attempt_1", problem_id="taco_train_1")
    record["content_tokens"][0]["bytes"] = [0]
    source_path = tmp_path / "source.jsonl"
    write_jsonl(source_path, [record])

    with pytest.raises(ValueError, match="do not reconstruct"):
        build_merged_candidate_outputs(
            sources=[CandidateSource("source", source_path, expected_records=1)],
            output_dir=tmp_path / "merged",
            alm_diagnostics=diagnostics([record]),
        )
