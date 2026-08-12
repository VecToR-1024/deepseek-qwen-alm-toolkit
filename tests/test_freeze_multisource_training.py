from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.freeze_multisource_training import main


def _record(problem_id: str, dataset: str, problem_text: str) -> dict:
    response_text = "print(input())"
    return {
        "schema_version": "deepseek.teacher.normalized.v1",
        "id": f"{problem_id}__attempt_1",
        "finish_reason": "stop",
        "response_text": response_text,
        "content_tokens": [
            {
                "token": response_text,
                "bytes": list(response_text.encode("utf-8")),
                "logprob": -0.1,
                "top_logprobs": [],
            }
        ],
        "validation": {"content_bytes_match": True, "warnings": []},
        "task": {
            "problem_text": problem_text,
            "source": {"dataset": dataset},
            "tests": ["1\n", "1\n"],
        },
        "sampling": {"problem_id": problem_id, "attempt_number": 1},
        "coding_verification": {"failure_category": "passed"},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_freeze_round_robins_sources_and_preserves_per_source_order(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "frozen"
    _write_jsonl(
        first,
        [
            _record("a1", "source-a", "Alpha one"),
            _record("a2", "source-a", "Alpha two"),
            _record("b1", "source-b", "Beta one"),
        ],
    )
    _write_jsonl(
        second,
        [
            _record("c1", "source-c", "Gamma one"),
            _record("b2", "source-b", "Beta two"),
        ],
    )

    argv = [
        "--input",
        f"first={first}",
        "--input",
        f"second={second}",
        "--target",
        "4",
        "--output-dir",
        str(output),
    ]
    assert main(argv) == 0
    assert main(argv) == 0

    selected = _read_jsonl(output / "training_records.jsonl")
    reserve = _read_jsonl(output / "reserve_records.jsonl")
    assert [row["sampling"]["problem_id"] for row in selected] == [
        "a1",
        "b1",
        "c1",
        "a2",
    ]
    assert [row["sampling"]["problem_id"] for row in reserve] == ["b2"]


def test_freeze_removes_exact_normalized_problem_duplicates(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "frozen"
    _write_jsonl(first, [_record("a1", "source-a", "Add  ONE value!")])
    _write_jsonl(
        second,
        [
            _record("b1", "source-b", "add one value"),
            _record("b2", "source-b", "Subtract one value"),
        ],
    )

    assert (
        main(
            [
                "--input",
                f"first={first}",
                "--input",
                f"second={second}",
                "--target",
                "2",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    manifest = json.loads((output / "dataset_manifest.json").read_text("utf-8"))
    excluded = _read_jsonl(output / "excluded_records.jsonl")
    assert manifest["counts"]["input_records"] == 3
    assert manifest["counts"]["exact_duplicates_excluded"] == 1
    assert excluded[0]["id"] == "b1__attempt_1"
    assert excluded[0]["freeze_exclusion"]["duplicate_of"] == "a1__attempt_1"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"finish_reason": "length"}, "finish_reason"),
        ({"response_text": "```python\nprint(1)\n```"}, "Markdown fence"),
    ],
)
def test_freeze_rejects_records_that_violate_the_clean_contract(
    tmp_path: Path,
    mutation: dict,
    message: str,
) -> None:
    path = tmp_path / "input.jsonl"
    record = _record("a1", "source-a", "Echo one value")
    record.update(mutation)
    _write_jsonl(path, [record])

    with pytest.raises(ValueError, match=message):
        main(
            [
                "--input",
                f"source={path}",
                "--target",
                "1",
                "--output-dir",
                str(tmp_path / "frozen"),
            ]
        )


def test_freeze_reports_near_duplicates_without_removing_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.jsonl"
    output = tmp_path / "frozen"
    _write_jsonl(
        path,
        [
            _record(
                "a1",
                "source-a",
                "Given a list of integers, return the largest integer in the list.",
            ),
            _record(
                "b1",
                "source-b",
                "Given a list of integers return the largest number in the list.",
            ),
        ],
    )

    assert (
        main(
            [
                "--input",
                f"source={path}",
                "--target",
                "2",
                "--near-threshold",
                "0.75",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    manifest = json.loads((output / "dataset_manifest.json").read_text("utf-8"))
    assert manifest["near_duplicates"]["reported_pairs"] == 1
    assert len(_read_jsonl(output / "training_records.jsonl")) == 2


def test_freeze_all_keeps_every_unique_clean_record(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    output = tmp_path / "frozen"
    _write_jsonl(
        path,
        [
            _record("a1", "source-a", "Alpha task"),
            _record("a2", "source-a", "Beta task"),
        ],
    )

    assert (
        main(
            [
                "--input",
                f"source={path}",
                "--target",
                "all",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    manifest = json.loads((output / "dataset_manifest.json").read_text("utf-8"))
    assert manifest["selection"]["target"] == "all"
    assert manifest["counts"]["training_records"] == 2
    assert manifest["counts"]["reserve_records"] == 0
