from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_clean_multisource_target import main


def _record(problem_id: str, source: str) -> dict:
    return {
        "schema_version": "deepseek.teacher.normalized.v1",
        "id": f"{problem_id}__attempt_1",
        "finish_reason": "stop",
        "task": {"source": {"dataset": source}},
        "sampling": {"problem_id": problem_id, "attempt_number": 1},
        "coding_verification": {"failure_category": "passed"},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_build_clean_target_preserves_input_order_and_reserve(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "calibration.jsonl"
    bulk = tmp_path / "bulk.jsonl"
    output_dir = tmp_path / "target"
    _write_jsonl(
        calibration,
        [_record("apps_1", "codeparrot/apps"), _record("odex_1", "neulab/odex")],
    )
    _write_jsonl(
        bulk,
        [
            _record("codecontests_1", "deepmind/code_contests"),
            _record("taco_1", "BAAI/TACO"),
        ],
    )
    argv = [
        "--input",
        f"calibration={calibration}",
        "--input",
        f"bulk={bulk}",
        "--target",
        "3",
        "--output-dir",
        str(output_dir),
    ]

    assert main(argv) == 0
    assert main(argv) == 0

    selected = _read_jsonl(output_dir / "accepted_first_3.jsonl")
    reserve = _read_jsonl(output_dir / "reserve_clean.jsonl")
    manifest = json.loads(
        (output_dir / "clean_target_manifest.json").read_text(encoding="utf-8")
    )
    assert [row["sampling"]["problem_id"] for row in selected] == [
        "apps_1",
        "odex_1",
        "codecontests_1",
    ]
    assert [row["sampling"]["problem_id"] for row in reserve] == ["taco_1"]
    assert manifest["counts"] == {
        "all_new_clean": 4,
        "target": 3,
        "accepted_first_target": 3,
        "reserve": 1,
        "shortfall": 0,
    }
    assert manifest["input_order"] == ["calibration", "bulk"]
    assert manifest["duplicates"] == {"record_ids": 0, "problem_ids": 0}


def test_build_clean_target_refuses_duplicate_problem_ids(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [_record("apps_1", "codeparrot/apps")])
    duplicate = _record("apps_1", "codeparrot/apps")
    duplicate["id"] = "apps_1__attempt_2"
    _write_jsonl(second, [duplicate])

    with pytest.raises(ValueError, match="duplicate problem id"):
        main(
            [
                "--input",
                f"first={first}",
                "--input",
                f"second={second}",
                "--target",
                "1",
                "--output-dir",
                str(tmp_path / "target"),
            ]
        )


def test_build_clean_target_refuses_to_freeze_a_shortfall(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    output_dir = tmp_path / "target"
    _write_jsonl(source, [_record("apps_1", "codeparrot/apps")])

    with pytest.raises(RuntimeError, match="shortfall"):
        main(
            [
                "--input",
                f"source={source}",
                "--target",
                "2",
                "--output-dir",
                str(output_dir),
            ]
        )

    assert not output_dir.exists()
