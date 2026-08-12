from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepseek_distill.apps import APPS_REVISION
from deepseek_distill.multisource_tasks import make_multisource_task
from scripts.import_multisource import build_parser, main


def _task(index: int) -> dict:
    return make_multisource_task(
        task_id=f"apps_train_{index:06d}",
        source={
            "dataset": "codeparrot/apps",
            "config": "all",
            "split": "train",
            "original_id": index,
            "revision": APPS_REVISION,
            "license": "MIT",
            "provenance": "https://github.com/hendrycks/apps",
            "mirror": "https://huggingface.co/datasets/codeparrot/apps",
        },
        problem_text=f"Solve task {index}.",
        interface_type="stdin_stdout",
        required_interface=(
            "Complete Python program using standard input and standard output."
        ),
        tests=[{"input": "1\n", "output": "1\n"}],
        metadata={},
    )


def test_multisource_import_parser_has_explicit_immutable_outputs() -> None:
    args = build_parser().parse_args(
        [
            "--source",
            "apps",
            "--limit",
            "2",
            "--output",
            "tasks.jsonl",
            "--summary-output",
            "summary.json",
        ]
    )

    assert args.source == "apps"
    assert args.limit == 2
    assert args.selection == "random"
    assert args.seed == 20260731
    assert args.revision is None
    assert args.difficulty_profile is None


def test_multisource_import_passes_and_records_hard_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "tasks.jsonl"
    summary_output = tmp_path / "summary.json"
    calls: list[dict] = []

    def fake_loader(**kwargs: object) -> list[dict]:
        calls.append(dict(kwargs))
        return [_task(3)]

    assert (
        main(
            [
                "--source",
                "apps",
                "--limit",
                "1",
                "--difficulty-profile",
                "hard-v1",
                "--output",
                str(output),
                "--summary-output",
                str(summary_output),
            ],
            loaders={"apps": fake_loader},
        )
        == 0
    )
    capsys.readouterr()

    assert calls[0]["difficulty_profile"] == "hard-v1"
    assert json.loads(summary_output.read_text(encoding="utf-8"))[
        "difficulty_profile"
    ] == "hard-v1"


def test_multisource_import_is_idempotent_and_records_provenance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "tasks.jsonl"
    summary_output = tmp_path / "summary.json"
    calls: list[dict] = []

    def fake_loader(**kwargs: object) -> list[dict]:
        calls.append(dict(kwargs))
        return [_task(3), _task(7)]

    argv = [
        "--source",
        "apps",
        "--limit",
        "2",
        "--selection",
        "random",
        "--seed",
        "20260731",
        "--output",
        str(output),
        "--summary-output",
        str(summary_output),
    ]

    assert main(argv, loaders={"apps": fake_loader}) == 0
    first_runtime = json.loads(capsys.readouterr().out)
    assert main(argv, loaders={"apps": fake_loader}) == 0
    second_runtime = json.loads(capsys.readouterr().out)

    persisted = json.loads(summary_output.read_text(encoding="utf-8"))
    assert first_runtime["output_status"] == "created"
    assert second_runtime["output_status"] == "unchanged"
    assert persisted["dataset"] == {
        "id": "codeparrot/apps",
        "config": "all",
        "split": "train",
        "revision": APPS_REVISION,
        "license": "MIT",
        "provenance": "https://github.com/hendrycks/apps",
        "mirror": "https://huggingface.co/datasets/codeparrot/apps",
    }
    assert persisted["ordered_task_ids"] == [
        "apps_train_000003",
        "apps_train_000007",
    ]
    assert len(persisted["ordered_task_ids_sha256"]) == 64
    assert calls == [
        {
            "limit": 2,
            "selection": "random",
            "seed": 20260731,
            "revision": APPS_REVISION,
            "cache_dir": None,
        },
        {
            "limit": 2,
            "selection": "random",
            "seed": 20260731,
            "revision": APPS_REVISION,
            "cache_dir": None,
        },
    ]


def test_multisource_import_rejects_revision_drift_before_loading(
    tmp_path: Path,
) -> None:
    called = False

    def fake_loader(**kwargs: object) -> list[dict]:
        nonlocal called
        called = True
        return []

    with pytest.raises(ValueError, match="pinned revision"):
        main(
            [
                "--source",
                "apps",
                "--limit",
                "1",
                "--revision",
                "moving-main",
                "--output",
                str(tmp_path / "tasks.jsonl"),
                "--summary-output",
                str(tmp_path / "summary.json"),
            ],
            loaders={"apps": fake_loader},
        )

    assert called is False


def test_multisource_import_excludes_prior_tasks_before_publishing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prior_tasks = tmp_path / "prior_tasks.jsonl"
    prior_tasks.write_text(
        json.dumps(_task(3), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "new_tasks.jsonl"
    summary_output = tmp_path / "new_summary.json"
    calls: list[dict] = []

    def fake_loader(**kwargs: object) -> list[dict]:
        calls.append(dict(kwargs))
        return [_task(3), _task(7), _task(9)]

    assert (
        main(
            [
                "--source",
                "apps",
                "--limit",
                "2",
                "--exclude-tasks",
                str(prior_tasks),
                "--output",
                str(output),
                "--summary-output",
                str(summary_output),
            ],
            loaders={"apps": fake_loader},
        )
        == 0
    )
    capsys.readouterr()

    tasks = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert [task["id"] for task in tasks] == [
        "apps_train_000007",
        "apps_train_000009",
    ]
    assert calls[0]["limit"] == 3
    assert summary["exclusions"]["unique_task_ids"] == 1
    assert summary["exclusions"]["input_files"][0]["records"] == 1
    assert len(summary["exclusions"]["task_ids_sha256"]) == 64
    assert summary["loader_requested_tasks"] == 3


def test_multisource_import_rejects_exclusions_from_another_source(
    tmp_path: Path,
) -> None:
    foreign = _task(3)
    foreign["source"]["dataset"] = "deepmind/code_contests"
    prior_tasks = tmp_path / "foreign_tasks.jsonl"
    prior_tasks.write_text(json.dumps(foreign) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match source"):
        main(
            [
                "--source",
                "apps",
                "--limit",
                "1",
                "--exclude-tasks",
                str(prior_tasks),
                "--output",
                str(tmp_path / "new_tasks.jsonl"),
                "--summary-output",
                str(tmp_path / "new_summary.json"),
            ],
            loaders={"apps": lambda **_: [_task(7)]},
        )
