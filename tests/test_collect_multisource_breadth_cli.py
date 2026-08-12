from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from deepseek_distill.apps import APPS_REVISION
from deepseek_distill.multisource_tasks import make_multisource_task
from scripts.collect_multisource_breadth import build_parser, main


def _task() -> dict:
    return make_multisource_task(
        task_id="apps_train_000017",
        source={
            "dataset": "codeparrot/apps",
            "config": "all",
            "split": "train",
            "original_id": 17,
            "revision": APPS_REVISION,
            "license": "MIT",
            "provenance": "https://github.com/hendrycks/apps",
            "mirror": "https://huggingface.co/datasets/codeparrot/apps",
        },
        problem_text="Read one integer and print it.",
        interface_type="stdin_stdout",
        required_interface=(
            "Complete Python program using standard input and standard output."
        ),
        tests=[{"input": "SECRET_TEST_INPUT\n", "output": "17\n"}],
        metadata={},
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_multisource_breadth_parser_has_conservative_cloud_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--source",
            "apps",
            "--tasks",
            "tasks.jsonl",
            "--import-summary",
            "import.json",
            "--run-dir",
            "run",
        ]
    )

    assert args.workers == 4
    assert args.verifier_workers == 12
    assert args.max_tokens == 4096
    assert args.top_logprobs == 20
    assert args.trace_profile == "top20"
    assert args.streaming_pipeline is False
    assert args.raw_only is False
    assert args.max_attempts_per_task == 1


def test_multisource_breadth_help_runs_as_a_direct_script() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_multisource_breadth.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--import-summary" in result.stdout
    assert "--streaming-pipeline" in result.stdout
    assert "--raw-only" in result.stdout


def test_raw_only_is_mutually_exclusive_with_other_execution_modes() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--source",
                "apps",
                "--tasks",
                "tasks.jsonl",
                "--import-summary",
                "import.json",
                "--run-dir",
                "run",
                "--raw-only",
                "--collect-only",
            ]
        )


def test_prepare_only_freezes_manifest_without_tests_or_api_key(
    tmp_path: Path,
) -> None:
    task = _task()
    tasks_path = tmp_path / "tasks.jsonl"
    import_summary_path = tmp_path / "import_summary.json"
    run_dir = tmp_path / "run"
    _write_jsonl(tasks_path, [task])
    ordered_hash = hashlib.sha256(task["id"].encode("utf-8")).hexdigest()
    import_summary_path.write_text(
        json.dumps(
            {
                "schema_version": "coding.import.multisource.v1",
                "status": "ok",
                "source": "apps",
                "tasks": 1,
                "selection": "random",
                "seed": 20260731,
                "dataset": {
                    "id": "codeparrot/apps",
                    "config": "all",
                    "split": "train",
                    "revision": APPS_REVISION,
                    "license": "MIT",
                    "provenance": "https://github.com/hendrycks/apps",
                    "mirror": "https://huggingface.co/datasets/codeparrot/apps",
                },
                "ordered_task_ids": [task["id"]],
                "ordered_task_ids_sha256": ordered_hash,
                "difficulty_profile": "hard-v1",
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--source",
                "apps",
                "--tasks",
                str(tasks_path),
                "--import-summary",
                str(import_summary_path),
                "--run-dir",
                str(run_dir),
                "--max-attempts-per-task",
                "3",
                "--prepare-only",
            ]
        )
        == 0
    )

    manifest_text = (run_dir / "campaign_manifest.json").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(manifest_text)
    assert "SECRET_TEST_INPUT" not in manifest_text
    assert "DEEPSEEK_API_KEY" not in manifest_text
    assert manifest["prompt_contract"]["id"] == "deepseek.python.clean.v2"
    assert manifest["sampling"] == {
        "max_attempts_per_task": 3,
        "stop_after_first_pass": True,
        "blind": True,
        "verifier_feedback": False,
        "strategy": "blind_rejection_sampling",
    }
    assert manifest["dataset"]["ordered_task_ids_sha256"] == ordered_hash
    assert manifest["dataset"]["difficulty_profile"] == "hard-v1"


def test_prepare_only_actual_only_manifest_omits_top_logprobs(
    tmp_path: Path,
) -> None:
    task = _task()
    tasks_path = tmp_path / "tasks.jsonl"
    import_summary_path = tmp_path / "import_summary.json"
    run_dir = tmp_path / "run"
    _write_jsonl(tasks_path, [task])
    ordered_hash = hashlib.sha256(task["id"].encode("utf-8")).hexdigest()
    import_summary_path.write_text(
        json.dumps(
            {
                "schema_version": "coding.import.multisource.v1",
                "status": "ok",
                "source": "apps",
                "tasks": 1,
                "selection": "random",
                "seed": 20260806,
                "dataset": {
                    "id": "codeparrot/apps",
                    "config": "all",
                    "split": "train",
                    "revision": APPS_REVISION,
                    "license": "MIT",
                    "provenance": "https://github.com/hendrycks/apps",
                    "mirror": "https://huggingface.co/datasets/codeparrot/apps",
                },
                "ordered_task_ids": [task["id"]],
                "ordered_task_ids_sha256": ordered_hash,
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--source",
            "apps",
            "--tasks",
            str(tasks_path),
            "--import-summary",
            str(import_summary_path),
            "--run-dir",
            str(run_dir),
            "--trace-profile",
            "actual_only",
            "--prepare-only",
        ]
    ) == 0

    generation = json.loads(
        (run_dir / "campaign_manifest.json").read_text(encoding="utf-8")
    )["generation"]
    assert generation["trace_profile"] == "actual_only"
    assert generation["logprobs"] is True
    assert "top_logprobs" not in generation
