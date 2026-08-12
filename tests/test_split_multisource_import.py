from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.split_multisource_import import main


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


def test_split_import_preserves_order_and_builds_collectable_summaries(
    tmp_path: Path,
) -> None:
    tasks = [
        {
            "schema_version": "coding.task.multisource.v1",
            "id": f"codecontests_train_{index}",
        }
        for index in range(5)
    ]
    tasks_path = tmp_path / "tasks.jsonl"
    summary_path = tmp_path / "import.json"
    output_dir = tmp_path / "batches"
    _write_jsonl(tasks_path, tasks)
    ordered_ids = [task["id"] for task in tasks]
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "coding.import.multisource.v1",
                "status": "ok",
                "source": "code-contests",
                "tasks": 5,
                "selection": "random",
                "seed": 20260731,
                "dataset": {
                    "id": "deepmind/code_contests",
                    "config": "all",
                    "split": "train",
                    "revision": "pinned",
                    "license": "Apache-2.0",
                    "provenance": "https://example.test/original",
                    "mirror": "https://example.test/mirror",
                },
                "ordered_task_ids": ordered_ids,
                "ordered_task_ids_sha256": hashlib.sha256(
                    "\n".join(ordered_ids).encode("utf-8")
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    argv = [
        "--tasks",
        str(tasks_path),
        "--import-summary",
        str(summary_path),
        "--output-dir",
        str(output_dir),
        "--prefix",
        "code_contests_bulk",
        "--first-batch-size",
        "2",
        "--batch-size",
        "3",
    ]
    assert main(argv) == 0
    assert main(argv) == 0

    first_tasks = output_dir / "code_contests_bulk_batch_001_tasks_2.jsonl"
    second_tasks = output_dir / "code_contests_bulk_batch_002_tasks_3.jsonl"
    first_summary = output_dir / "code_contests_bulk_batch_001_import_2.json"
    manifest = json.loads(
        (output_dir / "code_contests_bulk_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert [task["id"] for task in _read_jsonl(first_tasks)] == ordered_ids[:2]
    assert [task["id"] for task in _read_jsonl(second_tasks)] == ordered_ids[2:]
    summary = json.loads(first_summary.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "coding.import.multisource.v1"
    assert summary["tasks"] == 2
    assert summary["ordered_task_ids"] == ordered_ids[:2]
    assert summary["derivation"]["start_index"] == 0
    assert summary["derivation"]["end_index_exclusive"] == 2
    assert manifest["total_tasks"] == 5
    assert [batch["tasks"] for batch in manifest["batches"]] == [2, 3]
    assert manifest["duplicate_task_ids"] == 0


def test_split_import_help_runs_as_a_direct_script() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/split_multisource_import.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--first-batch-size" in result.stdout
