from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.deploy_hard_training import (
    build_parser,
    build_remote_start_command,
    validate_frozen_dataset,
)


def _frozen_dataset(tmp_path: Path) -> Path:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    data = frozen / "training_records.jsonl"
    data.write_text('{"id":"one"}\n', encoding="utf-8")
    payload = data.read_bytes()
    manifest = {
        "schema_version": "offline_alm.frozen_multisource_training.v1",
        "counts": {"training_records": 1},
        "outputs": {
            "training_records": {
                "path": data.as_posix(),
                "records": 1,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
        "training_started": False,
    }
    (frozen / "dataset_manifest.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )
    return frozen


def test_validate_frozen_dataset_reconciles_count_size_and_hash(tmp_path: Path) -> None:
    frozen = _frozen_dataset(tmp_path)

    validated = validate_frozen_dataset(frozen)

    assert validated["records"] == 1
    assert validated["data_path"] == frozen / "training_records.jsonl"
    assert len(validated["data_sha256"]) == 64


def test_validate_frozen_dataset_rejects_changed_payload(tmp_path: Path) -> None:
    frozen = _frozen_dataset(tmp_path)
    (frozen / "training_records.jsonl").write_text(
        '{"id":"changed"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="byte-size|SHA-256"):
        validate_frozen_dataset(frozen)


def test_deploy_connection_and_remote_paths_are_required() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--local-frozen-dir", "frozen"])

    args = parser.parse_args(
        [
            "--local-frozen-dir",
            "frozen",
            "--host",
            "ssh.example.invalid",
            "--port",
            "2222",
            "--user",
            "trainer",
            "--remote-upload-root",
            "/srv/alm/data/frozen",
            "--remote-run-root",
            "/srv/alm/runs/example",
        ]
    )

    assert args.host == "ssh.example.invalid"
    assert args.port == 2222
    assert args.user == "trainer"
    assert args.remote_upload_root == "/srv/alm/data/frozen"
    assert args.remote_run_root == "/srv/alm/runs/example"


def test_remote_start_command_is_non_destructive_and_detached() -> None:
    command = build_remote_start_command(
        upload_root="/srv/alm/data/new/frozen_all",
        run_root="/srv/alm/experiments/new-run",
    )

    assert "refusing_existing_upload" in command
    assert "refusing_started_run" in command
    assert "nohup bash" in command
    assert "supervisor.pid" in command
    assert "supervisor.log" in command
