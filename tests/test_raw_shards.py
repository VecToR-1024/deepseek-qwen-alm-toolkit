from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deepseek_distill.raw_shards import seal_raw_shard


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _terminal_batch(tmp_path: Path) -> dict[str, Path]:
    raw = tmp_path / "raw_attempts.jsonl"
    normalized = tmp_path / "normalized_attempts.jsonl"
    normalization_errors = tmp_path / "normalization_errors.jsonl"
    verifier = tmp_path / "verifier_attempts.jsonl"
    state = tmp_path / "pipeline_state.json"
    _write_jsonl(
        raw,
        [
            {
                "schema_version": "deepseek.teacher.raw.v1",
                "id": "task_1__attempt_1",
                "status": "ok",
            },
            {
                "schema_version": "deepseek.teacher.raw.v1",
                "id": "task_2__attempt_1",
                "status": "error",
            },
        ],
    )
    _write_jsonl(
        normalized,
        [
            {
                "schema_version": "deepseek.teacher.normalized.v1",
                "id": "task_1__attempt_1",
            }
        ],
    )
    normalization_errors.write_text("", encoding="utf-8")
    _write_jsonl(
        verifier,
        [{"id": "task_1__attempt_1", "failure_category": "passed"}],
    )
    state.write_text(
        json.dumps(
            {
                "schema_version": "deepseek.durable.pipeline.state.v1",
                "phase": "completed",
                "queues": {
                    "raw": 2,
                    "normalized": 1,
                    "normalization_errors": 0,
                    "verifier": 1,
                    "raw_to_normalized_lag": 0,
                    "normalized_to_verifier_lag": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "raw_path": raw,
        "normalized_path": normalized,
        "normalization_errors_path": normalization_errors,
        "verifier_path": verifier,
        "state_path": state,
    }


def test_seal_raw_shard_roundtrips_and_is_resumable(tmp_path: Path) -> None:
    paths = _terminal_batch(tmp_path)
    archive = tmp_path / "raw_attempts.jsonl.zst"
    seal_manifest = tmp_path / "raw_attempts.seal.json"
    archive_manifest = tmp_path / "raw_attempts.archive.json"

    first = seal_raw_shard(
        **paths,
        archive_path=archive,
        seal_manifest_path=seal_manifest,
        archive_manifest_path=archive_manifest,
        expected_records=2,
    )
    archive_before = archive.read_bytes()
    second = seal_raw_shard(
        **paths,
        archive_path=archive,
        seal_manifest_path=seal_manifest,
        archive_manifest_path=archive_manifest,
        expected_records=2,
    )

    assert first.status == "created"
    assert second.status == "unchanged"
    assert archive.read_bytes() == archive_before
    assert first.records == 2
    assert first.successful_records == 1
    assert first.api_error_records == 1
    assert first.source_sha256 == hashlib.sha256(
        paths["raw_path"].read_bytes()
    ).hexdigest()
    seal = json.loads(seal_manifest.read_text(encoding="utf-8"))
    manifest = json.loads(archive_manifest.read_text(encoding="utf-8"))
    assert seal["downstream_acknowledged"] is True
    assert manifest["verification"]["decompressed_sha256"] == first.source_sha256
    assert manifest["verification"]["decompressed_records"] == 2


def test_seal_raw_shard_refuses_unverified_success(tmp_path: Path) -> None:
    paths = _terminal_batch(tmp_path)
    paths["verifier_path"].write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="successful raw IDs without verifier"):
        seal_raw_shard(
            **paths,
            archive_path=tmp_path / "raw.zst",
            seal_manifest_path=tmp_path / "seal.json",
            archive_manifest_path=tmp_path / "archive.json",
            expected_records=2,
        )


def test_seal_raw_shard_detects_corrupted_existing_archive(tmp_path: Path) -> None:
    paths = _terminal_batch(tmp_path)
    archive = tmp_path / "raw.zst"
    kwargs = {
        **paths,
        "archive_path": archive,
        "seal_manifest_path": tmp_path / "seal.json",
        "archive_manifest_path": tmp_path / "archive.json",
        "expected_records": 2,
    }
    seal_raw_shard(**kwargs)
    archive.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="existing archive"):
        seal_raw_shard(**kwargs)


def test_remove_source_requires_verified_archive(tmp_path: Path) -> None:
    paths = _terminal_batch(tmp_path)
    kwargs = {
        **paths,
        "archive_path": tmp_path / "raw.zst",
        "seal_manifest_path": tmp_path / "seal.json",
        "archive_manifest_path": tmp_path / "archive.json",
        "expected_records": 2,
        "remove_source": True,
    }

    result = seal_raw_shard(**kwargs)
    resumed = seal_raw_shard(**kwargs)

    assert result.source_removed is True
    assert resumed.status == "unchanged"
    assert resumed.source_removed is True
    assert not paths["raw_path"].exists()


def test_resume_without_source_rejects_inconsistent_verification_manifest(
    tmp_path: Path,
) -> None:
    paths = _terminal_batch(tmp_path)
    archive_manifest = tmp_path / "archive.json"
    kwargs = {
        **paths,
        "archive_path": tmp_path / "raw.zst",
        "seal_manifest_path": tmp_path / "seal.json",
        "archive_manifest_path": archive_manifest,
        "expected_records": 2,
        "remove_source": True,
    }
    seal_raw_shard(**kwargs)
    manifest = json.loads(archive_manifest.read_text(encoding="utf-8"))
    manifest["verification"]["decompressed_sha256"] = "0" * 64
    archive_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="archive manifest verification"):
        seal_raw_shard(**kwargs)


def test_resume_after_archive_creation_rejects_changed_compression_policy(
    tmp_path: Path,
) -> None:
    paths = _terminal_batch(tmp_path)
    archive_manifest = tmp_path / "archive.json"
    kwargs = {
        **paths,
        "archive_path": tmp_path / "raw.zst",
        "seal_manifest_path": tmp_path / "seal.json",
        "archive_manifest_path": archive_manifest,
        "expected_records": 2,
        "compression_level": 6,
    }
    seal_raw_shard(**kwargs)
    archive_manifest.unlink()

    with pytest.raises(FileExistsError, match="different content"):
        seal_raw_shard(**(kwargs | {"compression_level": 7}))
