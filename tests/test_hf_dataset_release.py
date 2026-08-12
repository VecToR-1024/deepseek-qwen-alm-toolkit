from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from deepseek_distill.hf_dataset_release import (
    audit_hf_package,
    package_hf_dataset,
    project_record,
    upload_hf_package,
)
from scripts import release_hf_dataset


def _record(record_id: str = "apps_1__attempt_1") -> dict:
    response_text = "def add(a, b):\n    return a + b\n"
    response_bytes = list(response_text.encode("utf-8"))
    return {
        "schema_version": "deepseek.teacher.normalized.v1",
        "id": record_id,
        "api_response_id": "secret-provider-id",
        "system_fingerprint": "provider-fingerprint",
        "teacher_model": "deepseek-v4-pro",
        "finish_reason": "stop",
        "request": {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "system", "content": "Return Python source."},
                {"role": "user", "content": "Add two integers."},
            ],
            "generation_config": {
                "temperature": 0.2,
                "top_p": 1.0,
                "logprobs": True,
                "top_logprobs": 20,
            },
            "prompt_contract": "coding.teacher.v1",
        },
        "response_text": response_text,
        "content_tokens": [
            {
                "token": response_text,
                "bytes": response_bytes,
                "logprob": -0.25,
                "top_logprobs": [
                    {"token": "pass", "bytes": [112, 97, 115, 115], "logprob": -2.0}
                ],
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 12},
        "task": {
            "schema_version": "coding.task.multisource.v1",
            "id": "apps_1",
            "source": {
                "dataset": "APPS",
                "config": "competition",
                "split": "train",
                "original_id": "1",
                "revision": "abc123",
                "license": "MIT",
                "provenance": "https://github.com/hendrycks/apps",
                "mirror": "https://huggingface.co/datasets/codeparrot/apps",
                "raw_file": "C:\\Users\\alice\\private.jsonl",
            },
            "problem_text": "Add two integers.",
            "interface_type": "function",
            "required_interface": "def add(a, b)",
            "tests": ["assert add(1, 2) == 3"],
            "metadata": {"reference_solution": "do not publish"},
        },
        "sampling": {
            "problem_id": "apps_1",
            "attempt_id": "apps_1__attempt_1",
            "attempt_number": 1,
            "selection_index": 0,
        },
        "provider": {"base_url": "https://internal.example/v1"},
        "coding_verification": {
            "failure_category": "passed",
            "artifact": {"path": "C:\\Users\\alice\\artifact.py"},
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )


def _read_gzip_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_project_record_keeps_exact_alm_trace_without_private_tests() -> None:
    projected = project_record(_record())

    assert projected["schema_version"] == "deepseek.teacher.normalized.v1"
    assert projected["request"]["messages"][1]["content"] == "Add two integers."
    assert projected["response_text"].startswith("def add")
    assert projected["content_tokens"] == [
        {
            "bytes": list(projected["response_text"].encode("utf-8")),
            "logprob": -0.25,
        }
    ]
    assert projected["task"]["source"]["license"] == "MIT"
    assert "tests" not in projected["task"]
    assert "metadata" not in projected["task"]
    assert projected["release_redactions"] == {"local_path_replacements": 0}
    assert projected["request"]["generation_config"]["top_logprobs"] == 20
    assert all("top_logprobs" not in row for row in projected["content_tokens"])

    serialized = json.dumps(projected, ensure_ascii=False)
    for forbidden in (
        "secret-provider-id",
        "provider-fingerprint",
        "internal.example",
        "C:\\\\Users",
        "assert add(1, 2)",
        "reference_solution",
    ):
        assert forbidden not in serialized


def test_project_record_rejects_a_secret_in_a_teacher_prompt() -> None:
    record = _record()
    fake_token = "hf_" + "abcdefghijklmnop"
    record["request"]["messages"][1]["content"] = f"token={fake_token}"

    with pytest.raises(ValueError, match="credential-like value"):
        project_record(record)


def test_project_record_redacts_prompt_paths_but_never_rewrites_completion() -> None:
    record = _record()
    sensitive_prompt = (
        "Read /root/private/task.txt and "
        "C:\\Users\\alice\\Documents\\input.txt."
    )
    record["request"]["messages"][1]["content"] = sensitive_prompt
    record["task"]["problem_text"] = sensitive_prompt

    projected = project_record(record)

    assert projected["request"]["messages"][1]["content"] == (
        "Read <LOCAL_PATH> and <LOCAL_PATH>"
    )
    assert projected["task"]["problem_text"] == (
        "Read <LOCAL_PATH> and <LOCAL_PATH>"
    )
    assert projected["release_redactions"] == {"local_path_replacements": 4}
    assert projected["response_text"] == record["response_text"]

    response_with_path = 'def location():\n    return "/root/private/result.txt"\n'
    record["response_text"] = response_with_path
    record["content_tokens"] = [
        {
            "bytes": list(response_with_path.encode("utf-8")),
            "logprob": -0.25,
        }
    ]
    with pytest.raises(ValueError, match="local filesystem path"):
        project_record(record)


def test_package_is_deterministic_sharded_and_documents_mixed_licenses(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "training_records.jsonl"
    records = [_record(f"row_{index}__attempt_1") for index in range(3)]
    records[1]["task"]["source"]["dataset"] = "TACO"
    records[1]["task"]["source"]["license"] = "Apache-2.0 with upstream caveats"
    _write_jsonl(input_path, records)

    first = tmp_path / "release-first"
    second = tmp_path / "release-second"
    first_manifest = package_hf_dataset(
        input_path=input_path,
        output_dir=first,
        config_name="trained_2041",
        repo_id="owner/deepseek-qwen-alm-traces",
        records_per_shard=2,
    )
    second_manifest = package_hf_dataset(
        input_path=input_path,
        output_dir=second,
        config_name="trained_2041",
        repo_id="owner/deepseek-qwen-alm-traces",
        records_per_shard=2,
    )

    first_shards = sorted((first / "data" / "trained_2041").glob("*.jsonl.gz"))
    second_shards = sorted((second / "data" / "trained_2041").glob("*.jsonl.gz"))
    assert len(first_shards) == 2
    assert [_sha256(path) for path in first_shards] == [
        _sha256(path) for path in second_shards
    ]
    assert [row["id"] for path in first_shards for row in _read_gzip_jsonl(path)] == [
        "row_0__attempt_1",
        "row_1__attempt_1",
        "row_2__attempt_1",
    ]
    assert first_manifest == second_manifest
    assert first_manifest["counts"]["records"] == 3
    assert first_manifest["source_counts"] == {"APPS": 2, "TACO": 1}
    assert first_manifest["redactions"]["official_tests"] == 3
    assert first_manifest["redactions"]["prompt_local_path_replacements"] == 0

    card = (first / "README.md").read_text(encoding="utf-8")
    assert "license: other" in card
    assert "trained_2041" in card
    assert "APPS" in card and "TACO" in card
    assert "Official tests are intentionally excluded" in card


def test_package_refuses_wrong_authoritative_sha_or_nonempty_output(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "training_records.jsonl"
    _write_jsonl(input_path, [_record()])

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        package_hf_dataset(
            input_path=input_path,
            output_dir=tmp_path / "wrong-sha",
            config_name="trained_2041",
            repo_id="owner/repo",
            expected_sha256="0" * 64,
        )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        package_hf_dataset(
            input_path=input_path,
            output_dir=occupied,
            config_name="trained_2041",
            repo_id="owner/repo",
        )


def test_release_audit_revalidates_hashes_trace_and_sensitive_fields(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "training_records.jsonl"
    _write_jsonl(input_path, [_record("first"), _record("second")])
    package_dir = tmp_path / "release"
    package_hf_dataset(
        input_path=input_path,
        output_dir=package_dir,
        config_name="trained_2041",
        repo_id="owner/repo",
        records_per_shard=1,
    )

    audit = audit_hf_package(package_dir)

    assert audit["counts"] == {
        "duplicate_ids": 0,
        "records": 2,
        "shards": 2,
        "unique_ids": 2,
    }
    assert audit["sensitive_scan"] == {
        "credential_like_values": 0,
        "forbidden_keys": 0,
        "local_paths": 0,
    }
    assert audit["trace_reconstruction_records"] == 2

    first_shard = sorted((package_dir / "data" / "trained_2041").glob("*.gz"))[0]
    with first_shard.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match=r"shard (?:byte size|SHA256) mismatch"):
        audit_hf_package(package_dir)


class _FakeHfApi:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.uploaded: list[dict] = []
        self.remote_files: list[str] = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)
        package_dir = Path(kwargs["folder_path"])
        self.remote_files = sorted(
            path.relative_to(package_dir).as_posix()
            for path in package_dir.rglob("*")
            if path.is_file()
        )
        return "https://huggingface.co/datasets/owner/repo/commit/fake"

    def list_repo_files(self, **kwargs):
        return self.remote_files


def test_upload_creates_private_dataset_repo_and_verifies_every_package_file(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "training_records.jsonl"
    _write_jsonl(input_path, [_record()])
    package_dir = tmp_path / "release"
    package_hf_dataset(
        input_path=input_path,
        output_dir=package_dir,
        config_name="trained_2041",
        repo_id="owner/repo",
    )
    api = _FakeHfApi()

    result = upload_hf_package(
        package_dir=package_dir,
        repo_id="owner/repo",
        private=True,
        confirmed_manifest_sha256=_sha256(package_dir / "release_manifest.json"),
        api=api,
    )

    assert api.created == [
        {
            "repo_id": "owner/repo",
            "repo_type": "dataset",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert api.uploaded[0]["repo_type"] == "dataset"
    assert api.uploaded[0]["ignore_patterns"] == ["**/.git/**", "**/__pycache__/**"]
    assert result["visibility"] == "private"
    assert result["verified_files"] == 3


def test_upload_refuses_repo_id_that_differs_from_release_manifest(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "training_records.jsonl"
    _write_jsonl(input_path, [_record()])
    package_dir = tmp_path / "release"
    package_hf_dataset(
        input_path=input_path,
        output_dir=package_dir,
        config_name="trained_2041",
        repo_id="owner/correct",
    )

    with pytest.raises(ValueError, match="manifest repo_id"):
        upload_hf_package(
            package_dir=package_dir,
            repo_id="owner/wrong",
            private=True,
            confirmed_manifest_sha256=_sha256(
                package_dir / "release_manifest.json"
            ),
            api=_FakeHfApi(),
        )


def test_release_cli_packages_data_and_upload_is_dry_run_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "training_records.jsonl"
    output_dir = tmp_path / "release"
    _write_jsonl(input_path, [_record()])

    assert release_hf_dataset.main(
        [
            "package",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--config-name",
            "trained_2041",
            "--repo-id",
            "owner/repo",
        ]
    ) == 0
    packaged = json.loads(capsys.readouterr().out)
    assert packaged["event"] == "hf_dataset_package_created"
    assert packaged["records"] == 1

    assert release_hf_dataset.main(
        ["audit", "--package-dir", str(output_dir)]
    ) == 0
    audited = json.loads(capsys.readouterr().out)
    assert audited["event"] == "hf_dataset_release_audited"
    assert audited["counts"]["records"] == 1
    assert audited["sensitive_scan"]["forbidden_keys"] == 0

    monkeypatch.setattr(
        release_hf_dataset,
        "upload_hf_package",
        lambda **_: pytest.fail("dry-run must not call the Hugging Face API"),
    )
    assert release_hf_dataset.main(
        [
            "upload",
            "--package-dir",
            str(output_dir),
            "--repo-id",
            "owner/repo",
        ]
    ) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run == {
        "event": "hf_dataset_upload_plan",
        "execute": False,
        "manifest_sha256": _sha256(output_dir / "release_manifest.json"),
        "package_dir": str(output_dir),
        "repo_id": "owner/repo",
        "visibility": "private",
    }


def test_upload_requires_human_confirmation_of_the_exact_manifest(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "training_records.jsonl"
    _write_jsonl(input_path, [_record()])
    package_dir = tmp_path / "release"
    package_hf_dataset(
        input_path=input_path,
        output_dir=package_dir,
        config_name="trained_2041",
        repo_id="owner/repo",
    )

    with pytest.raises(ValueError, match="human-confirmed manifest SHA256"):
        upload_hf_package(
            package_dir=package_dir,
            repo_id="owner/repo",
            private=True,
            confirmed_manifest_sha256="0" * 64,
            api=_FakeHfApi(),
        )
