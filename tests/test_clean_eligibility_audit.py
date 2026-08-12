from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deepseek_distill.clean_eligibility_audit import (
    build_clean_eligibility_outputs,
)


def _record(record_id: str, response_text: str) -> dict:
    candidates = [
        {
            "token": f"candidate_{index}",
            "bytes": [97 + index % 26],
            "logprob": -float(index + 1),
        }
        for index in range(20)
    ]
    return {
        "schema_version": "deepseek.teacher.normalized.v1",
        "id": record_id,
        "request": {"generation_config": {"top_logprobs": 20}},
        "finish_reason": "stop",
        "response_text": response_text,
        "content_tokens": [
            {
                "token": response_text,
                "bytes": list(response_text.encode("utf-8")),
                "logprob": -0.1,
                "top_logprobs": candidates,
            }
        ],
        "validation": {"content_bytes_match": True},
        "task": {"source": {"dataset": "fixture"}, "tests": []},
        "coding_verification": {
            "status": "accepted",
            "failure_category": "passed",
            "source_extraction": {
                "removed_markdown_fence": response_text.startswith("```")
            },
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_supporting_inputs(
    tmp_path: Path,
    *,
    training_sha256: str,
) -> tuple[Path, Path]:
    diagnostics = tmp_path / "alm_diagnostics.json"
    diagnostics.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "id": "plain",
                        "sequence_length": 100,
                        "valid_alm_chunks": 10,
                        "prompt_completion_boundary_drops": 0,
                    },
                    {
                        "id": "fenced",
                        "sequence_length": 100,
                        "valid_alm_chunks": 10,
                        "prompt_completion_boundary_drops": 0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    eos_audit = tmp_path / "eos_audit.json"
    eos_audit.write_text(
        json.dumps(
            {
                "inputs": {"training_data_sha256": training_sha256},
                "training_contract": {
                    "records": 2,
                    "end_token_supervision": {
                        "eos_supervised_records": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return diagnostics, eos_audit


def test_build_outputs_streams_retained_records_and_lightweight_exclusions(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training.jsonl"
    plain = _record("plain", "def solve(x):\n    return x + 1")
    fenced = _record("fenced", "```python\nprint(1)\n```")
    _write_jsonl(training, [plain, fenced])
    diagnostics, eos_audit = _write_supporting_inputs(
        tmp_path,
        training_sha256=_sha256(training),
    )
    output_dir = tmp_path / "clean"

    report = build_clean_eligibility_outputs(
        training_data=training,
        alm_diagnostics=diagnostics,
        eos_attestation=eos_audit,
        output_dir=output_dir,
    )

    assert report["counts"] == {
        "total": 2,
        "eligible": 1,
        "excluded": 1,
    }
    assert report["reason_counts"] == {
        "markdown_fence": 1,
        "syntax_error": 1,
    }
    retained = [
        json.loads(line)
        for line in (output_dir / "existing_v3_retained.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    excluded = [
        json.loads(line)
        for line in (output_dir / "existing_v3_excluded.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    eligibility = [
        json.loads(line)
        for line in (output_dir / "existing_v3_eligibility.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert retained == [plain]
    assert excluded == [
        {
            "schema_version": "offline_alm.clean_exclusion.v1",
            "id": "fenced",
            "source": "fixture",
            "reasons": ["markdown_fence", "syntax_error"],
        }
    ]
    assert [row["eligible"] for row in eligibility] == [True, False]
    assert (output_dir / "existing_v3_clean_audit.json").exists()
    assert (output_dir / "existing_v3_clean_audit.md").exists()
    assert all(
        len(metadata["sha256"]) == 64 for metadata in report["outputs"].values()
    )


def test_build_outputs_refuses_to_overwrite_existing_artifacts(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training.jsonl"
    _write_jsonl(training, [_record("plain", "print(1)")])
    diagnostics = tmp_path / "alm.json"
    diagnostics.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "id": "plain",
                        "sequence_length": 10,
                        "valid_alm_chunks": 1,
                        "prompt_completion_boundary_drops": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    eos = tmp_path / "eos.json"
    eos.write_text(
        json.dumps(
            {
                "inputs": {"training_data_sha256": _sha256(training)},
                "training_contract": {
                    "records": 1,
                    "end_token_supervision": {"eos_supervised_records": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "clean"
    output.mkdir()
    (output / "existing_v3_eligibility.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_clean_eligibility_outputs(
            training_data=training,
            alm_diagnostics=diagnostics,
            eos_attestation=eos,
            output_dir=output,
        )


def test_build_outputs_rejects_eos_attestation_for_another_dataset(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training.jsonl"
    _write_jsonl(training, [_record("plain", "print(1)")])
    diagnostics, eos_audit = _write_supporting_inputs(
        tmp_path,
        training_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="SHA-256"):
        build_clean_eligibility_outputs(
            training_data=training,
            alm_diagnostics=diagnostics,
            eos_attestation=eos_audit,
            output_dir=tmp_path / "clean",
        )
