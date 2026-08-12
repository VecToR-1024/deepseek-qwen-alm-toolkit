from __future__ import annotations

from copy import deepcopy
import subprocess
import sys

import pytest

from deepseek_distill.multisource_clean_audit import (
    build_multisource_clean_audit,
    eos_supervision_by_record,
)
from scripts.audit_multisource_breadth import (
    build_parser,
    resolve_transformers_cache_dir,
)


def _top_candidates() -> list[dict]:
    return [
        {
            "token": f"candidate_{index}",
            "bytes": [97 + index % 26],
            "logprob": -float(index + 1),
        }
        for index in range(20)
    ]


def _record(
    record_id: str,
    source: str,
    response_text: str = "print(1)",
) -> dict:
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
                "top_logprobs": _top_candidates(),
            }
        ],
        "validation": {"content_bytes_match": True},
        "task": {"source": {"dataset": source}, "tests": []},
        "coding_verification": {"failure_category": "passed"},
    }


def _training_contract(
    *,
    records: int,
    supervised: int,
    missing: list[str] | None = None,
    ignored: list[str] | None = None,
) -> dict:
    return {
        "records": records,
        "end_token_supervision": {
            "eos_present_records": records - len(missing or []),
            "eos_supervised_records": supervised,
            "missing_eos_record_ids": missing or [],
            "ignored_eos_record_ids": ignored or [],
        },
    }


def test_eos_supervision_reconstructs_per_record_status() -> None:
    result = eos_supervision_by_record(
        ["apps_1__attempt_1", "apps_2__attempt_1"],
        _training_contract(
            records=2,
            supervised=1,
            ignored=["apps_2__attempt_1"],
        ),
    )

    assert result == {
        "apps_1__attempt_1": True,
        "apps_2__attempt_1": False,
    }


def test_eos_supervision_rejects_an_inconsistent_aggregate() -> None:
    with pytest.raises(ValueError, match="does not reconcile"):
        eos_supervision_by_record(
            ["apps_1__attempt_1", "apps_2__attempt_1"],
            _training_contract(records=2, supervised=1),
        )


def test_clean_audit_keeps_only_records_passing_all_contracts() -> None:
    clean = _record("apps_1__attempt_1", "codeparrot/apps")
    fenced = _record(
        "xcodeeval_2__attempt_1",
        "NTU-NLP-sg/xCodeEval",
        "```python\nprint(1)\n```",
    )
    records = [clean, fenced]
    original = deepcopy(records)
    alm = {
        "examples": [
            {
                "id": record["id"],
                "sequence_length": 128,
                "valid_alm_chunks": 1,
                "prompt_completion_boundary_drops": 0,
            }
            for record in records
        ],
        "preprocessing_errors": [],
    }

    result = build_multisource_clean_audit(
        records=records,
        alm=alm,
        training_contract=_training_contract(records=2, supervised=2),
    )

    assert result["report"]["counts"] == {
        "official_test_passed": 2,
        "clean_eligible": 1,
        "clean_excluded": 1,
    }
    assert result["report"]["reason_counts"] == {
        "markdown_fence": 1,
        "syntax_error": 1,
    }
    assert [record["id"] for record in result["retained_records"]] == [
        "apps_1__attempt_1"
    ]
    assert result["excluded_records"] == [
        {
            "schema_version": "offline_alm.clean_exclusion.v1",
            "id": "xcodeeval_2__attempt_1",
            "source": "NTU-NLP-sg/xCodeEval",
            "reasons": ["markdown_fence", "syntax_error"],
        }
    ]
    assert records == original


def test_clean_audit_rejects_missing_alm_and_unsupervised_eos() -> None:
    first = _record("odex_1__attempt_1", "neulab/odex")
    second = _record("odex_2__attempt_1", "neulab/odex")

    result = build_multisource_clean_audit(
        records=[first, second],
        alm={
            "examples": [
                {
                    "id": first["id"],
                    "sequence_length": 128,
                    "valid_alm_chunks": 1,
                    "prompt_completion_boundary_drops": 0,
                }
            ],
            "preprocessing_errors": [
                {"id": second["id"], "type": "ValueError", "message": "bad"}
            ],
        },
        training_contract=_training_contract(
            records=2,
            supervised=1,
            missing=[second["id"]],
        ),
    )

    by_id = {decision["id"]: decision for decision in result["decisions"]}
    assert by_id[first["id"]]["eligible"] is True
    assert by_id[second["id"]]["reasons"] == [
        "alm_preprocessing_failure",
        "eos_not_supervised",
    ]
    assert result["report"]["alm_preprocessing_errors"] == 1


def test_multisource_audit_parser_pins_the_training_tokenizer_contract() -> None:
    args = build_parser().parse_args(["--run-dir", "run"])

    assert args.student_tokenizer == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert args.student_revision == "c03e6d358207e414f1eca0bb1891e29f1db0e242"
    assert args.max_length == 4096


def test_multisource_audit_help_runs_as_a_direct_script() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/audit_multisource_breadth.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--local-files-only" in result.stdout


def test_cache_dir_accepts_either_hf_home_or_the_hub_directory(tmp_path) -> None:
    hf_home = tmp_path / "hf-cache"
    hub = hf_home / "hub"
    hub.mkdir(parents=True)

    assert resolve_transformers_cache_dir(hf_home) == hub
    assert resolve_transformers_cache_dir(hub) == hub
