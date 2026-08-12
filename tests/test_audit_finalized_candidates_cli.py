from __future__ import annotations

import json
from pathlib import Path


def test_finalized_candidate_audit_does_not_assume_single_attempt_funnel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import audit_finalized_candidates as audit_cli

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    accepted = {
        "id": "apps_train_1__attempt_2",
        "source": {"dataset": "APPS"},
        "response_text": "def solve():\n    return 1\n",
        "content_tokens": [{"token": "x", "bytes": [120], "logprob": -0.1}],
    }
    (run_dir / "accepted_unique.jsonl").write_text(
        json.dumps(accepted) + "\n",
        encoding="utf-8",
    )
    (run_dir / "dataset_summary.json").write_text(
        json.dumps({"counts": {"selected_tasks": 10}}) + "\n",
        encoding="utf-8",
    )

    class FakeTokenizer:
        eos_token_id = 1

    monkeypatch.setattr(audit_cli, "load_tokenizer", lambda **kwargs: FakeTokenizer())
    monkeypatch.setattr(
        audit_cli,
        "compute_alm_diagnostics",
        lambda records, **kwargs: {
            "examples": [{"id": records[0]["id"], "sequence_length": 8}],
            "preprocessing_errors": [],
            "examples_with_zero_valid_chunks": [],
            "records_exceeding_max_length": [],
            "prompt_completion_boundary_drops": 0,
            "sequence_length_distribution": {"count": 1},
            "chunks_per_example_distribution": {"count": 1},
            "group_counts": {"1:1": 1, "1:N": 0, "N:1": 0, "N:M": 0},
        },
    )
    monkeypatch.setattr(
        audit_cli,
        "build_training_contract_report",
        lambda records, tokenizer: {
            "records": 1,
            "end_token_supervision": {
                "eos_present_records": 1,
                "eos_supervised_records": 1,
                "missing_eos_record_ids": [],
                "ignored_eos_record_ids": [],
            },
        },
    )
    monkeypatch.setattr(
        audit_cli,
        "build_multisource_clean_audit",
        lambda **kwargs: {
            "report": {
                "counts": {
                    "official_test_passed": 1,
                    "clean_eligible": 1,
                    "clean_excluded": 0,
                },
                "reason_counts": {},
            },
            "decisions": [{"id": accepted["id"], "eligible": True}],
            "retained_records": [accepted],
            "excluded_records": [],
        },
    )

    result = audit_cli.audit_finalized_candidates(
        run_dir=run_dir,
        student_tokenizer="fake",
        student_revision="revision",
        tokenizer_cache_dir=None,
        local_files_only=True,
        max_length=4096,
    )

    assert result["schema_version"] == "coding.audit.finalized_candidates.v1"
    assert result["collection_summary"]["counts"]["selected_tasks"] == 10
    assert result["training_started"] is False
    assert (run_dir / "clean_accepted.v1.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_finalized_candidate_audit_requires_accepted_records(tmp_path: Path) -> None:
    from scripts.audit_finalized_candidates import audit_finalized_candidates

    try:
        audit_finalized_candidates(
            run_dir=tmp_path,
            student_tokenizer="fake",
            student_revision="revision",
            tokenizer_cache_dir=None,
            local_files_only=True,
            max_length=4096,
        )
    except FileNotFoundError as error:
        assert "accepted_unique.jsonl" in str(error)
    else:
        raise AssertionError("missing accepted records must fail")
