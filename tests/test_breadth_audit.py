from __future__ import annotations

import json
from pathlib import Path

from deepseek_distill.audit import AuditPricing
from deepseek_distill.breadth_audit import (
    build_single_attempt_breadth_audit,
    render_breadth_audit_markdown,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_breadth_audit_streams_trace_cost_and_pass_metrics(tmp_path: Path) -> None:
    ids = [
        "taco_train_000010__attempt_1",
        "taco_train_000020__attempt_1",
    ]
    write_jsonl(
        tmp_path / "raw_attempts.jsonl",
        [
            {
                "id": identifier,
                "status": "ok",
                "metrics": {"request_duration_seconds": 1.0 + index},
            }
            for index, identifier in enumerate(ids)
        ],
    )
    token = {
        "bytes": [120],
        "logprob": -0.1,
        "top_logprobs": [
            {"bytes": [120], "logprob": -0.1},
            {"bytes": [121], "logprob": -2.0},
        ],
    }
    write_jsonl(
        tmp_path / "normalized_attempts.jsonl",
        [
            {
                "id": identifier,
                "response_text": "x",
                "content_tokens": [token],
                "validation": {"content_bytes_match": True},
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "prompt_cache_hit_tokens": 5,
                    "prompt_cache_miss_tokens": 5,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            }
            for identifier in ids
        ],
    )
    write_jsonl(
        tmp_path / "verifier_attempts.jsonl",
        [
            {
                "id": ids[0],
                "failure_category": "passed",
                "source_extraction": {"status": "passed"},
                "phases": [
                    {"name": "compile", "status": "passed"},
                    {"name": "test_1", "status": "passed"},
                ],
            },
            {
                "id": ids[1],
                "failure_category": "assertion_failure",
                "source_extraction": {"status": "passed"},
                "phases": [
                    {"name": "compile", "status": "passed"},
                    {"name": "test_1", "status": "assertion_failure"},
                ],
            },
        ],
    )
    (tmp_path / "breadth_summary.json").write_text(
        json.dumps(
            {
                "counts": {
                    "selected_tasks": 2,
                    "raw_attempts": 2,
                    "normalized_attempts": 2,
                    "verifier_results": 2,
                    "accepted_unique": 1,
                },
                "failure_categories": {"assertion_failure": 1},
                "finish_reasons": {"stop": 2},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "campaign_manifest.json").write_text(
        json.dumps({"schema_version": "coding.collection.taco.breadth.v2"}),
        encoding="utf-8",
    )

    report = build_single_attempt_breadth_audit(
        run_dir=tmp_path,
        pricing=AuditPricing(0.025, 3.0, 6.0),
        alm={"preprocessing_errors": [], "examples": [{"id": ids[0]}]},
    )

    assert report["rates"]["api_success"]["rate"] == 1.0
    assert report["rates"]["trace_reconstruction"]["rate"] == 1.0
    assert report["rates"]["unique_task_pass"]["rate"] == 0.5
    assert report["failure_counts"] == {"assertion_failure": 1}
    assert report["trace"]["actual_logprobs"]["available"] == 2
    assert report["trace"]["top20"]["candidate_count"] == 4
    assert report["cost_rmb"]["total_estimated"] == 0.00015025
    assert report["duplicates"]["raw_attempt_ids"] == 0
    assert report["resumability"]["resume_safe"] is True

    markdown = render_breadth_audit_markdown(report)
    assert "Pass@1" in markdown
    assert "ALM preprocessing" in markdown
