from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from deepseek_distill.audit import (
    AuditPricing,
    build_audit_report,
    compute_alm_diagnostics,
    render_audit_markdown,
)
from deepseek_distill.coding_dataset import build_candidate_datasets
from deepseek_distill.cross_tokenizer_aligner import ByteOffsetEncoding
from deepseek_distill.offline_teacher import OfflineTeacherTraceProvider
from deepseek_distill.records import NORMALIZED_SCHEMA_VERSION, RAW_SCHEMA_VERSION


def write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def task(record_id: str, original_id: int) -> dict:
    return {
        "schema_version": "coding.task.mbpp.v1",
        "id": record_id,
        "source": {"dataset": "MBPP", "split": "train", "original_id": original_id},
        "problem_text": "Return a value.",
        "function_name": "answer",
        "function_signature": "answer()",
        "supporting_interfaces": [],
        "tests": ["assert answer() == 1"],
        "metadata": {"test_setup_code": "", "challenge_tests": []},
    }


def raw_success(record_id: str, task_record: dict) -> dict:
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "id": record_id,
        "status": "ok",
        "task": task_record,
        "request": {"generation_config": {"top_logprobs": 20}},
        "metrics": {"request_duration_seconds": 1.5},
        "response": {},
    }


def normalized(record_id: str, task_record: dict) -> dict:
    response = "def answer():\n    return 1\n"
    pieces = [response[:14].encode(), response[14:].encode()]
    top = [
        {"token": f"c{index}", "bytes": [index], "logprob": -5.0}
        for index in range(20)
    ]
    return {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "id": record_id,
        "task": task_record,
        "request": {
            "messages": [{"role": "user", "content": "problem"}],
            "generation_config": {"top_logprobs": 20},
        },
        "response_text": response,
        "content_tokens": [
            {
                "token": f"t{index}",
                "bytes": list(piece),
                "logprob": -0.1,
                "top_logprobs": top,
            }
            for index, piece in enumerate(pieces)
        ],
        "validation": {"content_bytes_match": True, "warnings": []},
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 10,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "metrics": {"request_duration_seconds": 1.5},
    }


def verification(record_id: str, *, category: str) -> dict:
    passed = category == "passed"
    return {
        "schema_version": "coding.verifier.mbpp.v1",
        "id": record_id,
        "status": "accepted" if passed else "rejected",
        "failure_category": category,
        "source_extraction": {"status": "passed", "removed_markdown_fence": False},
        "trace_validation": {"valid": True},
        "extracted_source": "def answer():\n    return 1\n",
        "phases": [
            {"name": "compile", "status": "passed"},
            {"name": "import", "status": "passed"},
            {"name": "test", "status": category},
        ],
    }


def test_candidate_builder_keeps_only_passed_normalized_records_as_accepted(tmp_path) -> None:
    first = task("mbpp_601", 601)
    second = task("mbpp_602", 602)
    third = task("mbpp_603", 603)
    raw_path = tmp_path / "raw.jsonl"
    normalized_path = tmp_path / "normalized.jsonl"
    verifier_path = tmp_path / "verifier.jsonl"
    accepted_path = tmp_path / "accepted.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    write_jsonl(
        raw_path,
        [
            raw_success("mbpp_601", first),
            raw_success("mbpp_602", second),
            {
                "schema_version": RAW_SCHEMA_VERSION,
                "id": "mbpp_603",
                "status": "error",
                "task": third,
                "error": {"type": "RateLimitError", "message": "slow down"},
            },
        ],
    )
    write_jsonl(normalized_path, [normalized("mbpp_601", first), normalized("mbpp_602", second)])
    write_jsonl(
        verifier_path,
        [verification("mbpp_601", category="passed"), verification("mbpp_602", category="assertion_failure")],
    )

    summary = build_candidate_datasets(
        raw_path=raw_path,
        normalized_path=normalized_path,
        verifier_path=verifier_path,
        accepted_path=accepted_path,
        rejected_path=rejected_path,
    )

    accepted = read_jsonl(accepted_path)
    rejected = read_jsonl(rejected_path)
    assert summary.total == 3
    assert summary.accepted == 1
    assert summary.rejected == 2
    assert accepted[0]["schema_version"] == NORMALIZED_SCHEMA_VERSION
    assert accepted[0]["coding_verification"]["failure_category"] == "passed"
    assert OfflineTeacherTraceProvider().get_trace(accepted[0]).response_text.startswith("def answer")
    assert {record["failure_category"] for record in rejected} == {
        "api_error",
        "assertion_failure",
    }


def test_candidate_builder_refuses_to_replace_outputs(tmp_path) -> None:
    for name in ("raw", "normalized", "verifier"):
        write_jsonl(tmp_path / f"{name}.jsonl", [])
    accepted_path = tmp_path / "accepted.jsonl"
    accepted_path.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_candidate_datasets(
            raw_path=tmp_path / "raw.jsonl",
            normalized_path=tmp_path / "normalized.jsonl",
            verifier_path=tmp_path / "verifier.jsonl",
            accepted_path=accepted_path,
            rejected_path=tmp_path / "rejected.jsonl",
        )

    assert accepted_path.read_text(encoding="utf-8") == "keep"


def test_audit_report_calculates_funnel_trace_availability_cost_and_alm() -> None:
    passed_task = task("mbpp_601", 601)
    failed_task = task("mbpp_602", 602)
    good_raw = raw_success("mbpp_601", passed_task)
    api_error = {
        "schema_version": RAW_SCHEMA_VERSION,
        "id": "mbpp_602",
        "status": "error",
        "task": failed_task,
        "error": {"type": "RuntimeError", "message": "offline"},
    }
    good_normalized = normalized("mbpp_601", passed_task)
    passed_verifier = verification("mbpp_601", category="passed")

    report = build_audit_report(
        tasks=[passed_task, failed_task],
        raw_records=[good_raw, api_error],
        normalized_records=[good_normalized],
        verifier_records=[passed_verifier],
        accepted_records=[good_normalized],
        pricing=AuditPricing(input_cache_hit_per_million=0.025, input_cache_miss_per_million=3, output_per_million=6),
        resumability={"total": 2, "skipped": 2, "succeeded": 0, "failed": 0},
        alm_diagnostics={
            "student_tokenizer": "Qwen/test",
            "student_revision": "abc123",
            "sequence_lengths": [100],
            "chunks_per_example": [9],
            "group_counts": {"1:1": 5, "1:N": 2, "N:1": 1, "N:M": 1},
            "prompt_completion_boundary_drops": 1,
            "examples_with_zero_valid_chunks": [],
            "records_exceeding_max_length": [],
            "max_length": 4096,
        },
    )

    assert report["counts"]["selected_tasks"] == 2
    assert report["rates"]["api_success"]["rate"] == 0.5
    assert report["rates"]["trace_reconstruction"]["rate"] == 1.0
    assert report["rates"]["official_unit_test_pass"]["rate"] == 0.5
    assert report["failure_counts"] == {"api_error": 1, "passed": 1}
    assert report["trace"]["actual_logprobs"]["available"] == 2
    assert report["trace"]["top20"]["positions_with_20"] == 2
    assert report["cost_rmb"]["total_estimated"] == pytest.approx(0.00006)
    assert report["cost_rmb"]["per_attempted_task"] == pytest.approx(0.00003)
    assert report["cost_rmb"]["per_accepted_task"] == pytest.approx(0.00006)
    assert report["resumability"]["skipped"] == 2
    assert report["alm"]["group_counts"]["1:N"] == 2

    markdown = render_audit_markdown(report)
    assert "# MBPP DeepSeek Collection Audit" in markdown
    assert "Official unit-test pass" in markdown
    assert "Qwen/test" in markdown
    assert "Length and latency distributions" in markdown
    assert "Token usage" in markdown


@dataclass
class AuditTokenizer:
    response: str
    context: str = "<ctx>"
    suffix: str = "<end>"
    eos_token_id: int = 13

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize is False
        if add_generation_prompt:
            return self.context
        return self.context + self.response + self.suffix

    def encode_with_byte_offsets(self, text: str, *, add_special_tokens: bool = False):
        assert text == self.context + self.response + self.suffix
        context_end = len(self.context.encode())
        response_end = context_end + len(self.response.encode())
        return ByteOffsetEncoding(
            token_ids=(10, 11, 12, 13),
            byte_offsets=(
                (0, context_end),
                (context_end, context_end + 1),
                (context_end + 1, response_end),
                (response_end, response_end + len(self.suffix.encode())),
            ),
        )


def test_compute_alm_diagnostics_reports_sequence_chunks_and_group_shapes() -> None:
    task_record = task("mbpp_601", 601)
    record = normalized("mbpp_601", task_record)
    record["response_text"] = "ab"
    record["content_tokens"] = [
        {"token": "a", "bytes": [97], "logprob": -0.1, "top_logprobs": []},
        {"token": "b", "bytes": [98], "logprob": -0.2, "top_logprobs": []},
    ]

    diagnostics = compute_alm_diagnostics(
        [record],
        tokenizer=AuditTokenizer(response="ab"),
        student_tokenizer="Qwen/test",
        student_revision="abc123",
        max_length=3,
    )

    assert diagnostics["sequence_lengths"] == [4]
    assert diagnostics["chunks_per_example"] == [2]
    assert diagnostics["group_counts"] == {"1:1": 2, "1:N": 0, "N:1": 0, "N:M": 0}
    assert diagnostics["examples_with_zero_valid_chunks"] == []
    assert diagnostics["records_exceeding_max_length"] == ["mbpp_601"]
    assert diagnostics["preprocessing_errors"] == []


def test_audit_counts_interrupted_normalized_record_without_verifier_as_malformed() -> None:
    first = task("mbpp_601", 601)
    second = task("mbpp_602", 602)

    report = build_audit_report(
        tasks=[first, second],
        raw_records=[raw_success("mbpp_601", first), raw_success("mbpp_602", second)],
        normalized_records=[normalized("mbpp_601", first), normalized("mbpp_602", second)],
        verifier_records=[verification("mbpp_601", category="passed")],
        accepted_records=[normalized("mbpp_601", first)],
        pricing=AuditPricing(0.025, 3, 6),
    )

    assert report["failure_counts"] == {"malformed_trace": 1, "passed": 1}
