from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from deepseek_distill.alignment_pipeline import (
    ALIGNED_SCHEMA_VERSION,
    align_jsonl,
    align_normalized_record,
)
from deepseek_distill.cross_tokenizer_aligner import (
    ByteOffsetEncoding,
    CrossTokenizerAligner,
)
from deepseek_distill.records import NORMALIZED_SCHEMA_VERSION
from deepseek_distill.validate import validate_jsonl


class FakeChatTokenizer:
    name_or_path = "fake-qwen"

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        if add_generation_prompt:
            return "<ctx>"
        assert messages[-1]["role"] == "assistant"
        return f"<ctx>{messages[-1]['content']}<end>"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return {
            "<ctx>": [100],
            "<ctx>OK": [100, 11],
            "<ctx>O": [100, 12],
            "<ctx>OK<end>": [100, 11, 999],
        }[text]


class FakeByteOffsetTokenizer:
    def encode_with_byte_offsets(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> ByteOffsetEncoding:
        assert add_special_tokens is False
        assert text == "<ctx>OK<end>"
        return ByteOffsetEncoding(
            token_ids=(100, 11, 999),
            byte_offsets=((0, 5), (5, 7), (7, 12)),
        )


class MissingResponseByteOffsetTokenizer:
    def encode_with_byte_offsets(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> ByteOffsetEncoding:
        assert add_special_tokens is False
        assert text == "<ctx>OK<end>"
        return ByteOffsetEncoding(
            token_ids=(100,),
            byte_offsets=((0, 5),),
        )


def diagnostic_aligner() -> CrossTokenizerAligner:
    return CrossTokenizerAligner(FakeByteOffsetTokenizer())


def normalized_record(record_id: str = "sample") -> dict:
    return {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "id": record_id,
        "request": {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "Reply OK"}],
            "generation_config": {"top_logprobs": 20},
        },
        "teacher_model": "deepseek-v4-pro",
        "finish_reason": "stop",
        "response_text": "OK",
        "content_tokens": [
            {
                "token": "OK",
                "bytes": [79, 75],
                "logprob": math.log(0.8),
                "top_logprobs": [
                    {"token": "OK", "bytes": [79, 75], "logprob": math.log(0.8)},
                    {"token": "O", "bytes": [79], "logprob": math.log(0.1)},
                ],
            }
        ],
        "validation": {"warnings": []},
    }


def test_align_normalized_record_emits_stable_training_contract() -> None:
    output = align_normalized_record(
        normalized_record(),
        tokenizer=FakeChatTokenizer(),
        student_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        tokenizer_revision="test-revision",
    )

    assert output["schema_version"] == ALIGNED_SCHEMA_VERSION
    assert output["source_normalized_schema_version"] == NORMALIZED_SCHEMA_VERSION
    assert output["student_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert output["student_tokenizer"]["name_or_path"] == "fake-qwen"
    assert output["student_tokenizer"]["revision"] == "test-revision"
    assert output["student_input_ids"] == [100, 11, 999]
    assert output["student_generation_context_ids"] == [100]
    assert output["soft_positions"][0]["mapped_student_token_ids"] == [11, 12]
    assert output["soft_positions"][0]["teacher_tail_prob"] == pytest.approx(0.1)
    assert output["alignment_stats"]["aligned_position_ratio"] == 1.0
    assert "alignment_diagnostics" not in output


def test_optional_span_diagnostics_do_not_change_strict_training_targets() -> None:
    baseline = align_normalized_record(
        normalized_record(),
        tokenizer=FakeChatTokenizer(),
        student_model="Qwen/Qwen2.5-Coder-7B-Instruct",
    )

    output = align_normalized_record(
        normalized_record(),
        tokenizer=FakeChatTokenizer(),
        student_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        span_aligner=diagnostic_aligner(),
    )

    assert output["student_input_ids"] == baseline["student_input_ids"]
    assert output["soft_positions"] == baseline["soft_positions"]
    diagnostic = output["alignment_diagnostics"]
    assert diagnostic["training_alignment"] == "strict_1_to_1"
    assert diagnostic["span_status"] == "aligned"
    assert diagnostic["span_error"] is None
    assert diagnostic["span_stats"]["teacher_position_coverage"] == 1.0
    assert diagnostic["comparison"]["strict_position_coverage"] == 1.0
    assert diagnostic["comparison"]["span_position_coverage"] == 1.0
    assert diagnostic["comparison"]["total_teacher_topk_mass"] == pytest.approx(0.9)
    assert diagnostic["comparison"]["strict_retained_topk_mass"] == pytest.approx(0.9)
    assert diagnostic["comparison"]["span_covered_topk_mass"] == pytest.approx(0.9)
    assert diagnostic["comparison"]["loss_ready_topk_mass"] == pytest.approx(0.9)
    assert output["soft_positions"][0]["teacher_tail_prob"] == pytest.approx(0.1)


def test_align_jsonl_atomically_writes_records_and_aggregate_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "normalized.jsonl"
    output_path = tmp_path / "aligned.jsonl"
    input_path.write_text(
        json.dumps(normalized_record(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = align_jsonl(
        input_path,
        output_path,
        tokenizer=FakeChatTokenizer(),
        student_model="Qwen/Qwen2.5-Coder-7B-Instruct",
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["id"] == "sample"
    assert summary.total_records == 1
    assert summary.records_with_alignment == 1
    assert summary.teacher_positions == 1
    assert summary.aligned_positions == 1
    assert summary.aligned_position_ratio == 1.0
    assert summary.diagnostics is None

    report = validate_jsonl(output_path)
    assert report.valid_records == 1
    assert report.invalid_records == 0


def test_align_jsonl_aggregates_opt_in_span_diagnostics(tmp_path: Path) -> None:
    input_path = tmp_path / "normalized.jsonl"
    output_path = tmp_path / "aligned.jsonl"
    input_path.write_text(
        json.dumps(normalized_record(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = align_jsonl(
        input_path,
        output_path,
        tokenizer=FakeChatTokenizer(),
        student_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        span_aligner=diagnostic_aligner(),
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["alignment_diagnostics"]["span_status"] == "aligned"
    assert summary.diagnostics is not None
    assert summary.diagnostics.attempted_records == 1
    assert summary.diagnostics.comparable_records == 1
    assert summary.diagnostics.strict_fallback_records == 0
    assert summary.diagnostics.teacher_positions == 1
    assert summary.diagnostics.span_aligned_positions == 1
    assert summary.diagnostics.span_position_coverage == 1.0
    assert summary.diagnostics.total_teacher_topk_mass == pytest.approx(0.9)
    assert summary.diagnostics.strict_retained_topk_mass == pytest.approx(0.9)
    assert summary.diagnostics.span_covered_topk_mass == pytest.approx(0.9)
    assert summary.diagnostics.loss_ready_topk_mass == pytest.approx(0.9)


def test_align_jsonl_reports_span_failure_without_dropping_strict_targets(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "normalized.jsonl"
    output_path = tmp_path / "aligned.jsonl"
    input_path.write_text(json.dumps(normalized_record()) + "\n", encoding="utf-8")
    span_aligner = CrossTokenizerAligner(MissingResponseByteOffsetTokenizer())

    summary = align_jsonl(
        input_path,
        output_path,
        tokenizer=FakeChatTokenizer(),
        student_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        span_aligner=span_aligner,
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["alignment_diagnostics"]["span_status"] == "strict_fallback"
    assert written["alignment_diagnostics"]["comparison"] is None
    assert written["soft_positions"][0]["mapped_student_token_ids"] == [11, 12]
    assert written["soft_positions"][0]["teacher_tail_prob"] == pytest.approx(0.1)
    assert summary.diagnostics is not None
    assert summary.diagnostics.comparable_records == 0
    assert summary.diagnostics.strict_fallback_records == 1
    assert summary.diagnostics.strict_position_coverage == 1.0
    assert summary.diagnostics.span_position_coverage == 0.0
    assert summary.diagnostics.total_teacher_topk_mass == 0.0


def test_align_jsonl_preserves_existing_output_without_force(tmp_path: Path) -> None:
    input_path = tmp_path / "normalized.jsonl"
    output_path = tmp_path / "aligned.jsonl"
    input_path.write_text(json.dumps(normalized_record()) + "\n", encoding="utf-8")
    output_path.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        align_jsonl(
            input_path,
            output_path,
            tokenizer=FakeChatTokenizer(),
            student_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        )

    assert output_path.read_text(encoding="utf-8") == "keep me"


def test_validator_rejects_aligned_probabilities_that_do_not_sum_to_one(tmp_path: Path) -> None:
    output = align_normalized_record(
        normalized_record(),
        tokenizer=FakeChatTokenizer(),
        student_model="Qwen/Qwen2.5-Coder-7B-Instruct",
    )
    output["soft_positions"][0]["teacher_tail_prob"] = 0.9
    path = tmp_path / "invalid-aligned.jsonl"
    path.write_text(json.dumps(output) + "\n", encoding="utf-8")

    report = validate_jsonl(path)

    assert report.valid_records == 0
    assert report.invalid_records == 1
    assert "sum to one" in report.issues[0].message
