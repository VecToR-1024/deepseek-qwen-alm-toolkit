"""Offline JSONL preprocessing for strict teacher/student token alignment."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .alignment import AlignmentError, TokenizerEncoder, align_teacher_content
from .cross_tokenizer_aligner import (
    AlignmentDiagnosticResult,
    CrossTokenizerAligner,
    diagnose_with_strict_fallback,
)
from .records import NORMALIZED_SCHEMA_VERSION

ALIGNED_SCHEMA_VERSION = "deepseek.teacher.aligned.v1"


class ChatTokenizer(TokenizerEncoder, Protocol):
    name_or_path: str

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class AlignmentDatasetSummary:
    total_records: int
    records_with_alignment: int
    records_without_alignment: int
    teacher_positions: int
    aligned_positions: int
    aligned_position_ratio: float
    diagnostics: AlignmentDiagnosticsDatasetSummary | None = None


@dataclass(frozen=True, slots=True)
class AlignmentDiagnosticsDatasetSummary:
    attempted_records: int
    comparable_records: int
    strict_fallback_records: int
    teacher_positions: int
    strict_aligned_positions: int
    span_aligned_positions: int
    strict_position_coverage: float
    span_position_coverage: float
    total_teacher_topk_mass: float
    strict_retained_topk_mass: float
    span_covered_topk_mass: float
    loss_ready_topk_mass: float
    strict_retained_topk_mass_ratio: float
    span_covered_topk_mass_ratio: float


def align_normalized_record(
    record: Mapping[str, Any],
    *,
    tokenizer: ChatTokenizer,
    student_model: str,
    tokenizer_revision: str | None = None,
    span_aligner: CrossTokenizerAligner | None = None,
) -> dict[str, Any]:
    """Add a stable student-token alignment contract to one normalized record."""
    record_id, messages, response_text, content_tokens = _record_fields(record)
    # Hugging Face documents add_generation_prompt=True as the assistant-start
    # marker for generation:
    # https://huggingface.co/docs/transformers/chat_templating#addgenerationprompt
    generation_context = tokenizer.apply_chat_template(
        copy.deepcopy(messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    full_messages = copy.deepcopy(messages)
    full_messages.append({"role": "assistant", "content": response_text})
    # For the final training sequence, the documented preprocessing form uses
    # the complete assistant message with add_generation_prompt=False:
    # https://huggingface.co/docs/transformers/chat_templating#model-training
    full_training_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(generation_context, str) or not isinstance(full_training_text, str):
        raise AlignmentError("tokenizer.apply_chat_template must return text when tokenize=False")

    diagnostic: AlignmentDiagnosticResult | None = None
    if span_aligner is None:
        result = align_teacher_content(
            tokenizer,
            context_text=generation_context,
            response_text=response_text,
            content_tokens=content_tokens,
            student_full_text=full_training_text,
        )
    else:
        diagnostic = diagnose_with_strict_fallback(
            strict_tokenizer=tokenizer,
            span_aligner=span_aligner,
            context_text=generation_context,
            response_text=response_text,
            content_tokens=content_tokens,
            student_full_text=full_training_text,
        )
        result = diagnostic.training_result
    context_ids = tokenizer.encode(generation_context, add_special_tokens=False)
    stats = asdict(result.stats)
    stats.update(
        {
            "aligned_position_ratio": result.stats.aligned_position_ratio,
            "mean_mapped_candidates_per_aligned_position": (
                result.stats.mean_mapped_candidates_per_aligned_position
            ),
            "candidate_mapping_ratio": result.stats.candidate_mapping_ratio,
        }
    )

    output = copy.deepcopy(dict(record))
    output["schema_version"] = ALIGNED_SCHEMA_VERSION
    output["source_normalized_schema_version"] = NORMALIZED_SCHEMA_VERSION
    output["id"] = record_id
    output["student_model"] = student_model
    output["student_tokenizer"] = {
        "name_or_path": str(getattr(tokenizer, "name_or_path", student_model)),
        "revision": tokenizer_revision,
    }
    output["student_generation_context_ids"] = list(context_ids)
    output["student_input_ids"] = list(result.student_input_ids)
    output["soft_positions"] = [
        {
            "teacher_position": item.teacher_position,
            "student_logit_position": item.student_logit_position,
            "mapped_student_token_ids": list(item.mapped_student_token_ids),
            "teacher_probs": list(item.teacher_probs),
            "teacher_tail_prob": item.teacher_tail_prob,
            "mapped_teacher_tokens": list(item.mapped_teacher_tokens),
        }
        for item in result.soft_positions
    ]
    output["alignment_stats"] = stats
    if diagnostic is not None:
        output["alignment_diagnostics"] = _serialize_diagnostic(diagnostic)
    return output


def align_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    tokenizer: ChatTokenizer,
    student_model: str,
    tokenizer_revision: str | None = None,
    span_aligner: CrossTokenizerAligner | None = None,
    force: bool = False,
) -> AlignmentDatasetSummary:
    """Align a normalized JSONL dataset and atomically publish the result."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    total_records = 0
    records_with_alignment = 0
    teacher_positions = 0
    aligned_positions = 0
    comparable_records = 0
    strict_fallback_records = 0
    span_aligned_positions = 0
    total_teacher_topk_masses: list[float] = []
    strict_retained_topk_masses: list[float] = []
    span_covered_topk_masses: list[float] = []
    loss_ready_topk_masses: list[float] = []
    seen_ids: set[str] = set()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_handle:
            temporary_path = Path(output_handle.name)
            with input_path.open("r", encoding="utf-8") as input_handle:
                for line_number, line in enumerate(input_handle, start=1):
                    if not line.strip():
                        continue
                    record = _parse_record(input_path, line_number, line)
                    record_id = record.get("id")
                    if not isinstance(record_id, str) or not record_id:
                        raise ValueError(f"{input_path}:{line_number}: id must be a non-empty string")
                    if record_id in seen_ids:
                        raise ValueError(f"{input_path}:{line_number}: duplicate id {record_id!r}")
                    seen_ids.add(record_id)
                    try:
                        aligned = align_normalized_record(
                            record,
                            tokenizer=tokenizer,
                            student_model=student_model,
                            tokenizer_revision=tokenizer_revision,
                            span_aligner=span_aligner,
                        )
                    except (AlignmentError, ValueError) as error:
                        raise ValueError(f"{input_path}:{line_number}: {error}") from error
                    output_handle.write(
                        json.dumps(aligned, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    total_records += 1
                    position_count = aligned["alignment_stats"]["aligned_positions"]
                    records_with_alignment += position_count > 0
                    teacher_positions += aligned["alignment_stats"]["teacher_positions"]
                    aligned_positions += position_count
                    diagnostic = aligned.get("alignment_diagnostics")
                    if isinstance(diagnostic, Mapping):
                        if diagnostic["span_status"] == "strict_fallback":
                            strict_fallback_records += 1
                        else:
                            comparable_records += 1
                            span_aligned_positions += diagnostic["span_stats"][
                                "aligned_teacher_positions"
                            ]
                            comparison = diagnostic["comparison"]
                            total_teacher_topk_masses.append(
                                comparison["total_teacher_topk_mass"]
                            )
                            strict_retained_topk_masses.append(
                                comparison["strict_retained_topk_mass"]
                            )
                            span_covered_topk_masses.append(
                                comparison["span_covered_topk_mass"]
                            )
                            loss_ready_topk_masses.append(
                                comparison["loss_ready_topk_mass"]
                            )
            output_handle.flush()
            os.fsync(output_handle.fileno())

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    ratio = 0.0 if teacher_positions == 0 else aligned_positions / teacher_positions
    diagnostics = None
    if span_aligner is not None:
        total_topk_mass = math.fsum(total_teacher_topk_masses)
        strict_retained_mass = math.fsum(strict_retained_topk_masses)
        span_covered_mass = math.fsum(span_covered_topk_masses)
        loss_ready_mass = math.fsum(loss_ready_topk_masses)
        diagnostics = AlignmentDiagnosticsDatasetSummary(
            attempted_records=total_records,
            comparable_records=comparable_records,
            strict_fallback_records=strict_fallback_records,
            teacher_positions=teacher_positions,
            strict_aligned_positions=aligned_positions,
            span_aligned_positions=span_aligned_positions,
            strict_position_coverage=ratio,
            span_position_coverage=_ratio(span_aligned_positions, teacher_positions),
            total_teacher_topk_mass=total_topk_mass,
            strict_retained_topk_mass=strict_retained_mass,
            span_covered_topk_mass=span_covered_mass,
            loss_ready_topk_mass=loss_ready_mass,
            strict_retained_topk_mass_ratio=_ratio(strict_retained_mass, total_topk_mass),
            span_covered_topk_mass_ratio=_ratio(span_covered_mass, total_topk_mass),
        )
    return AlignmentDatasetSummary(
        total_records=total_records,
        records_with_alignment=records_with_alignment,
        records_without_alignment=total_records - records_with_alignment,
        teacher_positions=teacher_positions,
        aligned_positions=aligned_positions,
        aligned_position_ratio=ratio,
        diagnostics=diagnostics,
    )


def _serialize_diagnostic(diagnostic: AlignmentDiagnosticResult) -> dict[str, Any]:
    span_stats = None
    if diagnostic.span_result is not None:
        span_stats = asdict(diagnostic.span_result.stats)
        span_stats.update(
            {
                "teacher_position_coverage": (
                    diagnostic.span_result.stats.teacher_position_coverage
                ),
                "student_position_coverage": (
                    diagnostic.span_result.stats.student_position_coverage
                ),
            }
        )
    return {
        "training_alignment": "strict_1_to_1",
        "span_status": "strict_fallback" if diagnostic.used_strict_fallback else "aligned",
        "span_error": diagnostic.span_error,
        "span_stats": span_stats,
        "comparison": (
            None if diagnostic.comparison is None else asdict(diagnostic.comparison)
        ),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _record_fields(
    record: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], str, list[Mapping[str, Any]]]:
    if record.get("schema_version") != NORMALIZED_SCHEMA_VERSION:
        raise AlignmentError(f"schema_version must be {NORMALIZED_SCHEMA_VERSION!r}")
    record_id = record.get("id")
    request = record.get("request")
    response_text = record.get("response_text")
    content_tokens = record.get("content_tokens")
    if not isinstance(record_id, str) or not record_id:
        raise AlignmentError("id must be a non-empty string")
    if not isinstance(request, Mapping):
        raise AlignmentError("request must be an object")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise AlignmentError("request.messages must be a non-empty list")
    if any(not isinstance(message, Mapping) for message in messages):
        raise AlignmentError("request.messages must contain objects")
    if not isinstance(response_text, str):
        raise AlignmentError("response_text must be a string")
    if not isinstance(content_tokens, list):
        raise AlignmentError("content_tokens must be a list")
    return record_id, [dict(message) for message in messages], response_text, content_tokens


def _parse_record(path: Path, line_number: int, line: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}:{line_number}: each JSONL value must be an object")
    return dict(value)
