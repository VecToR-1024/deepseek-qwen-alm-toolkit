"""Collection-funnel, trace-quality, cost, and ALM audit calculations."""

from __future__ import annotations

import copy
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .alm_preprocessing import ALMExampleBuilder
from .cross_tokenizer_aligner import (
    CrossTokenizerAligner,
    HuggingFaceByteOffsetTokenizer,
)
from .offline_teacher import OfflineTeacherTraceProvider


AUDIT_SCHEMA_VERSION = "coding.audit.mbpp.v1"


@dataclass(frozen=True, slots=True)
class AuditPricing:
    input_cache_hit_per_million: float
    input_cache_miss_per_million: float
    output_per_million: float
    currency: str = "CNY"

    def __post_init__(self) -> None:
        for value in (
            self.input_cache_hit_per_million,
            self.input_cache_miss_per_million,
            self.output_per_million,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("pricing values must be finite and non-negative")


def compute_alm_diagnostics(
    records: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    student_tokenizer: str,
    student_revision: str,
    max_length: int = 4096,
) -> dict[str, Any]:
    """Run current ALM preprocessing on accepted records without model training."""
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    byte_tokenizer = (
        tokenizer
        if hasattr(tokenizer, "encode_with_byte_offsets")
        else HuggingFaceByteOffsetTokenizer(tokenizer)
    )
    builder = ALMExampleBuilder(tokenizer, byte_offset_tokenizer=byte_tokenizer)
    aligner = CrossTokenizerAligner(byte_tokenizer)
    trace_provider = OfflineTeacherTraceProvider()
    sequence_lengths: list[int] = []
    chunks_per_example: list[int] = []
    group_counts: Counter[str] = Counter({"1:1": 0, "1:N": 0, "N:1": 0, "N:M": 0})
    boundary_drops = 0
    zero_chunks: list[str] = []
    over_limit: list[str] = []
    errors: list[dict[str, str]] = []
    examples: list[dict[str, Any]] = []
    for record in records:
        record_id = record.get("id")
        try:
            example = builder.build(record)
            trace = trace_provider.get_trace(record)
            request = record.get("request")
            if not isinstance(request, Mapping) or not isinstance(request.get("messages"), list):
                raise ValueError("accepted record request.messages is missing")
            messages = copy.deepcopy(request["messages"])
            generation_context = tokenizer.apply_chat_template(
                copy.deepcopy(messages), tokenize=False, add_generation_prompt=True
            )
            full_messages = copy.deepcopy(messages)
            full_messages.append({"role": "assistant", "content": trace.response_text})
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            alignment = aligner.align(
                context_text=generation_context,
                response_text=trace.response_text,
                student_full_text=full_text,
                teacher_token_bytes=trace.token_bytes,
            )
        except Exception as error:  # noqa: BLE001 - diagnostics must retain per-record failures
            errors.append(
                {
                    "id": str(record_id),
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue

        sequence_length = len(example["input_ids"])
        chunk_count = int(example["alm_chunk_count"])
        dropped = int(example["alm_dropped_boundary_chunks"])
        counts = {
            "1:1": alignment.stats.one_to_one_groups,
            "1:N": alignment.stats.one_teacher_to_many_student_groups,
            "N:1": alignment.stats.many_teacher_to_one_student_groups,
            "N:M": alignment.stats.many_to_many_groups,
        }
        sequence_lengths.append(sequence_length)
        chunks_per_example.append(chunk_count)
        group_counts.update(counts)
        boundary_drops += dropped
        if chunk_count == 0:
            zero_chunks.append(str(record_id))
        if sequence_length > max_length:
            over_limit.append(str(record_id))
        examples.append(
            {
                "id": record_id,
                "sequence_length": sequence_length,
                "valid_alm_chunks": chunk_count,
                "group_counts": counts,
                "prompt_completion_boundary_drops": dropped,
            }
        )
    return {
        "student_tokenizer": student_tokenizer,
        "student_revision": student_revision,
        "max_length": max_length,
        "sequence_lengths": sequence_lengths,
        "sequence_length_distribution": _distribution(sequence_lengths),
        "chunks_per_example": chunks_per_example,
        "chunks_per_example_distribution": _distribution(chunks_per_example),
        "group_counts": {key: group_counts[key] for key in ("1:1", "1:N", "N:1", "N:M")},
        "prompt_completion_boundary_drops": boundary_drops,
        "examples_with_zero_valid_chunks": zero_chunks,
        "records_exceeding_max_length": over_limit,
        "preprocessing_errors": errors,
        "examples": examples,
    }


def build_audit_report(
    *,
    tasks: Sequence[Mapping[str, Any]],
    raw_records: Sequence[Mapping[str, Any]],
    normalized_records: Sequence[Mapping[str, Any]],
    verifier_records: Sequence[Mapping[str, Any]],
    accepted_records: Sequence[Mapping[str, Any]],
    pricing: AuditPricing,
    resumability: Mapping[str, Any] | None = None,
    alm_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate a machine-readable report without mutating input records."""
    selected_count = len(tasks)
    normalized_by_id = _last_by_id(normalized_records)
    verifier_by_id = _last_by_id(verifier_records)
    accepted_ids = {
        record.get("id") for record in accepted_records if isinstance(record.get("id"), str)
    }
    raw_success_ids = {
        record.get("id") for record in raw_records if record.get("status") == "ok"
    }
    trace_valid_ids = {
        record_id
        for record_id, record in normalized_by_id.items()
        if isinstance(record.get("validation"), Mapping)
        and record["validation"].get("content_bytes_match") is True
    }
    extraction_ids = {
        record_id
        for record_id, record in verifier_by_id.items()
        if isinstance(record.get("source_extraction"), Mapping)
        and record["source_extraction"].get("status") == "passed"
    }
    compile_ids = _passed_phase_ids(verifier_by_id, "compile")
    import_ids = _passed_phase_ids(verifier_by_id, "import")

    failure_counts: Counter[str] = Counter()
    for raw in raw_records:
        record_id = raw.get("id")
        if raw.get("status") == "error":
            failure_counts["api_error"] += 1
        elif record_id in verifier_by_id:
            category = verifier_by_id[record_id].get("failure_category")
            failure_counts[category if isinstance(category, str) else "malformed_trace"] += 1
        else:
            failure_counts["malformed_trace"] += 1

    content_tokens = [
        token
        for record in normalized_records
        for token in (record.get("content_tokens") or [])
        if isinstance(token, Mapping)
    ]
    top_candidates = [
        candidate
        for token in content_tokens
        for candidate in (token.get("top_logprobs") or [])
        if isinstance(candidate, Mapping)
    ]
    top_counts = [
        len(token.get("top_logprobs") or [])
        if isinstance(token.get("top_logprobs"), list)
        else 0
        for token in content_tokens
    ]
    response_byte_lengths = [
        len(record.get("response_text", "").encode("utf-8"))
        for record in normalized_records
        if isinstance(record.get("response_text"), str)
    ]
    response_token_lengths = [
        len(record.get("content_tokens") or []) for record in normalized_records
    ]
    prompt_tokens = [
        int(record.get("usage", {}).get("prompt_tokens"))
        for record in normalized_records
        if isinstance(record.get("usage"), Mapping)
        and isinstance(record["usage"].get("prompt_tokens"), int)
    ]
    latencies = [
        float(record["metrics"]["request_duration_seconds"])
        for record in raw_records
        if isinstance(record.get("metrics"), Mapping)
        and isinstance(record["metrics"].get("request_duration_seconds"), (int, float))
    ]
    usage_totals: Counter[str] = Counter()
    total_cost = 0.0
    for record in normalized_records:
        usage = record.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage_totals[key] += value
        hit = _usage_number(usage, "prompt_cache_hit_tokens", default=0)
        prompt = _usage_number(usage, "prompt_tokens", default=0)
        miss = _usage_number(usage, "prompt_cache_miss_tokens", default=max(0, prompt - hit))
        output = _usage_number(usage, "completion_tokens", default=0)
        total_cost += (
            hit * pricing.input_cache_hit_per_million
            + miss * pricing.input_cache_miss_per_million
            + output * pricing.output_per_million
        ) / 1_000_000

    raw_id_counts = Counter(
        record.get("id") for record in raw_records if isinstance(record.get("id"), str)
    )
    original_id_counts = Counter(
        source.get("original_id")
        for record in raw_records
        if isinstance(record.get("task"), Mapping)
        and isinstance((source := record["task"].get("source")), Mapping)
        and source.get("original_id") is not None
    )
    inferred_resumability = {
        "completed_ids": len(raw_id_counts),
        "duplicate_record_ids": sum(max(0, count - 1) for count in raw_id_counts.values()),
        "duplicate_original_tasks": sum(
            max(0, count - 1) for count in original_id_counts.values()
        ),
        "resume_safe": all(count == 1 for count in raw_id_counts.values()),
    }
    if resumability is not None:
        inferred_resumability.update(dict(resumability))

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "counts": {
            "selected_tasks": selected_count,
            "raw_attempts": len(raw_records),
            "api_successes": len(raw_success_ids),
            "normalized_records": len(normalized_records),
            "verified_records": len(verifier_records),
            "accepted_records": len(accepted_ids),
        },
        "rates": {
            "api_success": _rate(len(raw_success_ids), selected_count),
            "trace_reconstruction": _rate(len(trace_valid_ids), len(raw_success_ids)),
            "source_extraction": _rate(len(extraction_ids), len(raw_success_ids)),
            "syntax_success": _rate(len(compile_ids), len(raw_success_ids)),
            "import_success": _rate(len(import_ids), len(raw_success_ids)),
            "official_unit_test_pass": _rate(len(accepted_ids), selected_count),
        },
        "failure_counts": dict(sorted(failure_counts.items())),
        "distributions": {
            "response_utf8_bytes": _distribution(response_byte_lengths),
            "response_actual_tokens": _distribution(response_token_lengths),
            "prompt_tokens": _distribution(prompt_tokens),
            "api_latency_seconds": _distribution(latencies),
            "top_candidates_per_position": _distribution(top_counts),
        },
        "trace": {
            "actual_logprobs": {
                "positions": len(content_tokens),
                "available": sum(
                    isinstance(token.get("logprob"), (int, float)) for token in content_tokens
                ),
            },
            "top20": {
                "positions": len(content_tokens),
                "positions_with_20": sum(count == 20 for count in top_counts),
                "candidate_count": len(top_candidates),
            },
            "missing_actual_byte_arrays": sum(
                not _valid_byte_array(token.get("bytes")) for token in content_tokens
            ),
            "missing_or_invalid_top_candidate_byte_arrays": sum(
                not _valid_byte_array(candidate.get("bytes")) for candidate in top_candidates
            ),
        },
        "finish_reasons": dict(
            sorted(
                Counter(
                    record.get("finish_reason")
                    for record in normalized_records
                    if isinstance(record.get("finish_reason"), str)
                ).items()
            )
        ),
        "token_usage": dict(sorted(usage_totals.items())),
        "cost_rmb": {
            "currency": pricing.currency,
            "pricing_per_million_tokens": {
                "input_cache_hit": pricing.input_cache_hit_per_million,
                "input_cache_miss": pricing.input_cache_miss_per_million,
                "output": pricing.output_per_million,
            },
            "total_estimated": total_cost,
            "per_attempted_task": total_cost / selected_count if selected_count else None,
            "per_accepted_task": total_cost / len(accepted_ids) if accepted_ids else None,
        },
        "duplicates": {
            "raw_record_ids": inferred_resumability["duplicate_record_ids"],
            "original_tasks": inferred_resumability["duplicate_original_tasks"],
        },
        "resumability": inferred_resumability,
        "alm": dict(alm_diagnostics) if alm_diagnostics is not None else None,
    }


def render_audit_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    rates = report["rates"]
    cost = report["cost_rmb"]
    lines = [
        "# MBPP DeepSeek Collection Audit",
        "",
        "## Funnel",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Selected tasks | {counts['selected_tasks']} |",
        f"| API success | {_format_rate(rates['api_success'])} |",
        f"| Trace reconstruction | {_format_rate(rates['trace_reconstruction'])} |",
        f"| Source extraction | {_format_rate(rates['source_extraction'])} |",
        f"| Syntax success | {_format_rate(rates['syntax_success'])} |",
        f"| Import success | {_format_rate(rates['import_success'])} |",
        f"| Official unit-test pass | {_format_rate(rates['official_unit_test_pass'])} |",
        f"| Normalized records | {counts['normalized_records']} |",
        f"| Accepted records | {counts['accepted_records']} |",
        "",
        "## Failure categories",
        "",
    ]
    failures = report.get("failure_counts") or {}
    if failures:
        lines.extend(f"- `{category}`: {count}" for category, count in failures.items())
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Trace and usage",
            "",
            f"- Actual-token logprobs: {report['trace']['actual_logprobs']['available']}/{report['trace']['actual_logprobs']['positions']}",
            f"- Positions with 20 candidates: {report['trace']['top20']['positions_with_20']}/{report['trace']['top20']['positions']}",
            f"- Missing actual byte arrays: {report['trace']['missing_actual_byte_arrays']}",
            f"- Missing/invalid candidate byte arrays: {report['trace']['missing_or_invalid_top_candidate_byte_arrays']}",
            f"- Total top-20 candidates retained: {report['trace']['top20']['candidate_count']}",
            f"- Estimated total cost: {cost['total_estimated']:.8f} {cost['currency']}",
            f"- Estimated cost per attempted task: {_format_optional(cost['per_attempted_task'])} {cost['currency']}",
            f"- Estimated cost per accepted task: {_format_optional(cost['per_accepted_task'])} {cost['currency']}",
            "",
            "### Length and latency distributions",
            "",
            "| Distribution | Count | Min | Median | Mean | P95 | Max |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *(
                _distribution_markdown_row(label, report["distributions"][key])
                for label, key in (
                    ("Response UTF-8 bytes", "response_utf8_bytes"),
                    ("Response actual tokens", "response_actual_tokens"),
                    ("Prompt tokens", "prompt_tokens"),
                    ("API latency seconds", "api_latency_seconds"),
                    ("Top candidates per position", "top_candidates_per_position"),
                )
            ),
            "",
            "### Finish reasons",
            "",
            *(
                f"- `{reason}`: {count}"
                for reason, count in (report.get("finish_reasons") or {}).items()
            ),
            "",
            "### Token usage",
            "",
            *(
                f"- `{name}`: {value}"
                for name, value in (report.get("token_usage") or {}).items()
            ),
            "",
            "## Resumability",
            "",
            f"- Completed IDs: {report['resumability']['completed_ids']}",
            f"- Duplicate record IDs: {report['duplicates']['raw_record_ids']}",
            f"- Duplicate original tasks: {report['duplicates']['original_tasks']}",
            f"- Resume-safe: {report['resumability']['resume_safe']}",
            f"- Resume check skipped: {report['resumability'].get('skipped', 'not run')}",
        ]
    )
    provenance = report.get("dataset_provenance")
    if isinstance(provenance, Mapping):
        lines.extend(
            [
                "",
                "## Dataset provenance",
                "",
                f"- Original: {provenance.get('original')}",
                f"- Mirror: {provenance.get('mirror')}",
                f"- Revision: `{provenance.get('revision')}`",
                f"- Config/split: `{provenance.get('config')}/{provenance.get('split')}`",
                f"- License: `{provenance.get('license')}`",
            ]
        )
    alm = report.get("alm")
    if isinstance(alm, Mapping):
        lines.extend(
            [
                "",
                "## ALM preprocessing diagnostics",
                "",
                f"- Student tokenizer: `{alm.get('student_tokenizer')}`",
                f"- Student revision: `{alm.get('student_revision')}`",
                f"- Maximum sequence length: {alm.get('max_length')}",
                f"- Group counts: {alm.get('group_counts')}",
                f"- Sequence-length distribution: {alm.get('sequence_length_distribution')}",
                f"- Chunks-per-example distribution: {alm.get('chunks_per_example_distribution')}",
                f"- Prompt/completion boundary drops: {alm.get('prompt_completion_boundary_drops')}",
                f"- Examples with zero valid chunks: {len(alm.get('examples_with_zero_valid_chunks') or [])}",
                f"- Records exceeding max length: {len(alm.get('records_exceeding_max_length') or [])}",
                f"- Preprocessing errors: {len(alm.get('preprocessing_errors') or [])}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _last_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        record_id: record
        for record in records
        if isinstance((record_id := record.get("id")), str)
    }


def _passed_phase_ids(
    records: Mapping[str, Mapping[str, Any]], phase_name: str
) -> set[str]:
    return {
        record_id
        for record_id, record in records.items()
        if any(
            isinstance(phase, Mapping)
            and phase.get("name") == phase_name
            and phase.get("status") == "passed"
            for phase in (record.get("phases") or [])
        )
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _distribution(values: Sequence[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "p95": None}
    ordered = sorted(float(value) for value in values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
    }


def _usage_number(usage: Mapping[str, Any], key: str, *, default: int) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _valid_byte_array(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 255
        for item in value
    )


def _format_rate(value: Mapping[str, Any]) -> str:
    rate = value.get("rate")
    percentage = "n/a" if rate is None else f"{100 * rate:.2f}%"
    return f"{value.get('numerator')}/{value.get('denominator')} ({percentage})"


def _format_optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.8f}"


def _distribution_markdown_row(label: str, distribution: Mapping[str, Any]) -> str:
    values = [distribution.get(key) for key in ("count", "min", "median", "mean", "p95", "max")]
    rendered = ["n/a" if value is None else f"{float(value):.4f}" for value in values]
    rendered[0] = str(distribution.get("count", 0))
    return f"| {label} | " + " | ".join(rendered) + " |"
