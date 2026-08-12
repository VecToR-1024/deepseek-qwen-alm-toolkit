"""Streaming audit calculations for one-attempt breadth campaigns."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .audit import AuditPricing


def build_single_attempt_breadth_audit(
    *,
    run_dir: Path,
    pricing: AuditPricing,
    alm: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit multi-gigabyte JSONL artifacts without materializing them."""
    run_dir = Path(run_dir)
    campaign_summary = _read_json(run_dir / "breadth_summary.json")
    campaign_manifest = _read_json(run_dir / "campaign_manifest.json")
    selected_count = int(campaign_summary["counts"]["selected_tasks"])

    raw_ids: Counter[str] = Counter()
    raw_success_ids: set[str] = set()
    latencies: list[float] = []
    for record in _read_jsonl_stream(run_dir / "raw_attempts.jsonl"):
        attempt_id = _record_id(record)
        raw_ids[attempt_id] += 1
        if record.get("status") == "ok":
            raw_success_ids.add(attempt_id)
        metrics = record.get("metrics")
        latency = (
            metrics.get("request_duration_seconds")
            if isinstance(metrics, Mapping)
            else None
        )
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            latencies.append(float(latency))

    normalized_ids: Counter[str] = Counter()
    trace_valid_ids: set[str] = set()
    response_bytes: list[int] = []
    response_tokens: list[int] = []
    prompt_tokens: list[int] = []
    usage_totals: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    actual_positions = 0
    actual_logprobs = 0
    positions_with_top20 = 0
    top_candidate_count = 0
    missing_actual_bytes = 0
    invalid_top_bytes = 0
    total_cost = 0.0
    for record in _read_jsonl_stream(run_dir / "normalized_attempts.jsonl"):
        attempt_id = _record_id(record)
        normalized_ids[attempt_id] += 1
        validation = record.get("validation")
        if (
            isinstance(validation, Mapping)
            and validation.get("content_bytes_match") is True
        ):
            trace_valid_ids.add(attempt_id)
        response_text = record.get("response_text")
        if isinstance(response_text, str):
            response_bytes.append(len(response_text.encode("utf-8")))
        content_tokens = record.get("content_tokens")
        tokens = content_tokens if isinstance(content_tokens, list) else []
        response_tokens.append(len(tokens))
        for token in tokens:
            if not isinstance(token, Mapping):
                continue
            actual_positions += 1
            if isinstance(token.get("logprob"), (int, float)):
                actual_logprobs += 1
            if not _valid_byte_array(token.get("bytes")):
                missing_actual_bytes += 1
            candidates = token.get("top_logprobs")
            candidates = candidates if isinstance(candidates, list) else []
            if len(candidates) == 20:
                positions_with_top20 += 1
            top_candidate_count += len(candidates)
            invalid_top_bytes += sum(
                not isinstance(candidate, Mapping)
                or not _valid_byte_array(candidate.get("bytes"))
                for candidate in candidates
            )
        finish_reason = record.get("finish_reason")
        if isinstance(finish_reason, str):
            finish_reasons[finish_reason] += 1
        usage = record.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage_totals[key] += value
        prompt = _usage(usage, "prompt_tokens")
        hit = _usage(usage, "prompt_cache_hit_tokens")
        miss = _usage(
            usage,
            "prompt_cache_miss_tokens",
            default=max(0, prompt - hit),
        )
        output = _usage(usage, "completion_tokens")
        prompt_tokens.append(prompt)
        total_cost += (
            hit * pricing.input_cache_hit_per_million
            + miss * pricing.input_cache_miss_per_million
            + output * pricing.output_per_million
        ) / 1_000_000

    verifier_ids: Counter[str] = Counter()
    extraction_ids: set[str] = set()
    compile_ids: set[str] = set()
    execution_ids: set[str] = set()
    pass_ids: set[str] = set()
    failure_counts: Counter[str] = Counter()
    for record in _read_jsonl_stream(run_dir / "verifier_attempts.jsonl"):
        attempt_id = _record_id(record)
        verifier_ids[attempt_id] += 1
        extraction = record.get("source_extraction")
        if isinstance(extraction, Mapping) and extraction.get("status") == "passed":
            extraction_ids.add(attempt_id)
        phases = record.get("phases")
        phases = phases if isinstance(phases, list) else []
        if any(
            isinstance(phase, Mapping)
            and phase.get("name") == "compile"
            and phase.get("status") == "passed"
            for phase in phases
        ):
            compile_ids.add(attempt_id)
        if any(
            isinstance(phase, Mapping)
            and isinstance(phase.get("name"), str)
            and phase["name"].startswith("test_")
            for phase in phases
        ):
            execution_ids.add(attempt_id)
        category = record.get("failure_category")
        if category == "passed":
            pass_ids.add(attempt_id)
        elif isinstance(category, str):
            failure_counts[category] += 1
    failure_counts["api_error"] += len(set(raw_ids) - raw_success_ids)
    if failure_counts["api_error"] == 0:
        del failure_counts["api_error"]

    duplicates = {
        "raw_attempt_ids": _duplicate_count(raw_ids),
        "normalized_attempt_ids": _duplicate_count(normalized_ids),
        "verifier_attempt_ids": _duplicate_count(verifier_ids),
    }
    raw_count = sum(raw_ids.values())
    accepted_count = len(pass_ids)
    return {
        "schema_version": "coding.audit.taco.breadth.v2",
        "benchmark": "TACO",
        "interface_type": "stdin_stdout",
        "campaign_manifest": campaign_manifest,
        "counts": {
            "selected_tasks": selected_count,
            "raw_attempts": raw_count,
            "api_successes": len(raw_success_ids),
            "normalized_records": sum(normalized_ids.values()),
            "verified_records": sum(verifier_ids.values()),
            "unique_accepted_tasks": accepted_count,
        },
        "sampling": {
            "max_attempts_per_task": 1,
            "pass_at_1": _rate(accepted_count, selected_count),
            "tasks_failing_all_attempts": selected_count - accepted_count,
            "actual_api_attempts_per_unique_accepted_task": (
                raw_count / accepted_count if accepted_count else None
            ),
        },
        "rates": {
            "api_success": _rate(len(raw_success_ids), raw_count),
            "trace_reconstruction": _rate(
                len(trace_valid_ids),
                len(raw_success_ids),
            ),
            "source_extraction": _rate(
                len(extraction_ids),
                len(raw_success_ids),
            ),
            "syntax_success": _rate(len(compile_ids), len(raw_success_ids)),
            "program_execution_started": _rate(
                len(execution_ids),
                len(raw_success_ids),
            ),
            "test_pass_per_attempt": _rate(accepted_count, raw_count),
            "unique_task_pass": _rate(accepted_count, selected_count),
        },
        "failure_counts": dict(sorted(failure_counts.items())),
        "distributions": {
            "response_utf8_bytes": _distribution(response_bytes),
            "response_actual_tokens": _distribution(response_tokens),
            "prompt_tokens": _distribution(prompt_tokens),
            "api_latency_seconds": _distribution(latencies),
        },
        "trace": {
            "actual_logprobs": {
                "positions": actual_positions,
                "available": actual_logprobs,
            },
            "top20": {
                "positions": actual_positions,
                "positions_with_20": positions_with_top20,
                "candidate_count": top_candidate_count,
            },
            "missing_actual_byte_arrays": missing_actual_bytes,
            "missing_or_invalid_top_candidate_byte_arrays": invalid_top_bytes,
        },
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "token_usage": dict(sorted(usage_totals.items())),
        "cost_rmb": {
            "currency": pricing.currency,
            "pricing_per_million_tokens": {
                "input_cache_hit": pricing.input_cache_hit_per_million,
                "input_cache_miss": pricing.input_cache_miss_per_million,
                "output": pricing.output_per_million,
            },
            "total_estimated": total_cost,
            "per_api_attempt": total_cost / raw_count if raw_count else None,
            "per_unique_accepted_task": (
                total_cost / accepted_count if accepted_count else None
            ),
        },
        "duplicates": duplicates,
        "resumability": {
            "completed_attempt_ids": len(raw_ids),
            "expected_attempt_slots": selected_count,
            "resume_safe": (
                raw_count == selected_count
                and all(value == 0 for value in duplicates.values())
            ),
        },
        "alm": {
            "all_unique_accepted": dict(alm),
            "preprocessing_success": _rate(
                len(alm.get("examples") or []),
                accepted_count,
            ),
        },
    }


def render_breadth_audit_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    rates = report["rates"]
    sampling = report["sampling"]
    alm = report["alm"]
    return "\n".join(
        [
            "# TACO breadth-first v2 audit",
            "",
            "## Outcomes",
            "",
            f"- Selected tasks: {counts['selected_tasks']}",
            f"- Raw attempts: {counts['raw_attempts']}",
            f"- Unique accepted: {counts['unique_accepted_tasks']}",
            f"- Pass@1: {_format_rate(sampling['pass_at_1'])}",
            f"- Failure counts: {report['failure_counts']}",
            "",
            "## Trace and cost",
            "",
            f"- API success: {_format_rate(rates['api_success'])}",
            f"- Trace reconstruction: {_format_rate(rates['trace_reconstruction'])}",
            f"- Source extraction: {_format_rate(rates['source_extraction'])}",
            f"- Syntax success: {_format_rate(rates['syntax_success'])}",
            f"- Program execution started: {_format_rate(rates['program_execution_started'])}",
            f"- Finish reasons: {report['finish_reasons']}",
            f"- Token usage: {report['token_usage']}",
            f"- Estimated cost: {report['cost_rmb']['total_estimated']:.8f} CNY",
            f"- Resume safe: {report['resumability']['resume_safe']}",
            "",
            "## ALM preprocessing",
            "",
            f"- Success: {_format_rate(alm['preprocessing_success'])}",
            f"- Sequence lengths: {alm['all_unique_accepted'].get('sequence_length_distribution')}",
            f"- Chunks/example: {alm['all_unique_accepted'].get('chunks_per_example_distribution')}",
            f"- Group counts: {alm['all_unique_accepted'].get('group_counts')}",
            f"- Zero chunks: {len(alm['all_unique_accepted'].get('examples_with_zero_valid_chunks') or [])}",
            f"- Over 4096: {len(alm['all_unique_accepted'].get('records_exceeding_max_length') or [])}",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _read_jsonl_stream(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def _record_id(record: Mapping[str, Any]) -> str:
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record has no valid id")
    return record_id


def _usage(usage: Mapping[str, Any], key: str, *, default: int = 0) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _valid_byte_array(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, int)
        and not isinstance(item, bool)
        and 0 <= item <= 255
        for item in value
    )


def _duplicate_count(counts: Counter[str]) -> int:
    return sum(count - 1 for count in counts.values() if count > 1)


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _distribution(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
    }


def _format_rate(value: Mapping[str, Any]) -> str:
    rate = value.get("rate")
    percentage = "n/a" if rate is None else f"{100 * rate:.2f}%"
    return f"{value.get('numerator')}/{value.get('denominator')} ({percentage})"
