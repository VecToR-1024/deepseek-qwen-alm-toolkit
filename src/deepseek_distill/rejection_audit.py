"""Attempt-aware audit calculations for blind MBPP rejection sampling."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .audit import AuditPricing
from .rejection_sampling import parse_attempt_id


REJECTION_AUDIT_SCHEMA_VERSION = "coding.audit.mbpp.rejection.v1"


def build_rejection_sampling_audit(
    *,
    tasks: Sequence[Mapping[str, Any]],
    raw_records: Sequence[Mapping[str, Any]],
    normalized_records: Sequence[Mapping[str, Any]],
    verifier_records: Sequence[Mapping[str, Any]],
    accepted_records: Sequence[Mapping[str, Any]],
    first_target_records: Sequence[Mapping[str, Any]],
    pricing: AuditPricing,
    alm_all: Mapping[str, Any] | None = None,
    alm_first_target: Mapping[str, Any] | None = None,
    resumability: Mapping[str, Any] | None = None,
    benchmark_name: str = "MBPP",
    interface_type: str = "function",
    max_attempts_per_task: int = 3,
) -> dict[str, Any]:
    """Build an audit whose rates distinguish attempts from unique tasks."""
    if (
        isinstance(max_attempts_per_task, bool)
        or not isinstance(max_attempts_per_task, int)
        or max_attempts_per_task not in {1, 2, 3}
    ):
        raise ValueError("max_attempts_per_task must be 1, 2, or 3")
    attempt_numbers = range(1, max_attempts_per_task + 1)
    selected_ids = [task.get("id") for task in tasks if isinstance(task.get("id"), str)]
    selected_set = set(selected_ids)
    raw_by_id = _last_by_id(raw_records)
    normalized_by_id = _last_by_id(normalized_records)
    verifier_by_id = _last_by_id(verifier_records)

    earliest_pass: dict[str, int] = {}
    for attempt_id, verification in verifier_by_id.items():
        try:
            problem_id, attempt_number = parse_attempt_id(attempt_id)
        except ValueError:
            continue
        if problem_id in selected_set and verification.get("failure_category") == "passed":
            earliest_pass[problem_id] = min(
                attempt_number, earliest_pass.get(problem_id, attempt_number)
            )
    cumulative = {
        attempt_number: sum(number <= attempt_number for number in earliest_pass.values())
        for attempt_number in attempt_numbers
    }
    accepted_by_attempt = Counter(earliest_pass.values())
    all_attempt_ids = {
        problem_id
        for problem_id in selected_set
        if all(
            f"{problem_id}__attempt_{number}" in raw_by_id
            for number in attempt_numbers
        )
    }
    failure_counts_per_attempt: dict[str, dict[str, int]] = {}
    total_failures: Counter[str] = Counter()
    for attempt_number in attempt_numbers:
        categories: Counter[str] = Counter()
        for attempt_id, raw in raw_by_id.items():
            try:
                _, number = parse_attempt_id(attempt_id)
            except ValueError:
                continue
            if number != attempt_number:
                continue
            category = _attempt_category(
                raw,
                normalized_by_id.get(attempt_id),
                verifier_by_id.get(attempt_id),
            )
            categories[category] += 1
            if category != "passed":
                total_failures[category] += 1
        failure_counts_per_attempt[str(attempt_number)] = dict(sorted(categories.items()))

    raw_success_ids = {
        record_id for record_id, record in raw_by_id.items() if record.get("status") == "ok"
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
    execution_ids = _passed_phase_prefix_ids(verifier_by_id, "test_")
    test_pass_ids = {
        record_id
        for record_id, record in verifier_by_id.items()
        if record.get("failure_category") == "passed"
    }

    response_bytes = [
        len(record["response_text"].encode("utf-8"))
        for record in normalized_records
        if isinstance(record.get("response_text"), str)
    ]
    response_tokens = [
        len(record.get("content_tokens") or []) for record in normalized_records
    ]
    prompt_tokens = [
        int(record["usage"]["prompt_tokens"])
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
    usage_totals: Counter[str] = Counter()
    total_cost = 0.0
    for record in normalized_records:
        usage = record.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage_totals[key] += value
        hit = _usage_number(usage, "prompt_cache_hit_tokens", 0)
        prompt = _usage_number(usage, "prompt_tokens", 0)
        miss = _usage_number(usage, "prompt_cache_miss_tokens", max(0, prompt - hit))
        output = _usage_number(usage, "completion_tokens", 0)
        total_cost += (
            hit * pricing.input_cache_hit_per_million
            + miss * pricing.input_cache_miss_per_million
            + output * pricing.output_per_million
        ) / 1_000_000

    raw_id_counts = Counter(
        record.get("id") for record in raw_records if isinstance(record.get("id"), str)
    )
    normalized_id_counts = Counter(
        record.get("id")
        for record in normalized_records
        if isinstance(record.get("id"), str)
    )
    verifier_id_counts = Counter(
        record.get("id")
        for record in verifier_records
        if isinstance(record.get("id"), str)
    )
    accepted_problem_counts = Counter(
        sampling.get("problem_id")
        for record in accepted_records
        if isinstance((sampling := record.get("sampling")), Mapping)
        and isinstance(sampling.get("problem_id"), str)
    )
    duplicate_metrics = {
        "selected_problem_ids": _duplicate_count(Counter(selected_ids)),
        "raw_attempt_ids": _duplicate_count(raw_id_counts),
        "normalized_attempt_ids": _duplicate_count(normalized_id_counts),
        "verifier_attempt_ids": _duplicate_count(verifier_id_counts),
        "accepted_problem_ids": _duplicate_count(accepted_problem_counts),
    }
    inferred_resumability = {
        "completed_attempt_ids": len(raw_by_id),
        "expected_attempt_slots": len(selected_set) * max_attempts_per_task,
        "resume_safe": all(value == 0 for value in duplicate_metrics.values()),
    }
    if resumability is not None:
        inferred_resumability.update(dict(resumability))

    accepted_count = len(earliest_pass)
    actual_attempt_count = len(raw_records)
    rates = {
        "api_success": _rate(len(raw_success_ids), actual_attempt_count),
        "trace_reconstruction": _rate(len(trace_valid_ids), len(raw_success_ids)),
        "source_extraction": _rate(len(extraction_ids), len(raw_success_ids)),
        "syntax_success": _rate(len(compile_ids), len(raw_success_ids)),
        "test_pass_per_attempt": _rate(len(test_pass_ids), actual_attempt_count),
        "test_pass_per_verified_attempt": _rate(
            len(test_pass_ids), len(verifier_by_id)
        ),
        "unique_task_pass": _rate(accepted_count, len(tasks)),
    }
    if interface_type == "stdin_stdout":
        rates["program_execution_started"] = _rate(
            len(execution_ids), len(raw_success_ids)
        )
    else:
        rates["import_success"] = _rate(len(import_ids), len(raw_success_ids))
    sampling = {
        "max_attempts_per_task": max_attempts_per_task,
        "pass_at_1": _rate(cumulative[1], len(tasks)),
        "accepted_by_attempt": {
            str(number): accepted_by_attempt[number] for number in attempt_numbers
        },
        "tasks_failing_all_attempts": len(all_attempt_ids - set(earliest_pass)),
        "average_attempts_per_accepted_task": (
            statistics.fmean(earliest_pass.values()) if earliest_pass else None
        ),
        "actual_api_attempts_per_unique_accepted_task": (
            actual_attempt_count / accepted_count if accepted_count else None
        ),
    }
    for attempt_number in range(2, max_attempts_per_task + 1):
        sampling[f"cumulative_pass_at_{attempt_number}"] = _rate(
            cumulative[attempt_number],
            len(tasks),
        )
    if max_attempts_per_task == 3:
        sampling["tasks_failing_all_3"] = sampling["tasks_failing_all_attempts"]
    return {
        "schema_version": (
            REJECTION_AUDIT_SCHEMA_VERSION
            if benchmark_name == "MBPP"
            else f"coding.audit.{benchmark_name.lower()}.rejection.v1"
        ),
        "benchmark": benchmark_name,
        "interface_type": interface_type,
        "counts": {
            "selected_tasks": len(tasks),
            "raw_attempts": actual_attempt_count,
            "api_successes": len(raw_success_ids),
            "normalized_records": len(normalized_records),
            "verified_records": len(verifier_records),
            "unique_accepted_tasks": accepted_count,
            "first_target_records": len(first_target_records),
        },
        "sampling": sampling,
        "rates": rates,
        "failure_counts": dict(sorted(total_failures.items())),
        "failure_counts_per_attempt": failure_counts_per_attempt,
        "distributions": {
            "response_utf8_bytes": _distribution(response_bytes),
            "response_actual_tokens": _distribution(response_tokens),
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
            "per_api_attempt": total_cost / actual_attempt_count if actual_attempt_count else None,
            "per_unique_accepted_task": total_cost / accepted_count if accepted_count else None,
        },
        "duplicates": duplicate_metrics,
        "resumability": inferred_resumability,
        "alm": {
            "all_unique_accepted": _decorate_alm(alm_all, len(accepted_records)),
            "first_target": _decorate_alm(alm_first_target, len(first_target_records)),
        },
    }


def render_rejection_sampling_markdown(report: Mapping[str, Any]) -> str:
    sampling = report["sampling"]
    rates = report["rates"]
    counts = report["counts"]
    max_attempts = sampling.get("max_attempts_per_task", 3)
    outcome_rows = [
        f"| Selected tasks | {counts['selected_tasks']} |",
        f"| Pass@1 | {_format_rate(sampling['pass_at_1'])} |",
    ]
    for attempt_number in range(2, max_attempts + 1):
        outcome_rows.append(
            f"| Cumulative pass@{attempt_number} | "
            f"{_format_rate(sampling[f'cumulative_pass_at_{attempt_number}'])} |"
        )
    outcome_rows.extend(
        [
            f"| Unique accepted | {counts['unique_accepted_tasks']} |",
            f"| Accepted by attempt | {sampling['accepted_by_attempt']} |",
            (
                f"| Tasks failing all {max_attempts} | "
                f"{sampling['tasks_failing_all_attempts']} |"
            ),
            (
                f"| Mean earliest passing attempt | "
                f"{sampling['average_attempts_per_accepted_task']} |"
            ),
        ]
    )
    lines = [
        f"# {report.get('benchmark', 'MBPP')} Blind Rejection-Sampling Audit",
        "",
        "## Unique-task outcomes",
        "",
        "| Metric | Result |",
        "|---|---:|",
        *outcome_rows,
        "",
        "## Attempt funnel",
        "",
        f"- API success: {_format_rate(rates['api_success'])}",
        f"- Trace reconstruction: {_format_rate(rates['trace_reconstruction'])}",
        f"- Source extraction: {_format_rate(rates['source_extraction'])}",
        f"- Syntax success: {_format_rate(rates['syntax_success'])}",
        (
            f"- Program execution started: "
            f"{_format_rate(rates['program_execution_started'])}"
            if "program_execution_started" in rates
            else f"- Import success: {_format_rate(rates['import_success'])}"
        ),
        f"- Test pass per attempt: {_format_rate(rates['test_pass_per_attempt'])}",
        f"- Test pass per verified attempt: {_format_rate(rates['test_pass_per_verified_attempt'])}",
        "",
        "## Failure categories by attempt",
        "",
    ]
    for attempt, categories in report["failure_counts_per_attempt"].items():
        lines.append(f"- Attempt {attempt}: {categories}")
    lines.extend(
        [
            "",
            "## Cost and resumability",
            "",
            f"- Total estimated API cost: {report['cost_rmb']['total_estimated']:.8f} {report['cost_rmb']['currency']}",
            f"- Cost per unique accepted task: {_format_optional(report['cost_rmb']['per_unique_accepted_task'])}",
            f"- Duplicate raw attempt IDs: {report['duplicates']['raw_attempt_ids']}",
            f"- Resume-safe: {report['resumability']['resume_safe']}",
            "",
            "## Trace, length, and usage",
            "",
            f"- Actual logprobs: {report['trace']['actual_logprobs']}",
            f"- Top-20 availability: {report['trace']['top20']}",
            f"- Missing actual byte arrays: {report['trace']['missing_actual_byte_arrays']}",
            f"- Invalid top-candidate byte arrays: {report['trace']['missing_or_invalid_top_candidate_byte_arrays']}",
            f"- Response UTF-8 bytes: {report['distributions']['response_utf8_bytes']}",
            f"- Response actual tokens: {report['distributions']['response_actual_tokens']}",
            f"- Prompt tokens: {report['distributions']['prompt_tokens']}",
            f"- API latency seconds: {report['distributions']['api_latency_seconds']}",
            f"- Token usage: {report['token_usage']}",
            f"- Finish reasons: {report['finish_reasons']}",
            "",
            "## ALM preprocessing",
            "",
        ]
    )
    for label, key in (
        ("All unique accepted", "all_unique_accepted"),
        ("First target", "first_target"),
    ):
        alm = report["alm"][key]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Success: {_format_rate(alm['preprocessing_success'])}",
                f"- Sequence lengths: {alm.get('sequence_length_distribution')}",
                f"- ALM chunks/example: {alm.get('chunks_per_example_distribution')}",
                f"- Group counts: {alm.get('group_counts')}",
                f"- Boundary drops: {alm.get('prompt_completion_boundary_drops')}",
                f"- Zero-chunk examples: {len(alm.get('examples_with_zero_valid_chunks') or [])}",
                f"- Over 4096: {len(alm.get('records_exceeding_max_length') or [])}",
                "",
            ]
        )
    provenance = report.get("dataset_provenance")
    if isinstance(provenance, Mapping):
        lines.extend(
            [
                "## Dataset provenance",
                "",
                f"- Original: {provenance.get('original')}",
                f"- Mirror: {provenance.get('mirror')}",
                f"- Revision: `{provenance.get('revision')}`",
                f"- Config/split: `{provenance.get('config')}/{provenance.get('split')}`",
                f"- License: `{provenance.get('license')}`",
                f"- Selection/seed: `{provenance.get('selection')}/{provenance.get('seed')}`",
                "",
            ]
        )
    return "\n".join(lines)


def _last_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        record_id: record
        for record in records
        if isinstance((record_id := record.get("id")), str)
    }


def _attempt_category(
    raw: Mapping[str, Any],
    normalized: Mapping[str, Any] | None,
    verifier: Mapping[str, Any] | None,
) -> str:
    if raw.get("status") == "error":
        return "api_error"
    if raw.get("status") != "ok" or normalized is None:
        return "malformed_trace"
    if verifier is None:
        return "verification_missing"
    category = verifier.get("failure_category")
    return category if isinstance(category, str) and category else "verification_missing"


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


def _passed_phase_prefix_ids(
    records: Mapping[str, Mapping[str, Any]], phase_prefix: str
) -> set[str]:
    return {
        record_id
        for record_id, record in records.items()
        if any(
            isinstance(phase, Mapping)
            and isinstance(phase.get("name"), str)
            and phase["name"].startswith(phase_prefix)
            for phase in (record.get("phases") or [])
        )
    }


def _decorate_alm(value: Mapping[str, Any] | None, expected: int) -> dict[str, Any]:
    result = dict(value) if value is not None else {}
    examples = result.get("examples") or []
    errors = result.get("preprocessing_errors") or []
    succeeded = len(examples) if isinstance(examples, list) else 0
    result["preprocessing_success"] = _rate(succeeded, expected)
    result["preprocessing_error_count"] = len(errors) if isinstance(errors, list) else 0
    return result


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
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
    }


def _usage_number(usage: Mapping[str, Any], key: str, default: int) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _valid_byte_array(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 255
        for item in value
    )


def _duplicate_count(counts: Counter[Any]) -> int:
    return sum(max(0, count - 1) for count in counts.values())


def _format_rate(value: Mapping[str, Any]) -> str:
    rate = value.get("rate")
    percentage = "n/a" if rate is None else f"{100 * float(rate):.2f}%"
    return f"{value.get('numerator')}/{value.get('denominator')} ({percentage})"


def _format_optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.8f}"
