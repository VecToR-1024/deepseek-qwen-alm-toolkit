#!/usr/bin/env python3
"""Audit TACO length-retry v2 and run ALM preprocessing without training."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from deepseek_distill.audit import compute_alm_diagnostics
from deepseek_distill.rejection_sampling import publish_json_once


DEFAULT_STUDENT_TOKENIZER = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_STUDENT_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--student-tokenizer", default=DEFAULT_STUDENT_TOKENIZER)
    parser.add_argument("--student-revision", default=DEFAULT_STUDENT_REVISION)
    parser.add_argument("--tokenizer-cache-dir", type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--input-cache-hit-price", type=float, default=0.025)
    parser.add_argument("--input-cache-miss-price", type=float, default=3.0)
    parser.add_argument("--output-price", type=float, default=6.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw = _read_jsonl(args.run_dir / "raw_retries.jsonl")
    normalized = _read_jsonl(args.run_dir / "normalized_retries.jsonl")
    verifier = _read_jsonl(args.run_dir / "verifier_retries.jsonl")
    new_accepted = _read_jsonl(args.run_dir / "newly_accepted_unique.jsonl")
    combined = _read_jsonl(args.run_dir / "combined_accepted_unique.jsonl")
    manifest = json.loads(
        (args.run_dir / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    retry_summary = json.loads(
        (args.run_dir / "retry_summary.json").read_text(encoding="utf-8")
    )
    if args.tokenizer_cache_dir is not None:
        os.environ["HF_HOME"] = str(args.tokenizer_cache_dir.resolve())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.student_tokenizer,
        revision=args.student_revision,
        cache_dir=str(args.tokenizer_cache_dir) if args.tokenizer_cache_dir else None,
        use_fast=True,
    )
    alm_args = {
        "tokenizer": tokenizer,
        "student_tokenizer": args.student_tokenizer,
        "student_revision": args.student_revision,
        "max_length": args.max_length,
    }
    alm_new = compute_alm_diagnostics(new_accepted, **alm_args)
    alm_combined = compute_alm_diagnostics(combined, **alm_args)

    content_tokens = [
        token
        for record in normalized
        for token in (record.get("content_tokens") or [])
        if isinstance(token, dict)
    ]
    top_counts = [
        len(token.get("top_logprobs") or [])
        if isinstance(token.get("top_logprobs"), list)
        else 0
        for token in content_tokens
    ]
    usage = Counter()
    total_cost = 0.0
    for record in normalized:
        record_usage = record.get("usage")
        if not isinstance(record_usage, dict):
            continue
        for key, value in record_usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] += value
        hit = _usage(record_usage, "prompt_cache_hit_tokens")
        prompt = _usage(record_usage, "prompt_tokens")
        miss = _usage(record_usage, "prompt_cache_miss_tokens", max(0, prompt - hit))
        output = _usage(record_usage, "completion_tokens")
        total_cost += (
            hit * args.input_cache_hit_price
            + miss * args.input_cache_miss_price
            + output * args.output_price
        ) / 1_000_000
    failure_counts = Counter(
        record.get("failure_category")
        for record in verifier
        if record.get("failure_category") != "passed"
    )
    retry_problem_count = len(
        {
            record["task"]["problem_id"]
            for record in normalized
            if isinstance(record.get("task"), dict)
            and isinstance(record["task"].get("problem_id"), str)
        }
    )
    report = {
        "schema_version": "coding.audit.taco.length_retry.v2",
        "campaign_manifest": manifest,
        "retry_summary": retry_summary,
        "counts": {
            "raw_retries": len(raw),
            "normalized_retries": len(normalized),
            "verified_retries": len(verifier),
            "passing_retry_attempts": sum(
                record.get("failure_category") == "passed" for record in verifier
            ),
            "newly_accepted_tasks": len(new_accepted),
            "combined_accepted_tasks": len(combined),
        },
        "rates": {
            "api_success": _rate(
                sum(record.get("status") == "ok" for record in raw), len(raw)
            ),
            "trace_reconstruction": _rate(len(normalized), len(raw)),
            "retry_attempt_pass": _rate(
                sum(record.get("failure_category") == "passed" for record in verifier),
                len(raw),
            ),
            "new_unique_task_yield": _rate(len(new_accepted), retry_problem_count),
            "stop_after_8192": _rate(
                sum(record.get("finish_reason") == "stop" for record in normalized),
                len(normalized),
            ),
        },
        "failure_counts": dict(sorted(failure_counts.items())),
        "finish_reasons": dict(
            sorted(
                Counter(
                    record.get("finish_reason")
                    for record in normalized
                    if isinstance(record.get("finish_reason"), str)
                ).items()
            )
        ),
        "distributions": {
            "response_actual_tokens": _distribution(
                [len(record.get("content_tokens") or []) for record in normalized]
            ),
            "response_utf8_bytes": _distribution(
                [
                    len(record["response_text"].encode("utf-8"))
                    for record in normalized
                    if isinstance(record.get("response_text"), str)
                ]
            ),
        },
        "trace": {
            "actual_logprob_positions": len(content_tokens),
            "actual_logprobs_available": sum(
                isinstance(token.get("logprob"), (int, float))
                for token in content_tokens
            ),
            "positions_with_top20": sum(count == 20 for count in top_counts),
            "top_candidate_count": sum(top_counts),
        },
        "token_usage": dict(sorted(usage.items())),
        "cost_rmb": {
            "total_estimated": total_cost,
            "per_retry_attempt": total_cost / len(raw) if raw else None,
            "per_new_unique_task": (
                total_cost / len(new_accepted) if new_accepted else None
            ),
        },
        "alm": {
            "newly_accepted": alm_new,
            "combined_accepted": alm_combined,
        },
    }
    json_path = args.run_dir / "audit_report.json"
    markdown_path = args.run_dir / "audit_report.md"
    json_status = publish_json_once(json_path, report)
    markdown_status = _publish_text_once(markdown_path, _render_markdown(report))
    print(
        json.dumps(
            {
                "status": "ok",
                "json_status": json_status,
                "markdown_status": markdown_status,
                "newly_accepted": len(new_accepted),
                "combined_accepted": len(combined),
                "alm_new_errors": len(alm_new["preprocessing_errors"]),
                "alm_combined_errors": len(alm_combined["preprocessing_errors"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    rates = report["rates"]
    cost = report["cost_rmb"]
    alm_new = report["alm"]["newly_accepted"]
    alm_combined = report["alm"]["combined_accepted"]
    return "\n".join(
        [
            "# TACO length-retry v2 audit",
            "",
            f"- Retry attempts: {counts['raw_retries']}",
            f"- Passing attempts: {counts['passing_retry_attempts']}",
            f"- Newly accepted tasks: {counts['newly_accepted_tasks']}",
            f"- Combined accepted tasks: {counts['combined_accepted_tasks']}",
            f"- Stop after 8192: {rates['stop_after_8192']}",
            f"- Failure counts: {report['failure_counts']}",
            f"- Estimated cost: {cost['total_estimated']:.6f} CNY",
            "",
            "## ALM preprocessing",
            "",
            f"- New errors: {len(alm_new['preprocessing_errors'])}",
            f"- New sequence lengths: {alm_new['sequence_length_distribution']}",
            f"- Combined errors: {len(alm_combined['preprocessing_errors'])}",
            f"- Combined sequence lengths: {alm_combined['sequence_length_distribution']}",
            f"- Combined over 4096: {len(alm_combined['records_exceeding_max_length'])}",
            "",
        ]
    )


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
    }


def _usage(usage: dict[str, Any], key: str, default: int = 0) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _publish_text_once(path: Path, content: str) -> str:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"{path} already exists with different content")
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return "created"


if __name__ == "__main__":
    raise SystemExit(main())
