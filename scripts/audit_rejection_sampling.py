#!/usr/bin/env python3
"""Audit the 300-task blind MBPP campaign and run ALM preprocessing only."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from deepseek_distill.audit import AuditPricing, compute_alm_diagnostics
from deepseek_distill.mbpp import (
    MBPP_LICENSE,
    MBPP_MIRROR,
    MBPP_PROVENANCE,
    MBPP_REVISION,
)
from deepseek_distill.rejection_audit import (
    build_rejection_sampling_audit,
    render_rejection_sampling_markdown,
)
from deepseek_distill.rejection_sampling import publish_json_once, validate_campaign_tasks


DEFAULT_STUDENT_TOKENIZER = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_STUDENT_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
DEEPSEEK_PRICING_URL = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--resume-summary", type=Path)
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
    json_output = args.json_output or args.run_dir / "audit_report.json"
    markdown_output = args.markdown_output or args.run_dir / "audit_report.md"
    tasks = _read_jsonl(args.run_dir / "selected_tasks_300.jsonl")
    validate_campaign_tasks(tasks, expected_count=300, expected_revision=MBPP_REVISION)
    raw = _read_jsonl(args.run_dir / "raw_attempts.jsonl")
    normalized = _read_jsonl(args.run_dir / "normalized_attempts.jsonl")
    verifier = _read_jsonl(args.run_dir / "verifier_attempts.jsonl")
    accepted = _read_jsonl(args.run_dir / "accepted_unique.jsonl")
    first_target = _read_jsonl(args.run_dir / f"accepted_first_{args.target}.jsonl")

    if args.tokenizer_cache_dir is not None:
        os.environ["HF_HOME"] = str(args.tokenizer_cache_dir.resolve())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.student_tokenizer,
        revision=args.student_revision,
        cache_dir=str(args.tokenizer_cache_dir) if args.tokenizer_cache_dir else None,
        use_fast=True,
    )
    common_alm = {
        "tokenizer": tokenizer,
        "student_tokenizer": args.student_tokenizer,
        "student_revision": args.student_revision,
        "max_length": args.max_length,
    }
    alm_all = compute_alm_diagnostics(accepted, **common_alm)
    alm_first_target = compute_alm_diagnostics(first_target, **common_alm)
    resumability = (
        json.loads(args.resume_summary.read_text(encoding="utf-8"))
        if args.resume_summary is not None
        else None
    )
    report = build_rejection_sampling_audit(
        tasks=tasks,
        raw_records=raw,
        normalized_records=normalized,
        verifier_records=verifier,
        accepted_records=accepted,
        first_target_records=first_target,
        pricing=AuditPricing(
            input_cache_hit_per_million=args.input_cache_hit_price,
            input_cache_miss_per_million=args.input_cache_miss_price,
            output_per_million=args.output_price,
        ),
        alm_all=alm_all,
        alm_first_target=alm_first_target,
        resumability=resumability,
    )
    report["dataset_provenance"] = {
        "original": MBPP_PROVENANCE,
        "mirror": MBPP_MIRROR,
        "revision": MBPP_REVISION,
        "config": "full",
        "split": "train",
        "license": MBPP_LICENSE,
        "selection": "random",
        "seed": 20260721,
    }
    report["pricing_provenance"] = {
        "source": DEEPSEEK_PRICING_URL,
        "checked_on": "2026-07-21",
        "currency": "CNY",
        "unit": "per million tokens",
    }
    manifest_path = args.run_dir / "campaign_manifest.json"
    if manifest_path.exists():
        report["campaign_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    json_status = publish_json_once(json_output, report)
    markdown_status = _publish_text_once(
        markdown_output, render_rejection_sampling_markdown(report)
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "json_output": str(json_output.resolve()),
                "json_status": json_status,
                "markdown_output": str(markdown_output.resolve()),
                "markdown_status": markdown_status,
                "unique_accepted": report["counts"]["unique_accepted_tasks"],
                "first_target": report["counts"]["first_target_records"],
                "alm_all_errors": len(alm_all["preprocessing_errors"]),
                "alm_first_target_errors": len(alm_first_target["preprocessing_errors"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


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
