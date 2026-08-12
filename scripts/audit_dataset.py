#!/usr/bin/env python3
"""Generate JSON and Markdown audits, including no-training ALM diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from deepseek_distill.audit import (
    AuditPricing,
    build_audit_report,
    compute_alm_diagnostics,
    render_audit_markdown,
)
from deepseek_distill.mbpp import (
    MBPP_LICENSE,
    MBPP_MIRROR,
    MBPP_PROVENANCE,
    MBPP_REVISION,
)


DEFAULT_STUDENT_TOKENIZER = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_STUDENT_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
DEEPSEEK_PRICING_URL = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--resume-summary", type=Path)
    parser.add_argument("--student-tokenizer", default=DEFAULT_STUDENT_TOKENIZER)
    parser.add_argument("--student-revision", default=DEFAULT_STUDENT_REVISION)
    parser.add_argument("--tokenizer-cache-dir", type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--input-cache-hit-price", type=float, default=0.025)
    parser.add_argument("--input-cache-miss-price", type=float, default=3.0)
    parser.add_argument("--output-price", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.json_output.exists():
        raise FileExistsError(args.json_output)
    if args.markdown_output.exists():
        raise FileExistsError(args.markdown_output)
    if args.tokenizer_cache_dir is not None:
        os.environ["HF_HOME"] = str(args.tokenizer_cache_dir.resolve())
    from transformers import AutoTokenizer

    tasks = _read_jsonl(args.tasks)
    raw = _read_jsonl(args.raw)
    normalized = _read_jsonl(args.normalized)
    verifier = _read_jsonl(args.verifier)
    accepted = _read_jsonl(args.accepted)
    resumability = (
        json.loads(args.resume_summary.read_text(encoding="utf-8"))
        if args.resume_summary is not None
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_tokenizer,
        revision=args.student_revision,
        cache_dir=str(args.tokenizer_cache_dir) if args.tokenizer_cache_dir else None,
        use_fast=True,
    )
    alm = compute_alm_diagnostics(
        accepted,
        tokenizer=tokenizer,
        student_tokenizer=args.student_tokenizer,
        student_revision=args.student_revision,
        max_length=args.max_length,
    )
    report = build_audit_report(
        tasks=tasks,
        raw_records=raw,
        normalized_records=normalized,
        verifier_records=verifier,
        accepted_records=accepted,
        pricing=AuditPricing(
            input_cache_hit_per_million=args.input_cache_hit_price,
            input_cache_miss_per_million=args.input_cache_miss_price,
            output_per_million=args.output_price,
        ),
        resumability=resumability,
        alm_diagnostics=alm,
    )
    report["generated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    report["dataset_provenance"] = {
        "original": MBPP_PROVENANCE,
        "mirror": MBPP_MIRROR,
        "revision": MBPP_REVISION,
        "config": "full",
        "split": "train",
        "license": MBPP_LICENSE,
    }
    report["pricing_provenance"] = {
        "source": DEEPSEEK_PRICING_URL,
        "checked_on": "2026-07-21",
        "currency": "CNY",
        "unit": "per million tokens",
    }
    markdown = render_audit_markdown(report)
    _write_pair_atomic(args.json_output, report, args.markdown_output, markdown)
    print(
        json.dumps(
            {
                "status": "ok",
                "json_output": str(args.json_output.resolve()),
                "markdown_output": str(args.markdown_output.resolve()),
                "selected_tasks": report["counts"]["selected_tasks"],
                "accepted_records": report["counts"]["accepted_records"],
                "alm_errors": len(alm["preprocessing_errors"]),
            },
            ensure_ascii=False,
        )
    )


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(value)
    return records


def _write_pair_atomic(json_path: Path, report: dict, markdown_path: Path, markdown: str) -> None:
    json_temp = _write_temp(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    try:
        markdown_temp = _write_temp(markdown_path, markdown)
    except BaseException:
        os.unlink(json_temp)
        raise
    try:
        os.replace(json_temp, json_path)
        os.replace(markdown_temp, markdown_path)
    except BaseException:
        for temporary in (json_temp, markdown_temp):
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def _write_temp(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


if __name__ == "__main__":
    main()
