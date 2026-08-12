#!/usr/bin/env python3
"""Audit the TACO breadth campaign with streaming I/O and ALM preprocessing."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from deepseek_distill.audit import AuditPricing, compute_alm_diagnostics
from deepseek_distill.breadth_audit import (
    build_single_attempt_breadth_audit,
    render_breadth_audit_markdown,
)
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
    if args.tokenizer_cache_dir is not None:
        os.environ["HF_HOME"] = str(args.tokenizer_cache_dir.resolve())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.student_tokenizer,
        revision=args.student_revision,
        cache_dir=str(args.tokenizer_cache_dir) if args.tokenizer_cache_dir else None,
        use_fast=True,
    )
    alm = compute_alm_diagnostics(
        _read_jsonl_stream(args.run_dir / "accepted_unique.jsonl"),
        tokenizer=tokenizer,
        student_tokenizer=args.student_tokenizer,
        student_revision=args.student_revision,
        max_length=args.max_length,
    )
    report = build_single_attempt_breadth_audit(
        run_dir=args.run_dir,
        pricing=AuditPricing(
            input_cache_hit_per_million=args.input_cache_hit_price,
            input_cache_miss_per_million=args.input_cache_miss_price,
            output_per_million=args.output_price,
        ),
        alm=alm,
    )
    json_path = args.run_dir / "audit_report.json"
    markdown_path = args.run_dir / "audit_report.md"
    json_status = publish_json_once(json_path, report)
    markdown_status = _publish_text_once(
        markdown_path,
        render_breadth_audit_markdown(report),
    )
    print(
        json.dumps(
            {
                "event": "breadth_audit_complete",
                "json_status": json_status,
                "markdown_status": markdown_status,
                "accepted_unique": report["counts"]["unique_accepted_tasks"],
                "alm_errors": len(alm["preprocessing_errors"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _read_jsonl_stream(path: Path):
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


def _publish_text_once(path: Path, content: str) -> str:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"{path} already exists with different content")
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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
