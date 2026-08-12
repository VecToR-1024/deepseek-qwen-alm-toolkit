#!/usr/bin/env python3
"""Audit a TACO pilot and run Qwen ALM preprocessing without training."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from deepseek_distill.audit import AuditPricing, compute_alm_diagnostics
from deepseek_distill.rejection_audit import (
    build_rejection_sampling_audit,
    render_rejection_sampling_markdown,
)
from deepseek_distill.rejection_sampling import publish_json_once, validate_campaign_tasks
from deepseek_distill.taco import (
    TACO_CARD,
    TACO_DATASET_ID,
    TACO_LICENSE,
    TACO_PILOT_SEED,
    TACO_PROVENANCE,
    TACO_REVISION,
    TACO_SPLIT,
    TACO_TASK_SCHEMA_VERSION,
    TACO_TRAIN_SHARD,
)


DEFAULT_STUDENT_TOKENIZER = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_STUDENT_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--student-tokenizer", default=DEFAULT_STUDENT_TOKENIZER)
    parser.add_argument("--student-revision", default=DEFAULT_STUDENT_REVISION)
    parser.add_argument("--tokenizer-cache-dir", type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-attempts-per-task", type=int, default=3)
    parser.add_argument(
        "--output-prefix",
        default="audit_report",
        help="Output basename inside run-dir (default: audit_report)",
    )
    parser.add_argument("--input-cache-hit-price", type=float, default=0.025)
    parser.add_argument("--input-cache-miss-price", type=float, default=3.0)
    parser.add_argument("--output-price", type=float, default=6.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tasks = _read_jsonl(
        args.run_dir / f"selected_tasks_{args.expected_tasks}.jsonl"
    )
    validate_campaign_tasks(
        tasks,
        expected_count=args.expected_tasks,
        expected_revision=TACO_REVISION,
        expected_schema_version=TACO_TASK_SCHEMA_VERSION,
        expected_dataset=TACO_DATASET_ID,
        expected_config=None,
        expected_split=TACO_SPLIT,
    )
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
    common = {
        "tokenizer": tokenizer,
        "student_tokenizer": args.student_tokenizer,
        "student_revision": args.student_revision,
        "max_length": args.max_length,
    }
    alm_all = compute_alm_diagnostics(accepted, **common)
    alm_target = compute_alm_diagnostics(first_target, **common)
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
        alm_first_target=alm_target,
        benchmark_name="TACO",
        interface_type="stdin_stdout",
        max_attempts_per_task=args.max_attempts_per_task,
    )
    manifest = args.run_dir / "campaign_manifest.json"
    campaign_manifest = (
        json.loads(manifest.read_text(encoding="utf-8"))
        if manifest.exists()
        else None
    )
    manifest_dataset = (
        campaign_manifest.get("dataset")
        if isinstance(campaign_manifest, dict)
        and isinstance(campaign_manifest.get("dataset"), dict)
        else {}
    )
    report["dataset_provenance"] = {
        "original": TACO_PROVENANCE,
        "mirror": TACO_CARD,
        "revision": TACO_REVISION,
        "config": None,
        "split": TACO_SPLIT,
        "shard": TACO_TRAIN_SHARD,
        "license": TACO_LICENSE,
        "selection": manifest_dataset.get(
            "selection",
            "random_within_single_shard_eligible_pool",
        ),
        "seed": manifest_dataset.get("seed", TACO_PILOT_SEED),
        "selection_scope": manifest_dataset.get("selection_scope"),
        "excluded_sources": manifest_dataset.get("excluded_sources", []),
    }
    if campaign_manifest is not None:
        report["campaign_manifest"] = campaign_manifest
    json_path = args.run_dir / f"{args.output_prefix}.json"
    markdown_path = args.run_dir / f"{args.output_prefix}.md"
    json_status = publish_json_once(json_path, report)
    markdown_status = _publish_text_once(
        markdown_path,
        render_rejection_sampling_markdown(report),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "json_status": json_status,
                "markdown_status": markdown_status,
                "json_output": str(json_path.resolve()),
                "markdown_output": str(markdown_path.resolve()),
                "unique_accepted": len(accepted),
                "alm_errors": len(alm_all["preprocessing_errors"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


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
