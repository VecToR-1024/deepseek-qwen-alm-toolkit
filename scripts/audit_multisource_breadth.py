#!/usr/bin/env python3
"""Audit one verified multi-source breadth run and publish clean candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from deepseek_distill.audit import AuditPricing, compute_alm_diagnostics
from deepseek_distill.breadth_audit import build_single_attempt_breadth_audit
from deepseek_distill.multisource_clean_audit import build_multisource_clean_audit
from deepseek_distill.rejection_sampling import publish_json_once, publish_jsonl_once
from deepseek_distill.training_contract_audit import build_training_contract_report


DEFAULT_STUDENT_TOKENIZER = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_STUDENT_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--student-tokenizer", default=DEFAULT_STUDENT_TOKENIZER)
    parser.add_argument("--student-revision", default=DEFAULT_STUDENT_REVISION)
    parser.add_argument(
        "--tokenizer-cache-dir",
        type=Path,
        help=(
            "Either HF_HOME or its hub subdirectory. If PATH/hub exists, "
            "the hub subdirectory is used as the Transformers cache_dir."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--input-cache-hit-price", type=float, default=0.025)
    parser.add_argument("--input-cache-miss-price", type=float, default=3.0)
    parser.add_argument("--output-price", type=float, default=6.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    cache_dir = resolve_transformers_cache_dir(args.tokenizer_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_tokenizer,
        revision=args.student_revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    accepted_path = args.run_dir / "accepted_unique.jsonl"
    accepted = _read_jsonl(accepted_path)
    alm = compute_alm_diagnostics(
        accepted,
        tokenizer=tokenizer,
        student_tokenizer=args.student_tokenizer,
        student_revision=args.student_revision,
        max_length=args.max_length,
    )
    training_contract = build_training_contract_report(accepted, tokenizer)
    clean = build_multisource_clean_audit(
        records=accepted,
        alm=alm,
        training_contract=training_contract,
    )
    funnel = build_single_attempt_breadth_audit(
        run_dir=args.run_dir,
        pricing=AuditPricing(
            input_cache_hit_per_million=args.input_cache_hit_price,
            input_cache_miss_per_million=args.input_cache_miss_price,
            output_per_million=args.output_price,
        ),
        alm=alm,
    )
    manifest = funnel["campaign_manifest"]
    dataset = manifest.get("dataset") if isinstance(manifest, Mapping) else None
    source = (
        str(dataset.get("dataset"))
        if isinstance(dataset, Mapping) and dataset.get("dataset")
        else "unknown"
    )
    report = {
        "schema_version": "coding.audit.multisource.clean_breadth.v1",
        "source": source,
        "inputs": {
            "run_dir": str(args.run_dir),
            "accepted_unique": str(accepted_path),
            "accepted_unique_sha256": _sha256(accepted_path),
        },
        "funnel": funnel,
        "training_contract": training_contract,
        "alm": alm,
        "clean_eligibility": clean["report"],
        "training_started": False,
    }
    statuses = {
        "alm": publish_json_once(args.run_dir / "alm_diagnostics.clean.v1.json", alm),
        "training_contract": publish_json_once(
            args.run_dir / "training_contract.clean.v1.json",
            training_contract,
        ),
        "eligibility": publish_jsonl_once(
            args.run_dir / "clean_eligibility.v1.jsonl",
            clean["decisions"],
        ),
        "retained": publish_jsonl_once(
            args.run_dir / "clean_accepted.v1.jsonl",
            clean["retained_records"],
        ),
        "excluded": publish_jsonl_once(
            args.run_dir / "clean_excluded.v1.jsonl",
            clean["excluded_records"],
        ),
        "report_json": publish_json_once(
            args.run_dir / "clean_audit.v1.json",
            report,
        ),
        "report_markdown": _publish_text_once(
            args.run_dir / "clean_audit.v1.md",
            render_markdown(report),
        ),
    }
    print(
        json.dumps(
            {
                "event": "multisource_clean_audit_complete",
                "source": source,
                "official_test_passed": len(accepted),
                "clean_eligible": clean["report"]["counts"]["clean_eligible"],
                "alm_errors": len(alm["preprocessing_errors"]),
                "eos_supervised": training_contract["end_token_supervision"][
                    "eos_supervised_records"
                ],
                "statuses": statuses,
            },
            ensure_ascii=False,
        )
    )
    return 0


def render_markdown(report: Mapping[str, Any]) -> str:
    funnel = report["funnel"]
    counts = funnel["counts"]
    rates = funnel["rates"]
    clean = report["clean_eligibility"]
    training = report["training_contract"]
    ends = training["end_token_supervision"]
    alm = report["alm"]
    return "\n".join(
        [
            f"# Clean breadth audit: {report['source']}",
            "",
            "## Collection outcome",
            "",
            f"- Selected tasks: {counts['selected_tasks']}",
            f"- API successes: {counts['api_successes']}/{counts['raw_attempts']}",
            f"- Official-test passes: {counts['unique_accepted_tasks']}",
            f"- Pass@1: {_percentage(rates['unique_task_pass'])}",
            f"- Failure categories: {funnel['failure_counts']}",
            f"- Finish reasons: {funnel['finish_reasons']}",
            "",
            "## Final clean-training gate",
            "",
            f"- Clean eligible: {clean['counts']['clean_eligible']}",
            f"- Clean excluded: {clean['counts']['clean_excluded']}",
            f"- Exclusion reasons: {clean['reason_counts']}",
            f"- EOS present: {ends['eos_present_records']}/{training['records']}",
            f"- EOS supervised: {ends['eos_supervised_records']}/{training['records']}",
            f"- ALM preprocessing errors: {len(alm['preprocessing_errors'])}",
            f"- Zero valid ALM chunks: {len(alm['examples_with_zero_valid_chunks'])}",
            f"- Over 4096 tokens: {len(alm['records_exceeding_max_length'])}",
            f"- Prompt/completion boundary drops: {alm['prompt_completion_boundary_drops']}",
            "",
            "## ALM shape",
            "",
            f"- Qwen sequence lengths: {alm['sequence_length_distribution']}",
            f"- Chunks per example: {alm['chunks_per_example_distribution']}",
            f"- Chunk groups: {alm['group_counts']}",
            "",
            "Raw responses and teacher traces were not modified. No training was started.",
            "",
        ]
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def resolve_transformers_cache_dir(path: Path | None) -> Path | None:
    """Accept either an HF_HOME root or its concrete hub cache directory."""

    if path is None:
        return None
    resolved = path.resolve()
    hub = resolved / "hub"
    return hub if hub.is_dir() else resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _percentage(value: Mapping[str, Any]) -> str:
    rate = value.get("rate")
    return "n/a" if rate is None else f"{100 * float(rate):.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
