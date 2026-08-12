#!/usr/bin/env python3
"""Audit finalized multi-attempt candidates without starting training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from deepseek_distill.audit import compute_alm_diagnostics
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
    parser.add_argument("--tokenizer-cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    return parser


def load_tokenizer(
    *,
    student_tokenizer: str,
    student_revision: str,
    tokenizer_cache_dir: Path | None,
    local_files_only: bool,
) -> Any:
    from transformers import AutoTokenizer

    cache_dir = resolve_transformers_cache_dir(tokenizer_cache_dir)
    return AutoTokenizer.from_pretrained(
        student_tokenizer,
        revision=student_revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        use_fast=True,
        local_files_only=local_files_only,
    )


def audit_finalized_candidates(
    *,
    run_dir: Path,
    student_tokenizer: str,
    student_revision: str,
    tokenizer_cache_dir: Path | None,
    local_files_only: bool,
    max_length: int,
) -> dict[str, Any]:
    """Publish the ALM, EOS/style, and clean-eligibility artifacts for a run."""

    run_dir = run_dir.resolve()
    accepted_path = run_dir / "accepted_unique.jsonl"
    if not accepted_path.is_file():
        raise FileNotFoundError(f"accepted records not found: {accepted_path}")
    accepted = _read_jsonl(accepted_path)
    tokenizer = load_tokenizer(
        student_tokenizer=student_tokenizer,
        student_revision=student_revision,
        tokenizer_cache_dir=tokenizer_cache_dir,
        local_files_only=local_files_only,
    )
    alm = compute_alm_diagnostics(
        accepted,
        tokenizer=tokenizer,
        student_tokenizer=student_tokenizer,
        student_revision=student_revision,
        max_length=max_length,
    )
    training_contract = build_training_contract_report(accepted, tokenizer)
    clean = build_multisource_clean_audit(
        records=accepted,
        alm=alm,
        training_contract=training_contract,
    )
    collection_summary = _load_collection_summary(run_dir)
    report = {
        "schema_version": "coding.audit.finalized_candidates.v1",
        "inputs": {
            "run_dir": str(run_dir),
            "accepted_unique": str(accepted_path),
            "accepted_unique_sha256": _sha256(accepted_path),
        },
        "collection_summary": collection_summary,
        "training_contract": training_contract,
        "alm": alm,
        "clean_eligibility": clean["report"],
        "training_started": False,
    }
    statuses = {
        "alm": publish_json_once(run_dir / "alm_diagnostics.clean.v1.json", alm),
        "training_contract": publish_json_once(
            run_dir / "training_contract.clean.v1.json",
            training_contract,
        ),
        "eligibility": publish_jsonl_once(
            run_dir / "clean_eligibility.v1.jsonl",
            clean["decisions"],
        ),
        "retained": publish_jsonl_once(
            run_dir / "clean_accepted.v1.jsonl",
            clean["retained_records"],
        ),
        "excluded": publish_jsonl_once(
            run_dir / "clean_excluded.v1.jsonl",
            clean["excluded_records"],
        ),
        "report_json": publish_json_once(run_dir / "clean_audit.v1.json", report),
        "report_markdown": _publish_text_once(
            run_dir / "clean_audit.v1.md",
            render_markdown(report),
        ),
    }
    report["publish_statuses"] = statuses
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    clean = report["clean_eligibility"]
    contract = report["training_contract"]
    ends = contract["end_token_supervision"]
    alm = report["alm"]
    counts = clean["counts"]
    return "\n".join(
        [
            "# Finalized candidate audit",
            "",
            f"- Official-test passes: {counts['official_test_passed']}",
            f"- Clean eligible: {counts['clean_eligible']}",
            f"- Clean excluded: {counts['clean_excluded']}",
            f"- Exclusion reasons: {clean['reason_counts']}",
            f"- EOS present: {ends['eos_present_records']}/{contract['records']}",
            f"- EOS supervised: {ends['eos_supervised_records']}/{contract['records']}",
            f"- ALM preprocessing errors: {len(alm['preprocessing_errors'])}",
            f"- Zero valid ALM chunks: {len(alm['examples_with_zero_valid_chunks'])}",
            f"- Over 4096 tokens: {len(alm['records_exceeding_max_length'])}",
            f"- Prompt/completion boundary drops: {alm['prompt_completion_boundary_drops']}",
            f"- Qwen sequence lengths: {alm['sequence_length_distribution']}",
            f"- Chunks per example: {alm['chunks_per_example_distribution']}",
            f"- Chunk groups: {alm['group_counts']}",
            "",
            "Raw responses and teacher traces were not modified. No training was started.",
            "",
        ]
    )


def resolve_transformers_cache_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.resolve()
    hub = resolved / "hub"
    return hub if hub.is_dir() else resolved


def _load_collection_summary(run_dir: Path) -> dict[str, Any] | None:
    for name in ("dataset_summary.json", "breadth_summary.json"):
        path = run_dir / name
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"collection summary must be an object: {path}")
            return value
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            records.append(value)
    return records


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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_finalized_candidates(
        run_dir=args.run_dir,
        student_tokenizer=args.student_tokenizer,
        student_revision=args.student_revision,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
    )
    clean = report["clean_eligibility"]["counts"]
    print(
        json.dumps(
            {
                "event": "finalized_candidate_audit_complete",
                "official_test_passed": clean["official_test_passed"],
                "clean_eligible": clean["clean_eligible"],
                "clean_excluded": clean["clean_excluded"],
                "training_started": False,
                "statuses": report["publish_statuses"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
