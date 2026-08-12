#!/usr/bin/env python3
"""Merge verified MBPP and TACO traces into a versioned ALM candidate set."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from deepseek_distill.audit import compute_alm_diagnostics
from deepseek_distill.candidate_merge import (
    CandidateSource,
    build_merged_candidate_outputs,
    iter_candidate_records,
)


DEFAULT_STUDENT_TOKENIZER = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_STUDENT_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mbpp", type=Path, required=True)
    parser.add_argument("--taco-pilot-retry", type=Path, required=True)
    parser.add_argument("--taco-breadth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-mbpp", type=int, default=200)
    parser.add_argument("--expected-taco-pilot-retry", type=int, default=49)
    parser.add_argument("--expected-taco-breadth", type=int, default=412)
    parser.add_argument("--student-tokenizer", default=DEFAULT_STUDENT_TOKENIZER)
    parser.add_argument("--student-revision", default=DEFAULT_STUDENT_REVISION)
    parser.add_argument("--tokenizer-cache-dir", type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sources = (
        CandidateSource(
            "mbpp_rejection_v1_first200",
            args.mbpp,
            expected_records=args.expected_mbpp,
        ),
        CandidateSource(
            "taco_pilot_length_retry_v2_combined49",
            args.taco_pilot_retry,
            expected_records=args.expected_taco_pilot_retry,
        ),
        CandidateSource(
            "taco_breadth_v2_accepted412",
            args.taco_breadth,
            expected_records=args.expected_taco_breadth,
        ),
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
    alm = compute_alm_diagnostics(
        iter_candidate_records(sources),
        tokenizer=tokenizer,
        student_tokenizer=args.student_tokenizer,
        student_revision=args.student_revision,
        max_length=args.max_length,
    )
    summary = build_merged_candidate_outputs(
        sources=sources,
        output_dir=args.output_dir,
        alm_diagnostics=alm,
    )
    report_status = _publish_text_once(
        args.output_dir / "merge_report.md",
        render_report_markdown(summary),
    )
    print(
        json.dumps(
            {
                "event": "merged_candidates_complete",
                "all_candidates": summary["counts"]["all_candidates"],
                f"trainable_max{args.max_length}": summary["counts"][
                    f"trainable_max{args.max_length}"
                ],
                "excluded": summary["counts"]["excluded"],
                "report_status": report_status,
                "write_status": summary["write_status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def render_report_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    max_length = next(
        int(key.removeprefix("trainable_max"))
        for key in counts
        if key.startswith("trainable_max")
    )
    trainable_key = f"trainable_max{max_length}"
    alm = report["alm"]
    outputs = report["outputs"]
    return "\n".join(
        [
            "# Merged MBPP + TACO training candidates",
            "",
            "## Counts",
            "",
            f"- All candidates: {counts['all_candidates']}",
            f"- Trainable at max length {max_length}: {counts[trainable_key]}",
            f"- Excluded: {counts['excluded']}",
            f"- MBPP: {counts['mbpp']}",
            f"- TACO: {counts['taco']}",
            f"- Source order: {report['source_order']}",
            f"- Duplicate IDs: {report['duplicates']}",
            "",
            "## Eligibility",
            "",
            f"- Exclusion counts: {report['exclusion_counts']}",
            f"- ALM preprocessing errors: {alm['preprocessing_errors']}",
            f"- Zero valid chunks: {alm['zero_valid_chunks']}",
            f"- Over max length: {alm['records_exceeding_max_length']}",
            f"- Prompt/completion boundary drops: "
            f"{alm['prompt_completion_boundary_drops']}",
            "",
            "## ALM diagnostics",
            "",
            f"- Sequence lengths: {alm['sequence_length_distribution']}",
            f"- Chunks/example: {alm['chunks_per_example_distribution']}",
            f"- Chunk groups: {alm['group_counts']}",
            "",
            "## Output hashes",
            "",
            f"- all_candidates.jsonl: {outputs['all_candidates']['sha256']}",
            f"- trainable_max{max_length}.jsonl: "
            f"{outputs[trainable_key]['sha256']}",
            f"- excluded_records.jsonl: {outputs['excluded_records']['sha256']}",
            "",
            "No student training was started by this command.",
            "",
        ]
    )


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
