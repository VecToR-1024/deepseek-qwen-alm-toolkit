#!/usr/bin/env python3
"""Run hard EOS/ALM/length/leakage gates on a frozen training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from deepseek_distill.training_contract_audit import build_training_contract_report
from deepseek_distill.training_data_audit import (
    extract_mbpp_problem,
    normalize_problem_text,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_chat_template_kwargs(value: str) -> dict[str, Any]:
    """Parse tokenizer-specific chat-template arguments from a CLI value."""

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("--chat-template-kwargs must be valid JSON") from error
    if not isinstance(parsed, Mapping) or any(
        not isinstance(key, str) for key in parsed
    ):
        raise ValueError("--chat-template-kwargs must be a JSON object")
    return dict(parsed)


def benchmark_problem_text(row: Mapping[str, Any]) -> str:
    """Read the natural-language problem from supported benchmark schemas."""

    question = row.get("question_content")
    if isinstance(question, str) and question.strip():
        return question.strip()
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return extract_mbpp_problem(prompt)
    raise ValueError("benchmark row has neither question_content nor prompt")


def _benchmark_id(row: Mapping[str, Any], position: int) -> str:
    for key in ("task_id", "question_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(position)


def build_overlap_report(
    training_records: Iterable[Mapping[str, Any]],
    benchmark_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Report exact normalized problem-text overlap with held-out benchmarks."""

    training: dict[str, list[str]] = {}
    training_count = 0
    for position, row in enumerate(training_records, start=1):
        training_count += 1
        task = row.get("task")
        if not isinstance(task, Mapping):
            raise ValueError(f"training row {position} has no task object")
        problem = task.get("problem_text")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"training row {position} has no problem_text")
        normalized = normalize_problem_text(problem)
        training.setdefault(normalized, []).append(str(row.get("id", position)))

    reports: dict[str, Any] = {}
    for label, path in benchmark_paths.items():
        matches: list[dict[str, str]] = []
        benchmark_count = 0
        for position, row in enumerate(iter_jsonl(path), start=1):
            benchmark_count += 1
            normalized = normalize_problem_text(benchmark_problem_text(row))
            for training_id in training.get(normalized, []):
                matches.append(
                    {
                        "training_id": training_id,
                        "benchmark_id": _benchmark_id(row, position),
                        "normalized_problem_text": normalized,
                    }
                )
        reports[label] = {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "training_records": training_count,
            "benchmark_records": benchmark_count,
            "exact_normalized_match_count": len(matches),
            "matches": matches,
        }
    return reports


def evaluate_preflight_checks(
    training_contract: Mapping[str, Any],
    overlap: Mapping[str, Mapping[str, Any]],
    *,
    expected_records: int,
    max_length: int,
) -> dict[str, bool]:
    ends = training_contract["end_token_supervision"]
    alm = training_contract["alm_preprocessing"]
    teacher = training_contract["teacher_response"]
    sequence_max = teacher["distributions"]["qwen_sequence_length"]["max"]
    return {
        "record_count_matches": training_contract["records"] == expected_records,
        "all_eos_labels_supervised": (
            ends["eos_supervised_records"] == expected_records
        ),
        "chat_template_boundaries_valid": not ends[
            "template_boundary_failure_record_ids"
        ],
        "no_alm_boundary_drops": alm["boundary_drops"] == 0,
        "no_zero_chunk_records": alm["zero_chunk_records"] == 0,
        "no_markdown_fences": teacher["records_with_code_fences"] == 0,
        "all_sequences_within_limit": (
            sequence_max is not None and sequence_max <= max_length
        ),
        "no_exact_benchmark_problem_overlap": all(
            report["exact_normalized_match_count"] == 0
            for report in overlap.values()
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    contract = report["training_contract"]
    ends = contract["end_token_supervision"]
    alm = contract["alm_preprocessing"]
    teacher = contract["teacher_response"]
    lengths = teacher["distributions"]["qwen_sequence_length"]
    lines = [
        "# Frozen training dataset preflight",
        "",
        f"- Status: `{'passed' if report['passed'] else 'failed'}`",
        f"- Records: `{contract['records']}`",
        f"- EOS labels supervised: `{ends['eos_supervised_records']}`",
        f"- Template-boundary failures: "
        f"`{len(ends['template_boundary_failure_record_ids'])}`",
        f"- ALM chunks: `{alm['total_chunks']}`",
        f"- Prompt/completion boundary drops: `{alm['boundary_drops']}`",
        f"- Zero-chunk records: `{alm['zero_chunk_records']}`",
        f"- Code fences: `{teacher['records_with_code_fences']}`",
        f"- Qwen sequence lengths: `{json.dumps(lengths, ensure_ascii=False)}`",
        "",
        "## Exact normalized problem overlap",
        "",
    ]
    for label, overlap in report["benchmark_overlap"].items():
        lines.append(
            f"- {label}: `{overlap['exact_normalized_match_count']}` matches "
            f"across `{overlap['benchmark_records']}` benchmark records"
        )
    lines.extend(["", "## Hard gates", ""])
    for name, passed in report["checks"].items():
        lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    lines.append("")
    return "\n".join(lines)


def _parse_benchmark(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--benchmark must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path:
        raise ValueError("--benchmark must be LABEL=PATH")
    return label.strip(), Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--chat-template-kwargs",
        default="{}",
        help='JSON object passed to apply_chat_template, e.g. '
        "'{\"enable_thinking\": false}'",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("refusing to overwrite an existing preflight audit")
    benchmarks = dict(_parse_benchmark(value) for value in args.benchmark)
    if len(benchmarks) != len(args.benchmark):
        raise ValueError("--benchmark labels must be unique")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    contract = build_training_contract_report(
        iter_jsonl(args.training_data),
        tokenizer,
        chat_template_kwargs=parse_chat_template_kwargs(
            args.chat_template_kwargs
        ),
    )
    overlap = build_overlap_report(iter_jsonl(args.training_data), benchmarks)
    checks = evaluate_preflight_checks(
        contract,
        overlap,
        expected_records=args.expected_records,
        max_length=args.max_length,
    )
    report = {
        "schema_version": "offline_alm.frozen_training_preflight.v1",
        "passed": all(checks.values()),
        "inputs": {
            "training_data": args.training_data.as_posix(),
            "training_data_bytes": args.training_data.stat().st_size,
            "training_data_sha256": sha256_file(args.training_data),
            "tokenizer": args.tokenizer,
            "expected_records": args.expected_records,
            "max_length": args.max_length,
        },
        "checks": checks,
        "training_contract": contract,
        "benchmark_overlap": overlap,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "frozen_training_preflight_complete",
                "passed": report["passed"],
                "records": contract["records"],
                "checks": checks,
                "output": args.output_json.as_posix(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
