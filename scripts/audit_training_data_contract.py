from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from deepseek_distill.training_contract_audit import (
    build_training_contract_report,
    tokenizer_contract,
)
from deepseek_distill.training_data_audit import build_mbpp_overlap_report


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


def load_heldout_rows(
    data_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_ids = manifest.get("heldout_task_ids")
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError("heldout manifest must contain heldout_task_ids")
    by_id = {str(row.get("task_id")): row for row in iter_jsonl(data_path)}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"heldout data is missing {len(missing)} manifest task IDs")
    return [by_id[task_id] for task_id in task_ids], manifest


def generation_contract(path: str, *, local_files_only: bool) -> dict[str, Any]:
    from transformers import GenerationConfig

    try:
        config = GenerationConfig.from_pretrained(
            path,
            local_files_only=local_files_only,
        )
    except Exception as error:
        return {
            "available": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "available": True,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
        "bos_token_id": config.bos_token_id,
        "max_length": config.max_length,
    }


def load_tokenizer_contract(
    path: str,
    *,
    local_files_only: bool,
) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        use_fast=True,
        local_files_only=local_files_only,
    )
    contract = tokenizer_contract(tokenizer)
    contract["path"] = path
    contract["generation_config"] = generation_contract(
        path,
        local_files_only=local_files_only,
    )
    return tokenizer, contract


def source_contract(paths: Iterable[Path]) -> list[dict[str, Any]]:
    contracts = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        contracts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "uses_apply_chat_template": "apply_chat_template" in text,
                "passes_eos_token_id": "eos_token_id=" in text,
                "passes_pad_token_id": "pad_token_id=" in text,
                "contains_markdown_fence_stop": "\\n```\\n" in text
                or '"```"' in text,
            }
        )
    return contracts


def build_report(
    *,
    training_data: Path,
    tokenizer_path: str,
    checkpoint_paths: Mapping[str, str],
    heldout_data: Path,
    heldout_manifest: Path,
    benchmark_source_paths: Iterable[Path],
    local_files_only: bool,
) -> dict[str, Any]:
    tokenizer, base_contract = load_tokenizer_contract(
        tokenizer_path,
        local_files_only=local_files_only,
    )
    training = build_training_contract_report(
        iter_jsonl(training_data),
        tokenizer,
    )
    checkpoint_contracts: dict[str, Any] = {}
    for name, path in checkpoint_paths.items():
        _, checkpoint_contracts[name] = load_tokenizer_contract(
            path,
            local_files_only=local_files_only,
        )
    heldout_rows, manifest = load_heldout_rows(heldout_data, heldout_manifest)
    overlap = build_mbpp_overlap_report(
        iter_jsonl(training_data),
        heldout_rows,
    )
    return {
        "schema_version": "offline_alm.format_overlap_audit.v1",
        "inputs": {
            "training_data": str(training_data),
            "training_data_sha256": sha256_file(training_data),
            "heldout_data": str(heldout_data),
            "heldout_data_sha256": sha256_file(heldout_data),
            "heldout_manifest": str(heldout_manifest),
            "heldout_manifest_sha256": sha256_file(heldout_manifest),
            "heldout_manifest_schema": manifest.get("schema_version"),
        },
        "tokenizer_contracts": {
            "training_base": base_contract,
            **checkpoint_contracts,
        },
        "benchmark_source_contracts": source_contract(benchmark_source_paths),
        "training_contract": training,
        "mbpp_overlap": overlap,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    training = report["training_contract"]
    ends = training["end_token_supervision"]
    alm = training["alm_preprocessing"]
    teacher = training["teacher_response"]
    overlap = report["mbpp_overlap"]
    text = overlap["text"]
    tests = overlap["tests"]
    lines = [
        "# Training-format and MBPP-overlap audit",
        "",
        "## EOS and label contract",
        "",
        f"- Audited records: `{training['records']}`",
        f"- EOS present: `{ends['eos_present_records']}/{training['records']}`",
        f"- EOS supervised (label != -100): "
        f"`{ends['eos_supervised_records']}/{training['records']}`",
        f"- Rendered assistant suffixes: `{json.dumps(ends['assistant_suffixes'], ensure_ascii=False)}`",
        "",
        "## ALM preprocessing",
        "",
        f"- Total valid chunks: `{alm['total_chunks']}`",
        f"- Chunk count per record: `{json.dumps(alm['chunk_count_distribution'])}`",
        f"- Prompt/completion boundary drops: `{alm['boundary_drops']}`",
        f"- Records with zero valid chunks: `{alm['zero_chunk_records']}`",
        "",
        "## Teacher response shape",
        "",
        f"- Contains comments: `{teacher['records_with_comments']}/{training['records']}`",
        f"- Contains docstrings: `{teacher['records_with_docstrings']}/{training['records']}`",
        f"- Contains code fences: `{teacher['records_with_code_fences']}/{training['records']}`",
        f"- Ends with newline: `{teacher['records_ending_with_newline']}/{training['records']}`",
        f"- Provider-token length: `{json.dumps(teacher['distributions']['provider_token_count'])}`",
        f"- Qwen supervised-token length: "
        f"`{json.dumps(teacher['distributions']['qwen_supervised_token_count'])}`",
        "",
        "## MBPP train versus held-out",
        "",
        f"- Train/held-out tasks: `{overlap['train_tasks']}/{overlap['heldout_tasks']}`",
        f"- Exact numeric task-ID overlap: `{len(overlap['exact_numeric_id_overlap'])}`",
        f"- Exact normalized problem-text matches: "
        f"`{text['exact_normalized_match_count']}`",
        f"- Held-out nearest-text similarity >= 0.9: "
        f"`{text['heldout_with_similarity_at_least']['0.9']}`",
        f"- Held-out nearest TF-IDF cosine >= 0.9: "
        f"`{text['heldout_with_tfidf_at_least']['0.9']}`",
        f"- Held-out nearest character-5-gram Jaccard >= 0.9: "
        f"`{text['heldout_with_char_5gram_at_least']['0.9']}`",
        f"- Exact normalized test matches: "
        f"`{tests['exact_normalized_test_match_count']}`",
        f"- Held-out tasks sharing at least one exact normalized assertion: "
        f"`{tests['heldout_with_any_exact_assertion_match']}`",
        f"- Held-out tasks sharing at least one exact assertion with the same "
        f"function name: "
        f"`{tests['heldout_with_any_exact_named_assertion_match']}`",
        f"- Exact literal-normalized assertion-structure matches: "
        f"`{tests['exact_structure_match_heldout_count']}`",
        "",
        "The structure metric intentionally removes identifiers and literal values. "
        "It measures benchmark-family assertion style, not semantic task identity.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit training EOS labels, response style, and MBPP overlap."
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--checkpoint-tokenizer",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--heldout-data", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument(
        "--benchmark-source",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.force and (args.output_json.exists() or args.output_md.exists()):
        raise FileExistsError("refusing to overwrite an existing audit")
    checkpoints = {}
    for value in args.checkpoint_tokenizer:
        if "=" not in value:
            raise ValueError("--checkpoint-tokenizer must be NAME=PATH")
        name, path = value.split("=", 1)
        checkpoints[name] = path
    report = build_report(
        training_data=args.training_data,
        tokenizer_path=args.tokenizer,
        checkpoint_paths=checkpoints,
        heldout_data=args.heldout_data,
        heldout_manifest=args.heldout_manifest,
        benchmark_source_paths=args.benchmark_source,
        local_files_only=args.local_files_only,
    )
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
                "records": report["training_contract"]["records"],
                "eos_supervised": report["training_contract"][
                    "end_token_supervision"
                ]["eos_supervised_records"],
                "heldout_tasks": report["mbpp_overlap"]["heldout_tasks"],
                "output": str(args.output_json),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
