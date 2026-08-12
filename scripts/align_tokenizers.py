"""Align normalized DeepSeek top-k rows to strict student token positions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deepseek_distill.alignment_pipeline import align_jsonl
from deepseek_distill.cross_tokenizer_aligner import (
    CrossTokenizerAligner,
    HuggingFaceByteOffsetTokenizer,
)

DEFAULT_STUDENT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="normalized teacher JSONL")
    parser.add_argument("--output", type=Path, required=True, help="aligned output JSONL")
    parser.add_argument("--student-model", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument(
        "--tokenizer",
        help="Hugging Face model ID or local tokenizer path; defaults to --student-model",
    )
    parser.add_argument("--revision", help="optional tokenizer repository revision")
    parser.add_argument("--cache-dir", type=Path, help="optional Hugging Face cache directory")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="do not access the network when loading tokenizer files",
    )
    parser.add_argument(
        "--alignment-diagnostics",
        action="store_true",
        help=(
            "add byte-span coverage/mass diagnostics while keeping strict "
            "soft_positions for training"
        ),
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    return parser.parse_args(argv)


def load_tokenizer(
    name_or_path: str,
    *,
    revision: str | None,
    cache_dir: Path | None,
    local_files_only: bool,
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError('install tokenizer support with: pip install -e ".[align]"') from error

    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def main() -> int:
    args = parse_args()
    tokenizer_name = args.tokenizer or args.student_model
    tokenizer = load_tokenizer(
        tokenizer_name,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    span_aligner = None
    if args.alignment_diagnostics:
        span_aligner = CrossTokenizerAligner(HuggingFaceByteOffsetTokenizer(tokenizer))
    summary = align_jsonl(
        args.input,
        args.output,
        tokenizer=tokenizer,
        student_model=args.student_model,
        tokenizer_revision=args.revision,
        span_aligner=span_aligner,
        force=args.force,
    )
    summary_payload = asdict(summary)
    if summary.diagnostics is None:
        summary_payload.pop("diagnostics")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
