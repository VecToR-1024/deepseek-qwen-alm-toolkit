from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepseek_distill.clean_eligibility import CleanEligibilityPolicy
from deepseek_distill.clean_eligibility_audit import (
    build_clean_eligibility_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-destructive clean-training eligibility audit for "
            "offline DeepSeek ALM records."
        )
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--alm-diagnostics", type=Path, required=True)
    parser.add_argument("--eos-attestation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-comment-line-ratio", type=float, default=0.2)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_clean_eligibility_outputs(
        training_data=args.training_data,
        alm_diagnostics=args.alm_diagnostics,
        eos_attestation=args.eos_attestation,
        output_dir=args.output_dir,
        policy=CleanEligibilityPolicy(
            max_comment_line_ratio=args.max_comment_line_ratio,
            max_sequence_length=args.max_sequence_length,
        ),
    )
    print(
        json.dumps(
            {
                "event": "clean_training_audit_complete",
                "counts": report["counts"],
                "reason_counts": report["reason_counts"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
