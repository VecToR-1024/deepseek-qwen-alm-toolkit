#!/usr/bin/env python3
"""Verify normalized benchmark solutions in isolated child processes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from deepseek_distill.code_verifier import verify_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Normalized trace JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Append-only verifier JSONL")
    parser.add_argument(
        "--phase-timeout",
        type=float,
        default=5.0,
        help="Wall-clock timeout for each compile/import/test child phase",
    )
    parser.add_argument(
        "--max-output-characters",
        type=int,
        default=65_536,
        help="Maximum captured stdout/stderr characters per field",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = verify_jsonl(
        input_path=args.input,
        output_path=args.output,
        phase_timeout_seconds=args.phase_timeout,
        max_output_characters=args.max_output_characters,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
