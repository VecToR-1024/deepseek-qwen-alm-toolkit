#!/usr/bin/env python3
"""Build passed-only accepted and durable rejected coding datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from deepseek_distill.coding_dataset import build_candidate_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Raw collection JSONL")
    parser.add_argument("--normalized", type=Path, required=True, help="Normalized JSONL")
    parser.add_argument("--verifier", type=Path, required=True, help="Verifier result JSONL")
    parser.add_argument("--accepted-output", type=Path, required=True)
    parser.add_argument("--rejected-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_candidate_datasets(
        raw_path=args.raw,
        normalized_path=args.normalized,
        verifier_path=args.verifier,
        accepted_path=args.accepted_output,
        rejected_path=args.rejected_output,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
