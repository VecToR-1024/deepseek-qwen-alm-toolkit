"""Normalize successful records from a raw DeepSeek teacher JSONL file."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from deepseek_distill.normalize import normalize_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="raw teacher JSONL")
    parser.add_argument("--output", type=Path, required=True, help="normalized output JSONL")
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = normalize_jsonl(args.input, args.output, force=args.force)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
