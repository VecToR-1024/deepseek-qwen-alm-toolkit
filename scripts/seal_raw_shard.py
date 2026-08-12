#!/usr/bin/env python3
"""Seal one completed durable raw queue into a verified zstd shard."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from deepseek_distill.raw_shards import seal_raw_shard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Delete raw_attempts.jsonl only after archive round-trip verification.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    result = seal_raw_shard(
        raw_path=run_dir / "raw_attempts.jsonl",
        normalized_path=run_dir / "normalized_attempts.jsonl",
        normalization_errors_path=run_dir / "normalization_errors.jsonl",
        verifier_path=run_dir / "verifier_attempts.jsonl",
        state_path=run_dir / "pipeline_state.json",
        archive_path=run_dir / "raw_attempts.jsonl.zst",
        seal_manifest_path=run_dir / "raw_attempts.seal.json",
        archive_manifest_path=run_dir / "raw_attempts.archive.json",
        expected_records=args.expected_records,
        compression_level=args.compression_level,
        remove_source=args.remove_source,
    )
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
