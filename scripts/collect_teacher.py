"""Collect DeepSeek teacher responses from an id/messages JSONL dataset."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from deepseek_distill.api import DEFAULT_BASE_URL, DeepSeekClient, GenerationConfig
from deepseek_distill.collector import collect_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--tasks", dest="input", type=Path, required=True)
    parser.add_argument("--output", "--raw-output", dest="output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--requests-per-minute", type=float, default=60)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional new JSON file for the machine-readable collection summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = DeepSeekClient(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    summary = collect_records(
        input_path=args.input,
        output_path=args.output,
        client=client,
        config=GenerationConfig(
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            top_logprobs=args.top_logprobs,
            max_tokens=args.max_tokens,
        ),
        max_workers=args.workers,
        requests_per_minute=args.requests_per_minute,
        provider={"name": "DeepSeek", "base_url": args.base_url},
    )
    summary_record = asdict(summary)
    if args.summary_output is not None:
        if args.summary_output.exists():
            raise FileExistsError(args.summary_output)
        _write_json_atomic(args.summary_output, summary_record)
    print(json.dumps(summary_record, ensure_ascii=False))


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    main()
