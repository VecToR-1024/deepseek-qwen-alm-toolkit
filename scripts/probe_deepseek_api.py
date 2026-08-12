"""Send one non-thinking DeepSeek request and persist the full raw response."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from deepseek_distill.api import DEFAULT_BASE_URL, DeepSeekClient, GenerationConfig, build_success_record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system")
    parser.add_argument("--id", default="probe_0001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--trace-profile",
        choices=("top20", "actual_only"),
        default="top20",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"output already exists: {args.output}; pass --force to replace it")
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    config = GenerationConfig(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_logprobs=None if args.trace_profile == "actual_only" else 20,
        max_tokens=args.max_tokens,
    )
    client = DeepSeekClient(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    response = client.create_completion(messages, config)
    record = build_success_record(
        record_id=args.id,
        messages=messages,
        config=config,
        response=response,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved raw response: {args.output}")


if __name__ == "__main__":
    main()
