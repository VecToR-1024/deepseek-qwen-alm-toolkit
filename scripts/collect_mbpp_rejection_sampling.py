#!/usr/bin/env python3
"""Run the pinned 300-task, three-attempt blind MBPP collection campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from deepseek_distill.api import DEFAULT_BASE_URL, DeepSeekClient, GenerationConfig
from deepseek_distill.mbpp import MBPP_REVISION
from deepseek_distill.rejection_sampling import (
    publish_jsonl_once,
    run_rejection_sampling,
    validate_campaign_tasks,
)


EXPECTED_TASKS = 300
SELECTION_SEED = 20260721


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="Pinned selected task JSONL")
    parser.add_argument("--run-dir", type=Path, required=True, help="Versioned output directory")
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
    parser.add_argument("--phase-timeout", type=float, default=5.0)
    parser.add_argument("--max-output-characters", type=int, default=65_536)
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional new JSON path for this invocation summary",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tasks = _read_jsonl(args.tasks)
    validate_campaign_tasks(
        tasks,
        expected_count=EXPECTED_TASKS,
        expected_revision=MBPP_REVISION,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    campaign_tasks_path = args.run_dir / "selected_tasks_300.jsonl"
    publish_jsonl_once(campaign_tasks_path, tasks)

    config = GenerationConfig(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_logprobs=args.top_logprobs,
        max_tokens=args.max_tokens,
    )
    manifest = {
        "schema_version": "coding.collection.mbpp.rejection.v1",
        "dataset": {
            "dataset": "MBPP",
            "config": "full",
            "split": "train",
            "revision": MBPP_REVISION,
            "selection": "random",
            "seed": SELECTION_SEED,
            "selected_tasks": EXPECTED_TASKS,
            "ordered_task_ids_sha256": hashlib.sha256(
                "\n".join(task["id"] for task in tasks).encode("utf-8")
            ).hexdigest(),
            "task_jsonl_sha256": hashlib.sha256(
                campaign_tasks_path.read_bytes()
            ).hexdigest(),
        },
        "generation": {"model": config.model, **config.as_metadata()},
        "sampling": {
            "max_attempts_per_task": 3,
            "stop_after_first_pass": True,
            "blind": True,
        },
        "provider": {"name": "DeepSeek", "base_url": args.base_url},
    }
    _publish_json_once(args.run_dir / "campaign_manifest.json", manifest)

    client = DeepSeekClient(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    def report_wave(wave) -> None:
        print(json.dumps({"event": "wave_complete", **asdict(wave)}, ensure_ascii=False), flush=True)

    summary = run_rejection_sampling(
        selected_tasks_path=campaign_tasks_path,
        run_dir=args.run_dir,
        client=client,
        config=config,
        max_workers=args.workers,
        requests_per_minute=args.requests_per_minute,
        provider={"name": "DeepSeek", "base_url": args.base_url},
        phase_timeout_seconds=args.phase_timeout,
        max_output_characters=args.max_output_characters,
        progress=report_wave,
    )
    summary_record = asdict(summary)
    if args.summary_output is not None:
        _write_json_atomic_new(args.summary_output, summary_record)
    print(json.dumps({"event": "run_complete", **summary_record}, ensure_ascii=False), flush=True)
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def _publish_json_once(path: Path, value: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise FileExistsError(f"{path} already exists with different content")
        return
    _write_json_atomic_new(path, value)


def _write_json_atomic_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
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
    raise SystemExit(main())
