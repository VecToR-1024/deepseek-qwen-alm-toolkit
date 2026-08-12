#!/usr/bin/env python3
"""Run one append-only breadth-first collection for a pinned coding source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deepseek_distill.api import DEFAULT_BASE_URL, DeepSeekClient, GenerationConfig
from deepseek_distill.breadth_aggregation import aggregate_attempt_campaign
from deepseek_distill.multisource_tasks import MULTISOURCE_TASK_SCHEMA_VERSION
from deepseek_distill.rejection_sampling import (
    collect_rejection_sampling_raw,
    publish_json_once,
    publish_jsonl_once,
    run_rejection_sampling,
    validate_campaign_tasks,
)
from deepseek_distill.source_catalog import SOURCE_SPECS
from deepseek_distill.teacher_prompt import CLEAN_PROMPT_CONTRACT_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCE_SPECS), required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--import-summary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    mode.add_argument(
        "--raw-only",
        action="store_true",
        help=(
            "Collect only the append-only raw API queue for a later "
            "normalization/verification consumer."
        ),
    )
    mode.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect, normalize, and verify without aggregating accepted records.",
    )
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-minute", type=float, default=120)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--trace-profile",
        choices=("top20", "actual_only"),
        default="top20",
        help=(
            "Persist full top-k candidates for the legacy baseline, or only "
            "the actual generated-token trace required by ALM."
        ),
    )
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--phase-timeout", type=float, default=8.0)
    parser.add_argument("--max-output-characters", type=int, default=65_536)
    parser.add_argument("--verifier-workers", type=int, default=12)
    parser.add_argument(
        "--max-attempts-per-task",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Blind attempts per task; stop requesting after the first pass.",
    )
    parser.add_argument(
        "--streaming-pipeline",
        action="store_true",
        help=(
            "Overlap API collection, durable normalization, and verification "
            "using the existing append-only JSONL artifacts."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = SOURCE_SPECS[args.source]
    tasks = _read_jsonl(args.tasks)
    import_summary = _read_json(args.import_summary)
    _validate_import_summary(import_summary, tasks=tasks, source=args.source)
    validate_campaign_tasks(
        tasks,
        expected_count=len(tasks),
        expected_revision=spec.revision,
        expected_schema_version=MULTISOURCE_TASK_SCHEMA_VERSION,
        expected_dataset=spec.dataset_id,
        expected_config=spec.config,
        expected_split=spec.split,
    )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    campaign_tasks_path = args.run_dir / f"selected_tasks_{len(tasks)}.jsonl"
    task_status = publish_jsonl_once(campaign_tasks_path, tasks)
    if args.trace_profile == "actual_only" and args.top_logprobs != 20:
        raise ValueError(
            "--trace-profile actual_only must not be combined with a "
            "--top-logprobs override"
        )
    effective_top_logprobs = (
        None if args.trace_profile == "actual_only" else args.top_logprobs
    )
    config = GenerationConfig(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_logprobs=effective_top_logprobs,
        max_tokens=args.max_tokens,
    )
    manifest = _build_campaign_manifest(
        tasks=tasks,
        tasks_path=args.tasks,
        campaign_tasks_path=campaign_tasks_path,
        import_summary=import_summary,
        import_summary_path=args.import_summary,
        source=args.source,
        config=config,
        base_url=args.base_url,
        phase_timeout=args.phase_timeout,
        max_attempts_per_task=args.max_attempts_per_task,
    )
    manifest_status = publish_json_once(
        args.run_dir / "campaign_manifest.json",
        manifest,
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "event": "multisource_breadth_prepared",
                    "source": args.source,
                    "tasks": len(tasks),
                    "task_status": task_status,
                    "manifest_status": manifest_status,
                    "run_dir": args.run_dir.as_posix(),
                },
                ensure_ascii=False,
            )
        )
        return 0

    run_summary = None
    if not args.aggregate_only:
        client = DeepSeekClient(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        if args.raw_only:
            if args.max_attempts_per_task != 1:
                raise ValueError("--raw-only supports exactly one attempt per task")
            raw_summary = collect_rejection_sampling_raw(
                selected_tasks_path=campaign_tasks_path,
                run_dir=args.run_dir,
                client=client,
                config=config,
                max_workers=args.workers,
                requests_per_minute=args.requests_per_minute,
                provider={"name": "DeepSeek", "base_url": args.base_url},
            )
            complete = raw_summary.raw_attempts == len(tasks)
            print(
                json.dumps(
                    {
                        "event": "multisource_raw_collection_complete",
                        "source": args.source,
                        "collection_complete": complete,
                        "run": asdict(raw_summary),
                    },
                    ensure_ascii=False,
                )
            )
            return 0 if complete else 2
        run_summary = run_rejection_sampling(
            selected_tasks_path=campaign_tasks_path,
            run_dir=args.run_dir,
            client=client,
            config=config,
            max_workers=args.workers,
            requests_per_minute=args.requests_per_minute,
            provider={"name": "DeepSeek", "base_url": args.base_url},
            phase_timeout_seconds=args.phase_timeout,
            max_output_characters=args.max_output_characters,
            verification_workers=args.verifier_workers,
            max_attempts_per_task=args.max_attempts_per_task,
            streaming_pipeline=args.streaming_pipeline,
        )
    if args.collect_only:
        assert run_summary is not None
        complete = True
        print(
            json.dumps(
                {
                    "event": "multisource_breadth_collection_complete",
                    "source": args.source,
                    "collection_complete": complete,
                    "run": asdict(run_summary),
                },
                ensure_ascii=False,
            )
        )
        return 0 if complete else 2

    summary = aggregate_attempt_campaign(
        run_dir=args.run_dir,
        selected_tasks_path=campaign_tasks_path,
        target=len(tasks),
        max_attempts_per_task=args.max_attempts_per_task,
    )
    print(
        json.dumps(
            {
                "event": "multisource_breadth_complete",
                "source": args.source,
                "operation": "aggregate_only" if args.aggregate_only else "collect",
                "run": asdict(run_summary) if run_summary is not None else None,
                **summary,
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["collection_complete"] else 2


def _build_campaign_manifest(
    *,
    tasks: list[dict[str, Any]],
    tasks_path: Path,
    campaign_tasks_path: Path,
    import_summary: Mapping[str, Any],
    import_summary_path: Path,
    source: str,
    config: GenerationConfig,
    base_url: str,
    phase_timeout: float,
    max_attempts_per_task: int,
) -> dict[str, Any]:
    spec = SOURCE_SPECS[source]
    return {
        "schema_version": f"coding.collection.{source}.breadth.v1",
        "dataset": {
            "dataset": spec.dataset_id,
            "config": spec.config,
            "split": spec.split,
            "revision": spec.revision,
            "license": spec.license,
            "selection": import_summary["selection"],
            "seed": import_summary["seed"],
            "selected_tasks": len(tasks),
            "ordered_task_ids_sha256": _ordered_id_hash(tasks),
            "input_tasks_path": tasks_path.as_posix(),
            "input_tasks_sha256": _sha256(tasks_path),
            "campaign_tasks_path": campaign_tasks_path.as_posix(),
            "campaign_tasks_sha256": _sha256(campaign_tasks_path),
            "import_summary_path": import_summary_path.as_posix(),
            "import_summary_sha256": _sha256(import_summary_path),
            "source_notes": list(import_summary.get("source_notes") or []),
            "difficulty_profile": import_summary.get("difficulty_profile"),
            "difficulty_policy": import_summary.get("difficulty_policy"),
        },
        "prompt_contract": {
            "id": CLEAN_PROMPT_CONTRACT_ID,
            "interface_types": sorted(
                {str(task.get("interface_type")) for task in tasks}
            ),
        },
        "generation": {"model": config.model, **config.as_metadata()},
        "sampling": {
            "max_attempts_per_task": max_attempts_per_task,
            "stop_after_first_pass": True,
            "blind": True,
            "verifier_feedback": False,
            "strategy": (
                "breadth_first"
                if max_attempts_per_task == 1
                else "blind_rejection_sampling"
            ),
        },
        "provider": {"name": "DeepSeek", "base_url": base_url},
        "verification": {
            "phase_timeout_seconds": phase_timeout,
            "output_comparison": (
                "normalize_newlines_strip_outer_whitespace_any_expected_v1"
            ),
            "host_security_boundary": "child_process_not_security_sandbox",
        },
    }


def _validate_import_summary(
    summary: Mapping[str, Any],
    *,
    tasks: list[dict[str, Any]],
    source: str,
) -> None:
    spec = SOURCE_SPECS[source]
    if summary.get("schema_version") != "coding.import.multisource.v1":
        raise ValueError("import summary has an incompatible schema_version")
    if summary.get("status") != "ok" or summary.get("source") != source:
        raise ValueError("import summary does not match the requested source")
    if summary.get("tasks") != len(tasks):
        raise ValueError("import summary task count does not match task JSONL")
    dataset = summary.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("import summary dataset metadata is missing")
    expected_dataset = {
        "id": spec.dataset_id,
        "config": spec.config,
        "split": spec.split,
        "revision": spec.revision,
        "license": spec.license,
        "provenance": spec.provenance,
        "mirror": spec.mirror,
    }
    if dict(dataset) != expected_dataset:
        raise ValueError("import summary dataset metadata does not match the pin")
    ordered_ids = [task.get("id") for task in tasks]
    if summary.get("ordered_task_ids") != ordered_ids:
        raise ValueError("import summary task order does not match task JSONL")
    if summary.get("ordered_task_ids_sha256") != _ordered_id_hash(tasks):
        raise ValueError("import summary ordered task hash does not match task JSONL")
    selection = summary.get("selection")
    seed = summary.get("seed")
    if selection not in {"first", "random"}:
        raise ValueError("import summary selection is invalid")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("import summary seed is invalid")


def _ordered_id_hash(tasks: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(task["id"]) for task in tasks).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    if not records:
        raise ValueError(f"{path}: task JSONL must not be empty")
    return records


if __name__ == "__main__":
    raise SystemExit(main())
