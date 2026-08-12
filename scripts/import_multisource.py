#!/usr/bin/env python3
"""Import an immutable task subset from a pinned external coding source."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from deepseek_distill.multisource_tasks import MULTISOURCE_TASK_SCHEMA_VERSION
from deepseek_distill.hard_tasks import (
    HARD_DIFFICULTY_PROFILE,
    HARD_PROFILE_SOURCES,
    hard_profile_metadata,
)
from deepseek_distill.rejection_sampling import publish_json_once, publish_jsonl_once
from deepseek_distill.source_catalog import Loader, SOURCE_SPECS, SourceSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCE_SPECS), required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--selection", choices=("first", "random"), default="random")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--revision",
        help="Optional assertion; must equal the source's pinned revision.",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--difficulty-profile",
        choices=(HARD_DIFFICULTY_PROFILE,),
        help="Apply a frozen source-specific difficulty filter before selection.",
    )
    parser.add_argument(
        "--exclude-tasks",
        type=Path,
        action="append",
        default=[],
        help=(
            "JSONL task artifact whose IDs must not be selected. Repeatable; "
            "the loader requests enough extra candidates to preserve --limit."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    loaders: Mapping[str, Loader] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    spec = SOURCE_SPECS[args.source]
    revision = args.revision or spec.revision
    if revision != spec.revision:
        raise ValueError(
            f"{spec.key} must use pinned revision {spec.revision!r}; "
            f"received {revision!r}"
        )
    excluded_ids, exclusion_summary = _load_exclusions(
        args.exclude_tasks,
        spec=spec,
    )
    loader_requested_tasks = args.limit + len(excluded_ids)
    selected_loader = (loaders or {}).get(spec.key, spec.loader)
    loader_kwargs: dict[str, Any] = {
        "limit": loader_requested_tasks,
        "selection": args.selection,
        "seed": args.seed,
        "revision": revision,
        "cache_dir": args.cache_dir,
    }
    if args.difficulty_profile is not None:
        if spec.key not in HARD_PROFILE_SOURCES:
            raise ValueError(
                f"difficulty profile {args.difficulty_profile!r} does not support "
                f"source {spec.key!r}"
            )
        loader_kwargs["difficulty_profile"] = args.difficulty_profile
    candidates = selected_loader(
        **loader_kwargs,
    )
    tasks = [
        task for task in candidates if task.get("id") not in excluded_ids
    ][: args.limit]
    _validate_tasks(tasks, spec=spec, expected_count=args.limit)
    persisted_summary = _build_summary(
        tasks,
        spec=spec,
        selection=args.selection,
        seed=args.seed,
        exclusion_summary=exclusion_summary,
        loader_requested_tasks=loader_requested_tasks,
        difficulty_profile=args.difficulty_profile,
    )
    output_status = publish_jsonl_once(args.output, tasks)
    summary_status = publish_json_once(args.summary_output, persisted_summary)
    runtime_summary = {
        **persisted_summary,
        "output": args.output.as_posix(),
        "summary_output": args.summary_output.as_posix(),
        "output_status": output_status,
        "summary_status": summary_status,
    }
    print(json.dumps(runtime_summary, ensure_ascii=False))
    return 0


def _validate_tasks(
    tasks: list[dict[str, Any]],
    *,
    spec: SourceSpec,
    expected_count: int,
) -> None:
    if len(tasks) != expected_count:
        raise ValueError(
            f"{spec.key} loader returned {len(tasks)} tasks; expected {expected_count}"
        )
    seen: set[str] = set()
    for position, task in enumerate(tasks):
        if task.get("schema_version") != MULTISOURCE_TASK_SCHEMA_VERSION:
            raise ValueError(
                f"{spec.key} task {position} has an incompatible schema_version"
            )
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{spec.key} task {position} has no valid id")
        if task_id in seen:
            raise ValueError(f"{spec.key} task id {task_id!r} is duplicated")
        seen.add(task_id)
        source = task.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"{spec.key} task {task_id!r} has no source metadata")
        expected = {
            "dataset": spec.dataset_id,
            "config": spec.config,
            "split": spec.split,
            "revision": spec.revision,
            "license": spec.license,
            "provenance": spec.provenance,
            "mirror": spec.mirror,
        }
        for field, value in expected.items():
            if source.get(field) != value:
                raise ValueError(
                    f"{spec.key} task {task_id!r} source.{field} must be {value!r}"
                )


def _build_summary(
    tasks: list[dict[str, Any]],
    *,
    spec: SourceSpec,
    selection: str,
    seed: int,
    exclusion_summary: Mapping[str, Any],
    loader_requested_tasks: int,
    difficulty_profile: str | None,
) -> dict[str, Any]:
    ordered_ids = [task["id"] for task in tasks]
    summary = {
        "schema_version": "coding.import.multisource.v1",
        "status": "ok",
        "source": spec.key,
        "task_schema_version": MULTISOURCE_TASK_SCHEMA_VERSION,
        "tasks": len(tasks),
        "selection": selection,
        "seed": seed,
        "dataset": {
            "id": spec.dataset_id,
            "config": spec.config,
            "split": spec.split,
            "revision": spec.revision,
            "license": spec.license,
            "provenance": spec.provenance,
            "mirror": spec.mirror,
        },
        "source_notes": list(spec.notes),
        "ordered_task_ids": ordered_ids,
        "ordered_task_ids_sha256": hashlib.sha256(
            "\n".join(ordered_ids).encode("utf-8")
        ).hexdigest(),
        "immutable_output": True,
    }
    if difficulty_profile is not None:
        summary["difficulty_profile"] = difficulty_profile
        summary["difficulty_policy"] = hard_profile_metadata()
    if exclusion_summary.get("input_files"):
        summary["loader_requested_tasks"] = loader_requested_tasks
        summary["exclusions"] = dict(exclusion_summary)
    return summary


def _load_exclusions(
    paths: Sequence[Path],
    *,
    spec: SourceSpec,
) -> tuple[set[str], dict[str, Any]]:
    excluded_ids: set[str] = set()
    inputs: list[dict[str, Any]] = []
    for path in paths:
        records = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    task = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {error.msg}"
                    ) from error
                if not isinstance(task, Mapping):
                    raise ValueError(
                        f"{path}:{line_number}: expected a JSON object"
                    )
                task_id = task.get("id")
                source = task.get("source")
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError(f"{path}:{line_number}: task id is invalid")
                if (
                    not isinstance(source, Mapping)
                    or source.get("dataset") != spec.dataset_id
                    or source.get("config") != spec.config
                    or source.get("split") != spec.split
                    or source.get("revision") != spec.revision
                ):
                    raise ValueError(
                        f"{path}:{line_number}: task {task_id!r} "
                        f"does not match source {spec.key!r}"
                    )
                excluded_ids.add(task_id)
                records += 1
        inputs.append(
            {
                "path": path.as_posix(),
                "records": records,
                "sha256": _sha256(path),
            }
        )
    ordered_ids = sorted(excluded_ids)
    return excluded_ids, {
        "unique_task_ids": len(ordered_ids),
        "task_ids_sha256": hashlib.sha256(
            "\n".join(ordered_ids).encode("utf-8")
        ).hexdigest(),
        "input_files": inputs,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
