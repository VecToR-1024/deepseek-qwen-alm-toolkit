"""Deterministic breadth-first TACO task selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .rejection_sampling import validate_campaign_tasks
from .taco import (
    TACO_DATASET_ID,
    TACO_REVISION,
    TACO_SPLIT,
    TACO_TASK_SCHEMA_VERSION,
    TACO_TRAIN_SHARD,
    import_taco_rows,
)


TACO_BREADTH_TASK_COUNT = 1000
TACO_BREADTH_SEED = 20260728
TACO_BREADTH_SELECTION_SCOPE = "single_pinned_train_shard_breadth_v2"
TACO_BREADTH_EXCLUDED_SOURCES = frozenset({"geeksforgeeks"})


def select_taco_breadth_tasks(
    rows: Iterable[Mapping[str, Any]],
    *,
    prior_tasks: Sequence[Mapping[str, Any]],
    limit: int = TACO_BREADTH_TASK_COUNT,
    seed: int = TACO_BREADTH_SEED,
    revision: str = TACO_REVISION,
    shard_path: str = TACO_TRAIN_SHARD,
) -> list[dict[str, Any]]:
    """Select new stdio tasks after applying the frozen breadth exclusions."""
    return import_taco_rows(
        rows,
        limit=limit,
        selection="random",
        seed=seed,
        revision=revision,
        shard_path=shard_path,
        excluded_task_ids=_task_ids(prior_tasks, label="prior task"),
        excluded_sources=TACO_BREADTH_EXCLUDED_SOURCES,
        selection_scope=TACO_BREADTH_SELECTION_SCOPE,
    )


def validate_taco_breadth_tasks(
    tasks: list[Mapping[str, Any]],
    *,
    prior_tasks: Sequence[Mapping[str, Any]],
    expected_count: int = TACO_BREADTH_TASK_COUNT,
) -> None:
    """Validate the immutable breadth campaign before any provider request."""
    validate_campaign_tasks(
        tasks,
        expected_count=expected_count,
        expected_revision=TACO_REVISION,
        expected_schema_version=TACO_TASK_SCHEMA_VERSION,
        expected_dataset=TACO_DATASET_ID,
        expected_config=None,
        expected_split=TACO_SPLIT,
    )
    prior_ids = _task_ids(prior_tasks, label="prior task")
    overlap = prior_ids & {str(task["id"]) for task in tasks}
    if overlap:
        raise ValueError(f"breadth tasks overlap prior task ids: {sorted(overlap)!r}")

    expected_sources = sorted(TACO_BREADTH_EXCLUDED_SOURCES)
    for position, task in enumerate(tasks):
        source = task.get("source")
        metadata = task.get("metadata")
        if not isinstance(source, Mapping) or source.get("shard") != TACO_TRAIN_SHARD:
            raise ValueError(f"breadth task {position} must use the pinned train shard")
        original_source = source.get("original_source")
        if (
            isinstance(original_source, str)
            and original_source.strip().lower() in TACO_BREADTH_EXCLUDED_SOURCES
        ):
            raise ValueError(f"breadth task {position} uses an excluded source")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("selection_scope") != TACO_BREADTH_SELECTION_SCOPE
        ):
            raise ValueError(
                f"breadth task {position} does not record the selection scope"
            )
        eligibility = metadata.get("eligibility")
        if (
            not isinstance(eligibility, Mapping)
            or eligibility.get("excluded_sources") != expected_sources
        ):
            raise ValueError(
                f"breadth task {position} does not record the eligibility exclusions"
            )


def _task_ids(
    tasks: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> set[str]:
    ids: set[str] = set()
    for position, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise ValueError(f"{label} {position} must be an object")
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{label} {position} has no valid id")
        if task_id in ids:
            raise ValueError(f"duplicate {label} id {task_id!r}")
        ids.add(task_id)
    return ids
