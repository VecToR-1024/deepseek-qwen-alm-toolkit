"""Pinned DeepMind CodeContests train ingestion with bounded random selection."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .multisource_tasks import (
    build_multisource_teacher_messages,
    make_multisource_task,
)
from .hard_tasks import is_hard_task


CODE_CONTESTS_DATASET_ID = "deepmind/code_contests"
CODE_CONTESTS_CONFIG = "default"
CODE_CONTESTS_SPLIT = "train"
CODE_CONTESTS_REVISION = "802411c3010cb00d1b05bad57ca77365a3c699d6"
CODE_CONTESTS_LICENSE = "CC-BY-4.0"
CODE_CONTESTS_PROVENANCE = "https://github.com/google-deepmind/code_contests"
CODE_CONTESTS_MIRROR = (
    "https://huggingface.co/datasets/deepmind/code_contests"
)
CODE_CONTESTS_DEFAULT_SEED = 20260731
_STDIO_INTERFACE = "Complete Python program using standard input and standard output."

build_teacher_messages = build_multisource_teacher_messages


def import_code_contests_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    selection: str = "random",
    seed: int = CODE_CONTESTS_DEFAULT_SEED,
    revision: str = CODE_CONTESTS_REVISION,
    difficulty_profile: str | None = None,
) -> list[dict[str, Any]]:
    """Select unique train problems without retaining the full 35 GB dataset."""

    _validate_selection(limit=limit, selection=selection, seed=seed)
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")

    seen_ids: set[str] = set()
    eligible_count = 0
    first_tasks: list[dict[str, Any]] = []
    random_heap: list[tuple[int, str, dict[str, Any]]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"CodeContests row {position} must be an object")
        if not _eligible_file_contract(row):
            continue
        task = _build_task(row, position=position, revision=revision)
        if task is None:
            continue
        if difficulty_profile is not None and not is_hard_task(
            "code-contests",
            task,
            profile=difficulty_profile,
        ):
            continue
        task_id = task["id"]
        if task_id in seen_ids:
            raise ValueError(f"duplicate CodeContests problem identity {task_id!r}")
        seen_ids.add(task_id)
        eligible_count += 1
        if selection == "first":
            first_tasks.append(task)
            if len(first_tasks) == limit:
                return first_tasks
            continue

        rank = _selection_rank(seed, task_id)
        entry = (-rank, task_id, task)
        if len(random_heap) < limit:
            heapq.heappush(random_heap, entry)
        elif rank < -random_heap[0][0]:
            heapq.heapreplace(random_heap, entry)

    if eligible_count < limit:
        raise ValueError(
            f"requested {limit} CodeContests tasks but only "
            f"{eligible_count} eligible tasks are available"
        )
    if selection == "first":
        return first_tasks
    return [
        task
        for negative_rank, task_id, task in sorted(
            random_heap,
            key=lambda item: (-item[0], item[1]),
        )
    ]


def load_code_contests_tasks(
    *,
    limit: int,
    selection: str = "random",
    seed: int = CODE_CONTESTS_DEFAULT_SEED,
    revision: str = CODE_CONTESTS_REVISION,
    cache_dir: str | Path | None = None,
    difficulty_profile: str | None = None,
) -> list[dict[str, Any]]:
    """Stream native pinned Parquet rows through the bounded selector."""

    if cache_dir is not None:
        os.environ["HF_HOME"] = str(Path(cache_dir).resolve())
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "CodeContests import requires the optional 'data' dependencies"
        ) from error

    dataset = load_dataset(
        CODE_CONTESTS_DATASET_ID,
        CODE_CONTESTS_CONFIG,
        split=CODE_CONTESTS_SPLIT,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        streaming=True,
    )
    return import_code_contests_rows(
        dataset,
        limit=limit,
        selection=selection,
        seed=seed,
        revision=revision,
        difficulty_profile=difficulty_profile,
    )


def _build_task(
    row: Mapping[str, Any],
    *,
    position: int,
    revision: str,
) -> dict[str, Any] | None:
    problem_text = _required_text(
        row.get("description"),
        f"CodeContests row {position}.description",
    )
    name = _required_text(
        row.get("name"),
        f"CodeContests row {position}.name",
    )
    identity = _problem_identity(row, name=name, problem_text=problem_text)
    tests: list[dict[str, str]] = []
    test_counts: dict[str, int] = {}
    for label in ("public", "private", "generated"):
        group = _normalize_test_group(
            row.get(f"{label}_tests"),
            context=f"CodeContests row {position}.{label}_tests",
        )
        test_counts[label] = len(group)
        tests.extend(group)
    if not tests:
        return None

    correct_solutions = _solution_payload(
        row.get("solutions"),
        context=f"CodeContests row {position}.solutions",
    )
    incorrect_solutions = _solution_payload(
        row.get("incorrect_solutions"),
        context=f"CodeContests row {position}.incorrect_solutions",
    )
    return make_multisource_task(
        task_id=f"codecontests_train_{identity[:20]}",
        source={
            "dataset": CODE_CONTESTS_DATASET_ID,
            "config": CODE_CONTESTS_CONFIG,
            "split": CODE_CONTESTS_SPLIT,
            "original_id": identity,
            "revision": revision,
            "license": CODE_CONTESTS_LICENSE,
            "provenance": CODE_CONTESTS_PROVENANCE,
            "mirror": CODE_CONTESTS_MIRROR,
            "original_name": name,
            "original_source": row.get("source"),
        },
        problem_text=problem_text,
        interface_type="stdin_stdout",
        required_interface=_STDIO_INTERFACE,
        tests=tests,
        metadata={
            "difficulty": row.get("difficulty"),
            "test_counts": test_counts,
            "reference_solution_count": len(correct_solutions["solution"]),
            "reference_solutions_sha256": _json_digest(correct_solutions),
            "incorrect_solution_count": len(incorrect_solutions["solution"]),
            "incorrect_solutions_sha256": _json_digest(incorrect_solutions),
            "codeforces": {
                "contest_id": row.get("cf_contest_id"),
                "index": row.get("cf_index"),
                "points": row.get("cf_points"),
                "rating": row.get("cf_rating"),
                "tags": row.get("cf_tags") or [],
            },
            "is_description_translated": row.get("is_description_translated"),
            "untranslated_description_sha256": _text_digest(
                row.get("untranslated_description")
            ),
            "time_limit": row.get("time_limit"),
            "memory_limit_bytes": row.get("memory_limit_bytes"),
            "eligibility": {
                "interface_type": "stdin_stdout",
                "public_private_generated_tests_retained": True,
                "file_io_tasks_excluded": True,
            },
        },
    )


def _problem_identity(
    row: Mapping[str, Any],
    *,
    name: str,
    problem_text: str,
) -> str:
    value = {
        "name": name,
        "source": row.get("source"),
        "cf_contest_id": row.get("cf_contest_id"),
        "cf_index": row.get("cf_index"),
        "description": problem_text,
    }
    return _json_digest(value)


def _normalize_test_group(
    value: Any,
    *,
    context: str,
) -> list[dict[str, str]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    inputs = value.get("input")
    outputs = value.get("output")
    if (
        not isinstance(inputs, list)
        or not isinstance(outputs, list)
        or len(inputs) != len(outputs)
        or any(not isinstance(item, str) for item in [*inputs, *outputs])
    ):
        raise ValueError(
            f"{context} input/output must be equal-length string lists"
        )
    return [
        {"input": test_input, "output": expected_output}
        for test_input, expected_output in zip(inputs, outputs)
    ]


def _solution_payload(value: Any, *, context: str) -> dict[str, list[Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    languages = value.get("language")
    solutions = value.get("solution")
    if (
        not isinstance(languages, list)
        or not isinstance(solutions, list)
        or len(languages) != len(solutions)
        or any(not isinstance(solution, str) for solution in solutions)
    ):
        raise ValueError(
            f"{context} language/solution must be equal-length lists"
        )
    return {
        "language": list(languages),
        "solution": list(solutions),
    }


def _eligible_file_contract(row: Mapping[str, Any]) -> bool:
    return row.get("input_file") in {None, ""} and row.get("output_file") in {
        None,
        "",
    }


def _selection_rank(seed: int, task_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{task_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _text_digest(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    if not isinstance(value, str):
        raise ValueError("untranslated_description must be a string or null")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _required_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _validate_selection(*, limit: int, selection: str, seed: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if selection not in {"first", "random"}:
        raise ValueError("selection must be 'first' or 'random'")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
