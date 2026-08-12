"""Pinned Open-R1 Codeforces ingestion for exact-output stdio verification."""

from __future__ import annotations

import hashlib
import heapq
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .multisource_tasks import (
    build_multisource_teacher_messages,
    make_multisource_task,
)
from .hard_tasks import is_hard_task


OPEN_R1_CODEFORCES_DATASET_ID = "open-r1/codeforces"
OPEN_R1_CODEFORCES_CONFIG = "verifiable"
OPEN_R1_CODEFORCES_SPLIT = "train"
OPEN_R1_CODEFORCES_REVISION = "fbe3f6e903ee854eec2e69e9d96d0306cde59baf"
OPEN_R1_CODEFORCES_LICENSE = "CC-BY-4.0"
OPEN_R1_CODEFORCES_PROVENANCE = (
    "https://huggingface.co/datasets/open-r1/codeforces"
)
OPEN_R1_CODEFORCES_MIRROR = OPEN_R1_CODEFORCES_PROVENANCE
OPEN_R1_CODEFORCES_DEFAULT_SEED = 20260803
_STDIO_INTERFACE = "Complete Python program using standard input and standard output."

build_teacher_messages = build_multisource_teacher_messages


def import_open_r1_codeforces_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    selection: str = "random",
    seed: int = OPEN_R1_CODEFORCES_DEFAULT_SEED,
    revision: str = OPEN_R1_CODEFORCES_REVISION,
    difficulty_profile: str | None = None,
) -> list[dict[str, Any]]:
    """Select rows whose correctness the existing exact-output verifier can prove."""

    _validate_selection(limit=limit, selection=selection, seed=seed)
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")

    seen_alias_groups: set[str] = set()
    eligible_count = 0
    first_tasks: list[dict[str, Any]] = []
    random_heap: list[tuple[int, str, dict[str, Any]]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Open-R1 Codeforces row {position} must be an object")
        if not _eligible_exact_output_contract(row):
            continue
        task, alias_group = _build_task(row, position=position, revision=revision)
        if difficulty_profile is not None and not is_hard_task(
            "open-r1-codeforces",
            task,
            profile=difficulty_profile,
        ):
            continue
        if alias_group in seen_alias_groups:
            continue
        seen_alias_groups.add(alias_group)
        eligible_count += 1

        if selection == "first":
            first_tasks.append(task)
            if len(first_tasks) == limit:
                return first_tasks
            continue

        rank = _selection_rank(seed, task["id"])
        entry = (-rank, task["id"], task)
        if len(random_heap) < limit:
            heapq.heappush(random_heap, entry)
        elif rank < -random_heap[0][0]:
            heapq.heapreplace(random_heap, entry)

    if eligible_count < limit:
        raise ValueError(
            f"requested {limit} Open-R1 Codeforces tasks but only "
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


def load_open_r1_codeforces_tasks(
    *,
    limit: int,
    selection: str = "random",
    seed: int = OPEN_R1_CODEFORCES_DEFAULT_SEED,
    revision: str = OPEN_R1_CODEFORCES_REVISION,
    cache_dir: str | Path | None = None,
    difficulty_profile: str | None = None,
) -> list[dict[str, Any]]:
    """Stream the pinned train split through the bounded selector."""

    if cache_dir is not None:
        os.environ["HF_HOME"] = str(Path(cache_dir).resolve())
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Open-R1 Codeforces import requires the optional 'data' dependencies"
        ) from error

    dataset = load_dataset(
        OPEN_R1_CODEFORCES_DATASET_ID,
        OPEN_R1_CODEFORCES_CONFIG,
        split=OPEN_R1_CODEFORCES_SPLIT,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        streaming=True,
    )
    return import_open_r1_codeforces_rows(
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
) -> tuple[dict[str, Any], str]:
    original_id = _required_text(
        row.get("id"),
        f"Open-R1 Codeforces row {position}.id",
    )
    aliases = _string_list(
        row.get("aliases"),
        context=f"Open-R1 Codeforces row {position}.aliases",
    )
    alias_group = min({original_id, *aliases})
    tests = _normalize_tests(
        row.get("official_tests"),
        context=f"Open-R1 Codeforces row {position}.official_tests",
    )
    examples = _normalize_tests(
        row.get("examples") or [],
        context=f"Open-R1 Codeforces row {position}.examples",
        allow_empty=True,
    )
    title = _required_text(
        row.get("title"),
        f"Open-R1 Codeforces row {position}.title",
    )
    description = _required_text(
        row.get("description"),
        f"Open-R1 Codeforces row {position}.description",
    )
    problem_text = _build_problem_text(
        title=title,
        description=description,
        input_format=row.get("input_format"),
        output_format=row.get("output_format"),
        note=row.get("note"),
        examples=examples,
        position=position,
    )
    task_digest = hashlib.sha256(alias_group.encode("utf-8")).hexdigest()[:20]
    editorial = _optional_text(
        row.get("editorial"),
        f"Open-R1 Codeforces row {position}.editorial",
    )
    return (
        make_multisource_task(
            task_id=f"openr1_codeforces_train_{task_digest}",
            source={
                "dataset": OPEN_R1_CODEFORCES_DATASET_ID,
                "config": OPEN_R1_CODEFORCES_CONFIG,
                "split": OPEN_R1_CODEFORCES_SPLIT,
                "original_id": original_id,
                "revision": revision,
                "license": OPEN_R1_CODEFORCES_LICENSE,
                "provenance": OPEN_R1_CODEFORCES_PROVENANCE,
                "mirror": OPEN_R1_CODEFORCES_MIRROR,
            },
            problem_text=problem_text,
            interface_type="stdin_stdout",
            required_interface=_STDIO_INTERFACE,
            tests=tests,
            metadata={
                "aliases": sorted(set(aliases)),
                "alias_group_identity": alias_group,
                "contest": {
                    "id": row.get("contest_id"),
                    "name": row.get("contest_name"),
                    "type": row.get("contest_type"),
                    "start": row.get("contest_start"),
                    "start_year": row.get("contest_start_year"),
                    "problem_index": row.get("index"),
                },
                "rating": row.get("rating"),
                "tags": _string_list(
                    row.get("tags"),
                    context=f"Open-R1 Codeforces row {position}.tags",
                ),
                "time_limit_seconds": row.get("time_limit"),
                "memory_limit_megabytes": row.get("memory_limit"),
                "testset_size": row.get("testset_size"),
                "official_test_count": len(tests),
                "example_count": len(examples),
                "generated_test_count": row.get("generated_tests"),
                "editorial_sha256": (
                    hashlib.sha256(editorial.encode("utf-8")).hexdigest()
                    if editorial
                    else None
                ),
                "eligibility": {
                    "interface_type": "stdin_stdout",
                    "executable": True,
                    "official_tests_complete": True,
                    "custom_checker_excluded": True,
                    "interactive_tasks_excluded": True,
                    "file_io_tasks_excluded": True,
                    "generated_tests_not_required": True,
                },
            },
        ),
        alias_group,
    )


def _build_problem_text(
    *,
    title: str,
    description: str,
    input_format: Any,
    output_format: Any,
    note: Any,
    examples: list[dict[str, str]],
    position: int,
) -> str:
    sections = [f"Title: {title}", description]
    for label, value in (
        ("Input", input_format),
        ("Output", output_format),
        ("Note", note),
    ):
        text = _optional_text(
            value,
            f"Open-R1 Codeforces row {position}.{label.lower()}_format",
        )
        if text:
            sections.append(f"{label}:\n{text}")
    if examples:
        rendered_examples = []
        for index, example in enumerate(examples, start=1):
            rendered_examples.append(
                f"Example {index} input:\n{example['input']}\n"
                f"Example {index} output:\n{example['output']}"
            )
        sections.append("\n\n".join(rendered_examples))
    return "\n\n".join(sections)


def _eligible_exact_output_contract(row: Mapping[str, Any]) -> bool:
    tests = row.get("official_tests")
    checker = row.get("generated_checker")
    interaction = row.get("interaction_format")
    title = row.get("title")
    description = row.get("description")
    return (
        row.get("executable") is True
        and row.get("official_tests_complete") is True
        and row.get("input_mode") == "stdio"
        and (checker is None or checker == "")
        and (interaction is None or interaction == "")
        and isinstance(title, str)
        and bool(title.strip())
        and isinstance(description, str)
        and bool(description.strip())
        and isinstance(tests, list)
        and bool(tests)
    )


def _normalize_tests(
    value: Any,
    *,
    context: str,
    allow_empty: bool = False,
) -> list[dict[str, str]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{context} must be a non-empty list")
    tests: list[dict[str, str]] = []
    for index, test in enumerate(value):
        if not isinstance(test, Mapping):
            raise ValueError(f"{context}[{index}] must be an object")
        test_input = test.get("input")
        test_output = test.get("output")
        if not isinstance(test_input, str) or not isinstance(test_output, str):
            raise ValueError(f"{context}[{index}] input/output must be strings")
        tests.append({"input": test_input, "output": test_output})
    return tests


def _string_list(value: Any, *, context: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{context} must be a list of strings")
    return list(value)


def _required_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, context: str) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string or null")
    return value.strip()


def _selection_rank(seed: int, task_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{task_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def _validate_selection(*, limit: int, selection: str, seed: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if selection not in {"first", "random"}:
        raise ValueError("selection must be 'first' or 'random'")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
