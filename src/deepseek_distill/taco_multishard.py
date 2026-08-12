"""Bounded, revision-pinned ingestion from TACO train shards 1 through 8."""

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
from .taco import (
    TACO_CARD,
    TACO_DATASET_ID,
    TACO_LICENSE,
    TACO_PROVENANCE,
    TACO_REVISION,
    TACO_SPLIT,
)


TACO_MULTISHARD_CONFIG = "ALL"
TACO_MULTISHARD_DEFAULT_SEED = 20260731
TACO_MULTISHARD_INDICES = tuple(range(1, 9))
TACO_MULTISHARD_FILES = tuple(
    f"train/data-{index:05d}-of-00009.arrow"
    for index in TACO_MULTISHARD_INDICES
)
TACO_MULTISHARD_EXCLUDED_SOURCES = frozenset({"geeksforgeeks", "hackerrank"})
_STDIO_INTERFACE = "Complete Python program using standard input and standard output."

build_teacher_messages = build_multisource_teacher_messages


def import_taco_multishard_rows(
    shards: Iterable[tuple[int, Iterable[Mapping[str, Any]]]],
    *,
    limit: int,
    selection: str = "random",
    seed: int = TACO_MULTISHARD_DEFAULT_SEED,
    revision: str = TACO_REVISION,
    difficulty_profile: str | None = None,
) -> list[dict[str, Any]]:
    """Select unique stdin/stdout tasks from non-zero TACO train shards."""

    _validate_selection(limit=limit, selection=selection, seed=seed)
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")
    seen_shards: set[int] = set()
    seen_task_ids: set[str] = set()
    eligible_count = 0
    first_tasks: list[dict[str, Any]] = []
    random_heap: list[tuple[int, str, dict[str, Any]]] = []

    for shard_index, rows in shards:
        if (
            isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or shard_index not in TACO_MULTISHARD_INDICES
        ):
            raise ValueError("TACO multi-shard index must be between 1 and 8")
        if shard_index in seen_shards:
            raise ValueError(f"duplicate TACO shard index {shard_index}")
        seen_shards.add(shard_index)
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"TACO shard {shard_index} row {row_index} must be an object"
                )
            task = _build_task(
                row,
                shard_index=shard_index,
                row_index=row_index,
                revision=revision,
            )
            if task is None:
                continue
            if difficulty_profile is not None and not is_hard_task(
                "taco-multishard",
                task,
                profile=difficulty_profile,
            ):
                continue
            task_id = task["id"]
            if task_id in seen_task_ids:
                raise ValueError(f"duplicate TACO task id {task_id!r}")
            seen_task_ids.add(task_id)
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
            f"requested {limit} TACO multi-shard tasks but only "
            f"{eligible_count} eligible tasks are available"
        )
    if selection == "first":
        return first_tasks
    return [
        task
        for _, _, task in sorted(
            random_heap,
            key=lambda item: (-item[0], item[1]),
        )
    ]


def load_taco_multishard_tasks(
    *,
    limit: int,
    selection: str = "random",
    seed: int = TACO_MULTISHARD_DEFAULT_SEED,
    revision: str = TACO_REVISION,
    cache_dir: str | Path | None = None,
    difficulty_profile: str | None = None,
) -> list[dict[str, Any]]:
    """Download and memory-map the eight pinned non-zero Arrow shards."""

    if cache_dir is not None:
        os.environ["HF_HOME"] = str(Path(cache_dir).resolve())
    try:
        from datasets import Dataset
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "TACO multi-shard import requires the optional 'data' dependencies"
        ) from error

    def shard_rows() -> Iterable[tuple[int, Iterable[Mapping[str, Any]]]]:
        for shard_index, filename in zip(
            TACO_MULTISHARD_INDICES,
            TACO_MULTISHARD_FILES,
        ):
            local_path = hf_hub_download(
                repo_id=TACO_DATASET_ID,
                repo_type="dataset",
                filename=filename,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
            yield shard_index, Dataset.from_file(local_path)

    return import_taco_multishard_rows(
        shard_rows(),
        limit=limit,
        selection=selection,
        seed=seed,
        revision=revision,
        difficulty_profile=difficulty_profile,
    )


def _build_task(
    row: Mapping[str, Any],
    *,
    shard_index: int,
    row_index: int,
    revision: str,
) -> dict[str, Any] | None:
    question = row.get("question")
    starter_code = row.get("starter_code")
    source_name = row.get("source")
    picture_count = row.get("picture_num")
    normalized_source = (
        source_name.strip().lower() if isinstance(source_name, str) else ""
    )
    if (
        not isinstance(question, str)
        or not question.strip()
        or (starter_code is not None and starter_code != "")
        or picture_count not in (None, 0, "0", "")
        or normalized_source in TACO_MULTISHARD_EXCLUDED_SOURCES
    ):
        return None
    raw_contract = row.get("input_output")
    if raw_contract is None or raw_contract == "":
        return None
    contract = _parse_json_object(
        raw_contract,
        context=f"TACO shard {shard_index} row {row_index}.input_output",
    )
    function_name = contract.get("fn_name")
    if function_name is not None and function_name != "":
        return None
    inputs = contract.get("inputs")
    outputs = contract.get("outputs")
    if (
        not isinstance(inputs, list)
        or not inputs
        or not isinstance(outputs, list)
        or len(inputs) != len(outputs)
        or any(not isinstance(value, str) for value in [*inputs, *outputs])
    ):
        return None
    reference_solutions = _parse_json_string_list(
        row.get("solutions"),
        context=f"TACO shard {shard_index} row {row_index}.solutions",
    )
    raw_file = f"train/data-{shard_index:05d}-of-00009.arrow"
    task_id = f"taco_train_s{shard_index:02d}_r{row_index:06d}"
    return make_multisource_task(
        task_id=task_id,
        source={
            "dataset": TACO_DATASET_ID,
            "config": TACO_MULTISHARD_CONFIG,
            "split": TACO_SPLIT,
            "original_id": f"{shard_index}:{row_index}",
            "revision": revision,
            "license": TACO_LICENSE,
            "provenance": TACO_PROVENANCE,
            "mirror": TACO_CARD,
            "raw_file": raw_file,
            "shard_index": shard_index,
            "row_index": row_index,
            "original_source": source_name,
            "original_url": row.get("url"),
        },
        problem_text=question,
        interface_type="stdin_stdout",
        required_interface=_STDIO_INTERFACE,
        tests=[
            {"input": test_input, "output": expected_output}
            for test_input, expected_output in zip(inputs, outputs)
        ],
        metadata={
            "difficulty": row.get("difficulty"),
            "date": row.get("date"),
            "reference_solution_count": len(reference_solutions),
            "reference_solutions_sha256": hashlib.sha256(
                json.dumps(
                    reference_solutions,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "selection_scope": "pinned_train_shards_1_through_8_v1",
            "eligibility": {
                "interface_type": "stdin_stdout",
                "pictures_excluded": True,
                "starter_code_excluded": True,
                "shard_zero_excluded": True,
                "excluded_sources": sorted(TACO_MULTISHARD_EXCLUDED_SOURCES),
            },
        },
    )


def _parse_json_object(value: Any, *, context: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a JSON object string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} must be valid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{context} must decode to an object")
    return parsed


def _parse_json_string_list(value: Any, *, context: str) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a JSON list string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} must be valid JSON: {error.msg}") from error
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError(f"{context} must decode to a list of strings")
    return parsed


def _selection_rank(seed: int, task_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{task_id}".encode("utf-8")).digest(),
        "big",
    )


def _validate_selection(*, limit: int, selection: str, seed: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if selection not in {"first", "random"}:
        raise ValueError("selection must be 'first' or 'random'")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
