"""Safe, pinned TACO ingestion for the stdin/stdout collection pilot.

The official repository contains a Python loading script. This module does not
execute it: it downloads a pinned Arrow shard and parses JSON columns with
``json.loads``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from .teacher_prompt import (
    STDIN_STDOUT_INTERFACE,
    STDIN_STDOUT_SYSTEM_MESSAGE as SYSTEM_MESSAGE,
    build_clean_teacher_messages,
)


TACO_TASK_SCHEMA_VERSION = "coding.task.taco.v1"
TACO_DATASET_ID = "BAAI/TACO"
TACO_SPLIT = "train"
TACO_REVISION = "d593ed0a2becbbc952230bb89be09189bf1056dc"
TACO_TRAIN_SHARD = "train/data-00000-of-00009.arrow"
TACO_CARD = "https://huggingface.co/datasets/BAAI/TACO"
TACO_PROVENANCE = "https://github.com/FlagOpen/TACO"
TACO_LICENSE = (
    "Dataset card/repository: Apache-2.0; upstream problem licenses vary, "
    "and the card notes unknown HackerRank rights"
)
TACO_PILOT_SEED = 20260728

def import_taco_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int = 100,
    selection: Literal["first", "random"] = "random",
    seed: int = TACO_PILOT_SEED,
    revision: str = TACO_REVISION,
    shard_path: str = TACO_TRAIN_SHARD,
    excluded_task_ids: Collection[str] = (),
    excluded_sources: Collection[str] = (),
    selection_scope: str = "single_pinned_train_shard_pilot",
) -> list[dict[str, Any]]:
    """Select pure stdin/stdout tasks from one pinned official train shard."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if selection not in {"first", "random"}:
        raise ValueError("selection must be 'first' or 'random'")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")
    if not isinstance(selection_scope, str) or not selection_scope.strip():
        raise ValueError("selection_scope must be a non-empty string")
    excluded_ids = _normalized_string_set(
        excluded_task_ids,
        label="excluded_task_ids",
        lowercase=False,
    )
    excluded_source_names = _normalized_string_set(
        excluded_sources,
        label="excluded_sources",
        lowercase=True,
    )

    eligible: list[tuple[int, Mapping[str, Any], dict[str, Any], list[str]]] = []
    for original_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"TACO row {original_index} must be an object")
        task_id = f"taco_train_{original_index:06d}"
        if task_id in excluded_ids:
            continue
        source_name = row.get("source")
        if (
            isinstance(source_name, str)
            and source_name.strip().lower() in excluded_source_names
        ):
            continue
        parsed_tests = _parse_json_object(
            row.get("input_output"),
            context=f"TACO row {original_index}.input_output",
        )
        reference_solutions = _parse_json_string_list(
            row.get("solutions"),
            context=f"TACO row {original_index}.solutions",
        )
        if _eligible_stdio_row(row, parsed_tests):
            eligible.append((original_index, row, parsed_tests, reference_solutions))

    if limit > len(eligible):
        raise ValueError(
            f"requested {limit} TACO tasks but only {len(eligible)} eligible "
            "stdin/stdout rows are available"
        )
    selected = (
        eligible[:limit]
        if selection == "first"
        else random.Random(seed).sample(eligible, limit)
    )
    return [
        _build_task(
            original_index,
            row,
            input_output=input_output,
            reference_solutions=reference_solutions,
            revision=revision,
            shard_path=shard_path,
            selection_scope=selection_scope,
            excluded_sources=excluded_source_names,
        )
        for original_index, row, input_output, reference_solutions in selected
    ]


def load_taco_tasks(
    *,
    limit: int = 100,
    selection: Literal["first", "random"] = "random",
    seed: int = TACO_PILOT_SEED,
    revision: str = TACO_REVISION,
    shard_path: str = TACO_TRAIN_SHARD,
    cache_dir: str | Path | None = None,
    excluded_task_ids: Collection[str] = (),
    excluded_sources: Collection[str] = (),
    selection_scope: str = "single_pinned_train_shard_pilot",
) -> list[dict[str, Any]]:
    """Download one pinned Arrow shard without executing remote Python code."""
    if cache_dir is not None:
        os.environ["HF_HOME"] = str(Path(cache_dir).resolve())
    try:
        from datasets import Dataset
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "TACO import requires the optional 'datasets' dependency"
        ) from error

    local_path = hf_hub_download(
        repo_id=TACO_DATASET_ID,
        repo_type="dataset",
        filename=shard_path,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    dataset = Dataset.from_file(local_path)
    return import_taco_rows(
        dataset,
        limit=limit,
        selection=selection,
        seed=seed,
        revision=revision,
        shard_path=shard_path,
        excluded_task_ids=excluded_task_ids,
        excluded_sources=excluded_sources,
        selection_scope=selection_scope,
    )


def build_teacher_messages(task: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build two messages without reading tests or reference solutions."""
    if not isinstance(task, Mapping):
        raise ValueError("task must be an object")
    task_id = task.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task.id must be a non-empty string")
    prompt_task_id = task.get("problem_id", task_id)
    if not isinstance(prompt_task_id, str) or not prompt_task_id.strip():
        raise ValueError("task.problem_id must be a non-empty string when provided")
    problem_text = task.get("problem_text")
    if not isinstance(problem_text, str) or not problem_text.strip():
        raise ValueError("teacher request requires an actual problem statement")
    if task.get("interface_type") != "stdin_stdout":
        raise ValueError("TACO pilot task interface_type must be 'stdin_stdout'")

    return build_clean_teacher_messages(
        task_id=prompt_task_id,
        problem_text=problem_text,
        required_interface=(
            "Complete Python program using standard input and standard output."
        ),
        interface_type=STDIN_STDOUT_INTERFACE,
    )


def _eligible_stdio_row(row: Mapping[str, Any], input_output: Mapping[str, Any]) -> bool:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        return False
    if input_output.get("fn_name") not in {None, ""}:
        return False
    inputs = input_output.get("inputs")
    outputs = input_output.get("outputs")
    if (
        not isinstance(inputs, list)
        or not inputs
        or not isinstance(outputs, list)
        or len(inputs) != len(outputs)
        or any(not isinstance(value, str) for value in [*inputs, *outputs])
    ):
        return False
    if row.get("picture_num") not in {None, 0, "0", ""}:
        return False
    source_name = row.get("source")
    if isinstance(source_name, str) and source_name.strip().lower() == "hackerrank":
        return False
    return True


def _build_task(
    original_index: int,
    row: Mapping[str, Any],
    *,
    input_output: Mapping[str, Any],
    reference_solutions: list[str],
    revision: str,
    shard_path: str,
    selection_scope: str,
    excluded_sources: set[str],
) -> dict[str, Any]:
    reference_digest = hashlib.sha256(
        json.dumps(reference_solutions, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    eligibility: dict[str, Any] = {
        "interface_type": "stdin_stdout",
        "pictures_excluded": True,
        "hackerrank_excluded": True,
    }
    if excluded_sources:
        eligibility["excluded_sources"] = sorted(excluded_sources)
    return {
        "schema_version": TACO_TASK_SCHEMA_VERSION,
        "id": f"taco_train_{original_index:06d}",
        "source": {
            "dataset": TACO_DATASET_ID,
            "split": TACO_SPLIT,
            "original_index": original_index,
            "revision": revision,
            "shard": shard_path,
            "license": TACO_LICENSE,
            "provenance": TACO_PROVENANCE,
            "mirror": TACO_CARD,
            "original_source": row.get("source"),
            "original_url": row.get("url"),
        },
        "problem_text": row["question"],
        "interface_type": "stdin_stdout",
        "tests": [
            {"input": test_input, "output": expected_output}
            for test_input, expected_output in zip(
                input_output["inputs"], input_output["outputs"]
            )
        ],
        "metadata": {
            "difficulty": row.get("difficulty"),
            "date": row.get("date"),
            "starter_code": row.get("starter_code") or "",
            "reference_solution_count": len(reference_solutions),
            "reference_solutions_sha256": reference_digest,
            "selection_scope": selection_scope,
            "eligibility": eligibility,
        },
    }


def _normalized_string_set(
    values: Collection[str],
    *,
    label: str,
    lowercase: bool,
) -> set[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise ValueError(f"{label} must be a collection of strings")
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must contain only non-empty strings")
        cleaned = value.strip()
        normalized.add(cleaned.lower() if lowercase else cleaned)
    return normalized


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
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a JSON list string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} must be valid JSON: {error.msg}") from error
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError(f"{context} must decode to a list of strings")
    return parsed
