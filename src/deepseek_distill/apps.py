"""Safe, revision-pinned APPS train ingestion for stdin/stdout tasks."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .multisource_tasks import (
    build_multisource_teacher_messages,
    make_multisource_task,
    select_unique_tasks,
)
from .hard_tasks import is_hard_task


APPS_DATASET_ID = "codeparrot/apps"
APPS_CONFIG = "all"
APPS_SPLIT = "train"
APPS_REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"
APPS_LICENSE = "MIT"
APPS_PROVENANCE = "https://github.com/hendrycks/apps"
APPS_MIRROR = "https://huggingface.co/datasets/codeparrot/apps"
APPS_TRAIN_FILE = "train.jsonl"
APPS_DEFAULT_SEED = 20260731
_STDIO_INTERFACE = "Complete Python program using standard input and standard output."

build_teacher_messages = build_multisource_teacher_messages


def import_apps_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    selection: str = "random",
    seed: int = APPS_DEFAULT_SEED,
    revision: str = APPS_REVISION,
    difficulty_profile: str | None = None,
) -> list[dict[str, Any]]:
    """Filter pure stdin/stdout rows and select a deterministic train subset."""

    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")
    eligible: list[dict[str, Any]] = []
    seen_original_ids: set[int] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"APPS row {position} must be an object")
        problem_id = _problem_id(row.get("problem_id", row.get("id")), position)
        if problem_id in seen_original_ids:
            raise ValueError(f"duplicate APPS problem_id {problem_id}")
        seen_original_ids.add(problem_id)
        if not _eligible_static_row(row):
            continue
        raw_tests = row.get("input_output")
        if raw_tests is None or raw_tests == "":
            continue
        parsed_tests = _parse_json_object(
            raw_tests,
            context=f"APPS row {position}.input_output",
        )
        if not _eligible_stdio_row(row, parsed_tests):
            continue
        reference_solutions = _parse_json_string_list(
            row.get("solutions"),
            context=f"APPS row {position}.solutions",
        )
        tests = [
            {"input": test_input, "output": expected_output}
            for test_input, expected_output in zip(
                parsed_tests["inputs"],
                parsed_tests["outputs"],
            )
        ]
        reference_digest = hashlib.sha256(
            json.dumps(
                reference_solutions,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        task = make_multisource_task(
                task_id=f"apps_train_{problem_id:06d}",
                source={
                    "dataset": APPS_DATASET_ID,
                    "config": APPS_CONFIG,
                    "split": APPS_SPLIT,
                    "original_id": problem_id,
                    "revision": revision,
                    "license": APPS_LICENSE,
                    "provenance": APPS_PROVENANCE,
                    "mirror": APPS_MIRROR,
                    "raw_file": APPS_TRAIN_FILE,
                },
                problem_text=row["question"],
                interface_type="stdin_stdout",
                required_interface=_STDIO_INTERFACE,
                tests=tests,
                metadata={
                    "difficulty": row.get("difficulty"),
                    "url": row.get("url"),
                    "starter_code": row.get("starter_code") or "",
                    "reference_solution_count": len(reference_solutions),
                    "reference_solutions_sha256": reference_digest,
                    "eligibility": {
                        "interface_type": "stdin_stdout",
                        "function_tasks_excluded": True,
                        "starter_code_required": False,
                    },
                },
            )
        if difficulty_profile is not None and not is_hard_task(
            "apps",
            task,
            profile=difficulty_profile,
        ):
            continue
        eligible.append(task)
    return select_unique_tasks(
        eligible,
        limit=limit,
        selection=selection,
        seed=seed,
    )


def load_apps_tasks(
    *,
    limit: int,
    selection: str = "random",
    seed: int = APPS_DEFAULT_SEED,
    revision: str = APPS_REVISION,
    cache_dir: str | Path | None = None,
    difficulty_profile: str | None = None,
) -> list[dict[str, Any]]:
    """Download the pinned raw train JSONL without executing its loader script."""

    if cache_dir is not None:
        os.environ["HF_HOME"] = str(Path(cache_dir).resolve())
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "APPS import requires the optional 'data' dependencies"
        ) from error

    local_path = hf_hub_download(
        repo_id=APPS_DATASET_ID,
        repo_type="dataset",
        filename=APPS_TRAIN_FILE,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )

    def rows() -> Iterable[dict[str, Any]]:
        with Path(local_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{local_path}:{line_number}: invalid JSON: {error.msg}"
                    ) from error
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{local_path}:{line_number}: row must be an object"
                    )
                yield value

    return import_apps_rows(
        rows(),
        limit=limit,
        selection=selection,
        seed=seed,
        revision=revision,
        difficulty_profile=difficulty_profile,
    )


def _eligible_stdio_row(
    row: Mapping[str, Any],
    tests: Mapping[str, Any],
) -> bool:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        return False
    if tests.get("fn_name") not in {None, ""}:
        return False
    inputs = tests.get("inputs")
    outputs = tests.get("outputs")
    if (
        not isinstance(inputs, list)
        or not inputs
        or not isinstance(outputs, list)
        or len(inputs) != len(outputs)
        or any(not isinstance(value, str) for value in [*inputs, *outputs])
    ):
        return False
    starter_code = row.get("starter_code")
    return starter_code in {None, ""}


def _eligible_static_row(row: Mapping[str, Any]) -> bool:
    question = row.get("question")
    starter_code = row.get("starter_code")
    return (
        isinstance(question, str)
        and bool(question.strip())
        and starter_code in {None, ""}
    )


def _problem_id(value: Any, position: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"APPS row {position}.problem_id must be a non-negative integer")
    return value


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
