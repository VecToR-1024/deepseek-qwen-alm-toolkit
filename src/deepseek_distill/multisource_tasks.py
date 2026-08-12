"""Common task contract for externally sourced coding benchmarks."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from .teacher_prompt import (
    FUNCTION_INTERFACE,
    STDIN_STDOUT_INTERFACE,
    build_clean_teacher_messages,
)


MULTISOURCE_TASK_SCHEMA_VERSION = "coding.task.multisource.v1"
MULTISOURCE_DATASET_SLUGS = {
    "BAAI/TACO": "taco",
    "codeparrot/apps": "apps",
    "deepmind/code_contests": "code_contests",
    "neulab/odex": "odex",
    "NTU-NLP-sg/xCodeEval": "xcodeeval",
    "open-r1/codeforces": "open_r1_codeforces",
}
_SOURCE_FIELDS = (
    "dataset",
    "split",
    "original_id",
    "revision",
    "license",
    "provenance",
    "mirror",
)


def make_multisource_task(
    *,
    task_id: str,
    source: Mapping[str, Any],
    problem_text: str,
    interface_type: str,
    required_interface: str,
    tests: Sequence[Any],
    metadata: Mapping[str, Any],
    function_name: str | None = None,
    function_signature: str | None = None,
) -> dict[str, Any]:
    """Validate and build a version-one common coding task."""

    identifier = _required_text(task_id, "task_id")
    problem = _required_text(problem_text, "problem_text")
    interface = _required_text(required_interface, "required_interface")
    source_copy = _validate_source(source)
    if interface_type not in {FUNCTION_INTERFACE, STDIN_STDOUT_INTERFACE}:
        raise ValueError("interface_type must be 'function' or 'stdin_stdout'")
    if isinstance(tests, (str, bytes)) or not isinstance(tests, Sequence) or not tests:
        raise ValueError("tests must be a non-empty sequence")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")

    task = {
        "schema_version": MULTISOURCE_TASK_SCHEMA_VERSION,
        "id": identifier,
        "source": source_copy,
        "problem_text": problem,
        "interface_type": interface_type,
        "required_interface": interface,
        "tests": copy.deepcopy(list(tests)),
        "metadata": copy.deepcopy(dict(metadata)),
    }
    if interface_type == FUNCTION_INTERFACE:
        task["function_name"] = _required_text(function_name, "function_name")
        signature = function_signature
        if signature is not None:
            task["function_signature"] = _required_text(
                signature,
                "function_signature",
            )
        else:
            task["function_signature"] = None
    elif function_name is not None or function_signature is not None:
        raise ValueError("stdin_stdout tasks cannot define a function interface")
    return task


def build_multisource_teacher_messages(
    task: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Build clean-v2 messages from the common task without reading tests."""

    if not isinstance(task, Mapping):
        raise ValueError("task must be an object")
    if task.get("schema_version") != MULTISOURCE_TASK_SCHEMA_VERSION:
        raise ValueError(
            f"task.schema_version must be {MULTISOURCE_TASK_SCHEMA_VERSION!r}"
        )
    task_id = task.get("problem_id", task.get("id"))
    return build_clean_teacher_messages(
        task_id=_required_text(task_id, "task.id"),
        problem_text=_required_text(
            task.get("problem_text"),
            "teacher request requires an actual problem statement",
            direct_error=True,
        ),
        required_interface=_required_text(
            task.get("required_interface"),
            "required_interface",
        ),
        interface_type=task.get("interface_type"),
    )


def multisource_dataset_slug(task: Mapping[str, Any]) -> str:
    """Return the frozen audit slug for one supported multi-source task."""

    if not isinstance(task, Mapping):
        raise ValueError("task must be an object")
    source = task.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("multi-source task has no source metadata")
    dataset = source.get("dataset")
    try:
        return MULTISOURCE_DATASET_SLUGS[dataset]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"unsupported multi-source dataset {dataset!r}"
        ) from error


def select_unique_tasks(
    tasks: Sequence[dict[str, Any]],
    *,
    limit: int,
    selection: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic ordered subset after validating unique IDs."""

    import random

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if selection not in {"first", "random"}:
        raise ValueError("selection must be 'first' or 'random'")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    identifiers = [task.get("id") for task in tasks]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("every task must have a non-empty string id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("task IDs must be unique before selection")
    if limit > len(tasks):
        raise ValueError(
            f"requested {limit} tasks but only {len(tasks)} eligible tasks are available"
        )
    selected = (
        list(tasks[:limit])
        if selection == "first"
        else random.Random(seed).sample(list(tasks), limit)
    )
    return copy.deepcopy(selected)


def _validate_source(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise ValueError("source must be an object")
    for field in _SOURCE_FIELDS:
        value = source.get(field)
        if field == "original_id":
            if (
                value is None
                or isinstance(value, bool)
                or (isinstance(value, str) and not value.strip())
            ):
                raise ValueError("source.original_id must be present")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"source.{field} must be a non-empty string")
    return copy.deepcopy(dict(source))


def _required_text(
    value: Any,
    label: str,
    *,
    direct_error: bool = False,
) -> str:
    if not isinstance(value, str) or not value.strip():
        if direct_error:
            raise ValueError(label)
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()
