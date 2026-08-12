"""Versioned blind retries for TACO attempts truncated by max_tokens."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .taco import TACO_TASK_SCHEMA_VERSION


TACO_LENGTH_RETRY_SCHEMA_VERSION = "coding.task.taco.length_retry.v2"
TACO_LENGTH_RETRY_MAX_TOKENS = 8192
_SOURCE_ATTEMPT_ID = re.compile(r"^(taco_train_[0-9]{6})__attempt_([1-3])$")


def build_length_retry_tasks(
    *,
    selected_tasks: Sequence[Mapping[str, Any]],
    normalized_attempts: Sequence[Mapping[str, Any]],
    accepted_v1: Sequence[Mapping[str, Any]],
    max_tokens: int = TACO_LENGTH_RETRY_MAX_TOKENS,
) -> list[dict[str, Any]]:
    """Select length-truncated attempts in original task/attempt order.

    Problems already accepted by v1 are omitted. Retry records retain hidden
    tests for local verification, but prompt construction reads neither
    ``retry`` nor ``tests``.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    selected_by_id: dict[str, Mapping[str, Any]] = {}
    ordered_problem_ids: list[str] = []
    for position, task in enumerate(selected_tasks):
        if not isinstance(task, Mapping) or task.get("schema_version") != TACO_TASK_SCHEMA_VERSION:
            raise ValueError(f"selected task {position} must use the TACO v1 schema")
        problem_id = task.get("id")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError(f"selected task {position} has no valid id")
        if problem_id in selected_by_id:
            raise ValueError(f"duplicate selected task id {problem_id!r}")
        selected_by_id[problem_id] = task
        ordered_problem_ids.append(problem_id)

    accepted_problem_ids = _accepted_problem_ids(accepted_v1)
    length_attempts: dict[tuple[str, int], str] = {}
    seen_attempt_ids: set[str] = set()
    for position, record in enumerate(normalized_attempts):
        if not isinstance(record, Mapping):
            raise ValueError(f"normalized attempt {position} must be an object")
        attempt_id = record.get("id")
        if not isinstance(attempt_id, str) or attempt_id in seen_attempt_ids:
            raise ValueError(f"normalized attempt {position} has an invalid or duplicate id")
        seen_attempt_ids.add(attempt_id)
        match = _SOURCE_ATTEMPT_ID.fullmatch(attempt_id)
        if match is None:
            raise ValueError(f"unsupported source attempt id {attempt_id!r}")
        problem_id, attempt_text = match.groups()
        if problem_id not in selected_by_id:
            raise ValueError(f"source attempt {attempt_id!r} is not in selected tasks")
        if record.get("finish_reason") == "length":
            length_attempts[(problem_id, int(attempt_text))] = attempt_id

    retries: list[dict[str, Any]] = []
    for problem_id in ordered_problem_ids:
        if problem_id in accepted_problem_ids:
            continue
        for attempt_number in (1, 2, 3):
            source_attempt_id = length_attempts.get((problem_id, attempt_number))
            if source_attempt_id is None:
                continue
            retry = copy.deepcopy(dict(selected_by_id[problem_id]))
            retry["schema_version"] = TACO_LENGTH_RETRY_SCHEMA_VERSION
            retry["id"] = f"{source_attempt_id}__length_retry_v2"
            retry["problem_id"] = problem_id
            retry["retry"] = {
                "policy_version": "taco.length_retry.v2",
                "source_attempt_id": source_attempt_id,
                "source_attempt_number": attempt_number,
                "source_finish_reason": "length",
                "max_tokens": max_tokens,
                "teacher_feedback": False,
            }
            retries.append(retry)
    return retries


def build_length_retry_datasets(
    *,
    selected_tasks: Sequence[Mapping[str, Any]],
    accepted_v1: Sequence[Mapping[str, Any]],
    retry_tasks: Sequence[Mapping[str, Any]],
    normalized_retries: Sequence[Mapping[str, Any]],
    verifier_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose the first passing retry per problem and combine in v1 task order."""
    retry_by_id = _unique_index(retry_tasks, label="retry task")
    normalized_by_id = _unique_index(normalized_retries, label="normalized retry")
    verifier_by_id = _unique_index(verifier_results, label="verifier result")
    if not set(normalized_by_id) <= set(retry_by_id):
        raise ValueError("normalized retries contain ids outside the retry task manifest")
    if not set(verifier_by_id) <= set(retry_by_id):
        raise ValueError("verifier results contain ids outside the retry task manifest")

    accepted_v1_by_problem = {
        _accepted_problem_id(record): copy.deepcopy(dict(record))
        for record in accepted_v1
    }
    newly_by_problem: dict[str, dict[str, Any]] = {}
    outcomes: list[dict[str, Any]] = []
    for retry_task in retry_tasks:
        retry_id = retry_task["id"]
        problem_id = retry_task["problem_id"]
        verification = verifier_by_id.get(retry_id)
        category = (
            verification.get("failure_category")
            if isinstance(verification, Mapping)
            else "verification_missing"
        )
        outcomes.append(
            {
                "schema_version": "coding.attempt.taco.length_retry.v2",
                "id": retry_id,
                "problem_id": problem_id,
                "source_attempt_id": retry_task["retry"]["source_attempt_id"],
                "failure_category": category,
                "selected_for_training": False,
            }
        )
        if (
            category != "passed"
            or problem_id in accepted_v1_by_problem
            or problem_id in newly_by_problem
        ):
            continue
        normalized = normalized_by_id.get(retry_id)
        if normalized is None:
            raise ValueError(f"passing retry {retry_id!r} has no normalized record")
        accepted = copy.deepcopy(dict(normalized))
        accepted["coding_verification"] = copy.deepcopy(dict(verification))
        accepted["sampling"] = {
            "problem_id": problem_id,
            "attempt_id": retry_id,
            "selection": "first_passing_length_retry_in_v1_task_order",
            "policy_version": "taco.length_retry.v2",
            "source_attempt_id": retry_task["retry"]["source_attempt_id"],
            "source_attempt_number": retry_task["retry"]["source_attempt_number"],
            "max_tokens": retry_task["retry"]["max_tokens"],
        }
        newly_by_problem[problem_id] = accepted
        outcomes[-1]["selected_for_training"] = True

    newly_accepted: list[dict[str, Any]] = []
    combined: list[dict[str, Any]] = []
    for task in selected_tasks:
        problem_id = task.get("id")
        if problem_id in accepted_v1_by_problem:
            combined.append(accepted_v1_by_problem[problem_id])
        elif problem_id in newly_by_problem:
            newly_accepted.append(newly_by_problem[problem_id])
            combined.append(newly_by_problem[problem_id])
    return {
        "newly_accepted_unique": newly_accepted,
        "combined_accepted_unique": combined,
        "retry_outcomes": outcomes,
        "summary": {
            "retry_attempts_planned": len(retry_tasks),
            "retry_attempts_normalized": len(normalized_retries),
            "retry_attempts_verified": len(verifier_results),
            "retry_attempt_passes": sum(
                outcome["failure_category"] == "passed" for outcome in outcomes
            ),
            "newly_accepted_tasks": len(newly_accepted),
            "accepted_v1_tasks": len(accepted_v1_by_problem),
            "combined_accepted_tasks": len(combined),
        },
    }


def select_first_retry_per_problem(
    retry_tasks: Sequence[Mapping[str, Any]],
    *,
    problem_limit: int,
) -> list[dict[str, Any]]:
    """Return the first canonical retry for each of the first N problems."""
    if (
        isinstance(problem_limit, bool)
        or not isinstance(problem_limit, int)
        or problem_limit <= 0
    ):
        raise ValueError("problem_limit must be a positive integer")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, retry in enumerate(retry_tasks):
        if not isinstance(retry, Mapping):
            raise ValueError(f"retry task {position} must be an object")
        problem_id = retry.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError(f"retry task {position} has no valid problem_id")
        if problem_id in seen:
            continue
        seen.add(problem_id)
        selected.append(copy.deepcopy(dict(retry)))
        if len(selected) == problem_limit:
            break
    if len(selected) < problem_limit:
        raise ValueError(
            f"requested {problem_limit} distinct problems but only "
            f"{len(selected)} are available"
        )
    return selected


def portable_manifest_path(path: Path) -> str:
    """Keep a user-supplied provenance path stable across local workspaces."""
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            pass
    return candidate.as_posix()


def _accepted_problem_ids(records: Sequence[Mapping[str, Any]]) -> set[str]:
    return {_accepted_problem_id(record) for record in records}


def _accepted_problem_id(record: Mapping[str, Any]) -> str:
    if not isinstance(record, Mapping):
        raise ValueError("accepted record must be an object")
    sampling = record.get("sampling")
    problem_id = sampling.get("problem_id") if isinstance(sampling, Mapping) else None
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError("accepted record sampling.problem_id is missing")
    return problem_id


def _unique_index(
    records: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} {position} must be an object")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{label} {position} has no valid id")
        if record_id in indexed:
            raise ValueError(f"duplicate {label} id {record_id!r}")
        indexed[record_id] = record
    return indexed
