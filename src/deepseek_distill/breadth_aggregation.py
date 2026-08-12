"""Memory-bounded aggregation for one-attempt breadth campaigns."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .durable_io import replace_file
from .mbpp import MBPP_TASK_SCHEMA_VERSION
from .multisource_tasks import (
    MULTISOURCE_TASK_SCHEMA_VERSION,
    multisource_dataset_slug,
)
from .rejection_sampling import make_attempt_id, parse_attempt_id, publish_json_once
from .taco import TACO_TASK_SCHEMA_VERSION


def aggregate_single_attempt_campaign(
    *,
    run_dir: Path,
    selected_tasks_path: Path,
    target: int,
) -> dict[str, Any]:
    """Build accepted/rejected artifacts while holding only indexes in memory."""
    run_dir = Path(run_dir)
    selected_tasks_path = Path(selected_tasks_path)
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError("target must be a positive integer")

    selected_ids, dataset_slug = _selected_campaign(selected_tasks_path)
    selected_index = {
        problem_id: selection_index
        for selection_index, problem_id in enumerate(selected_ids)
    }
    expected_attempt_ids = {
        make_attempt_id(problem_id, 1) for problem_id in selected_ids
    }

    raw_status: dict[str, str] = {}
    raw_path = run_dir / "raw_attempts.jsonl"
    for record in _read_jsonl_stream(raw_path):
        attempt_id = _record_id(record, label="raw")
        _validate_attempt_id(attempt_id, expected_attempt_ids)
        if attempt_id in raw_status:
            raise ValueError(f"duplicate raw attempt id {attempt_id!r}")
        status = record.get("status")
        if status not in {"ok", "error"}:
            raise ValueError(f"raw attempt {attempt_id!r} has invalid status")
        raw_status[attempt_id] = status

    verifier_category: dict[str, str] = {}
    verifier_path = run_dir / "verifier_attempts.jsonl"
    for record in _read_jsonl_stream(verifier_path):
        attempt_id = _record_id(record, label="verifier")
        _validate_attempt_id(attempt_id, expected_attempt_ids)
        if attempt_id in verifier_category:
            raise ValueError(f"duplicate verifier attempt id {attempt_id!r}")
        category = record.get("failure_category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"verifier attempt {attempt_id!r} has no failure category")
        verifier_category[attempt_id] = category

    normalized_ids: set[str] = set()
    finish_reasons: Counter[str] = Counter()
    accepted_problem_ids: set[str] = set()

    def accepted_records() -> Iterator[dict[str, Any]]:
        normalized_path = run_dir / "normalized_attempts.jsonl"
        for record in _read_jsonl_stream(normalized_path):
            attempt_id = _record_id(record, label="normalized")
            _validate_attempt_id(attempt_id, expected_attempt_ids)
            if attempt_id in normalized_ids:
                raise ValueError(f"duplicate normalized attempt id {attempt_id!r}")
            normalized_ids.add(attempt_id)
            finish_reason = record.get("finish_reason")
            if isinstance(finish_reason, str):
                finish_reasons[finish_reason] += 1
            if verifier_category.get(attempt_id) != "passed":
                continue
            problem_id, _ = parse_attempt_id(attempt_id)
            accepted_problem_ids.add(problem_id)
            accepted = dict(record)
            accepted["coding_verification"] = {
                "id": attempt_id,
                "failure_category": "passed",
                "artifact": {
                    "path": "verifier_attempts.jsonl",
                    "id": attempt_id,
                },
            }
            accepted["sampling"] = {
                "problem_id": problem_id,
                "attempt_id": attempt_id,
                "attempt_number": 1,
                "selection_index": selected_index[problem_id],
                "selection": "single_blind_attempt_in_seeded_task_order",
            }
            yield accepted

    _publish_jsonl_stream_once(
        run_dir / "accepted_unique.jsonl",
        accepted_records(),
    )
    _validate_trace_indexes(
        raw_status=raw_status,
        normalized_ids=normalized_ids,
        verifier_category=verifier_category,
    )

    def rejected_attempts() -> Iterator[dict[str, Any]]:
        for problem_id in selected_ids:
            if problem_id in accepted_problem_ids:
                continue
            attempt_id = make_attempt_id(problem_id, 1)
            if attempt_id not in raw_status:
                continue
            yield _rejected_attempt_record(
                problem_id=problem_id,
                attempt_id=attempt_id,
                selection_index=selected_index[problem_id],
                failure_category=_attempt_category(
                    attempt_id,
                    raw_status=raw_status,
                    normalized_ids=normalized_ids,
                    verifier_category=verifier_category,
                ),
            )

    _publish_jsonl_stream_once(
        run_dir / "rejected_attempts.jsonl",
        rejected_attempts(),
    )

    def rejected_tasks() -> Iterator[dict[str, Any]]:
        for task in _read_jsonl_stream(selected_tasks_path):
            problem_id = _record_id(task, label="selected task")
            if problem_id in accepted_problem_ids:
                continue
            attempt_id = make_attempt_id(problem_id, 1)
            actual_attempts = [attempt_id] if attempt_id in raw_status else []
            attempt_records = (
                [
                    _rejected_attempt_record(
                        problem_id=problem_id,
                        attempt_id=attempt_id,
                        selection_index=selected_index[problem_id],
                        failure_category=_attempt_category(
                            attempt_id,
                            raw_status=raw_status,
                            normalized_ids=normalized_ids,
                            verifier_category=verifier_category,
                        ),
                    )
                ]
                if actual_attempts
                else []
            )
            yield {
                "schema_version": f"coding.rejected.task.{dataset_slug}.v1",
                "id": problem_id,
                "problem_id": problem_id,
                "selection_index": selected_index[problem_id],
                "task": task,
                "attempt_ids": actual_attempts,
                "campaign_complete": bool(actual_attempts),
                "attempts": attempt_records,
            }

    _publish_jsonl_stream_once(
        run_dir / "rejected_tasks.jsonl",
        rejected_tasks(),
    )

    def attempt_ledger() -> Iterator[dict[str, Any]]:
        for problem_id in selected_ids:
            attempt_id = make_attempt_id(problem_id, 1)
            requested = attempt_id in raw_status
            selected = problem_id in accepted_problem_ids
            yield {
                "schema_version": f"coding.attempt.ledger.{dataset_slug}.v1",
                "id": attempt_id,
                "problem_id": problem_id,
                "attempt_number": 1,
                "selection_index": selected_index[problem_id],
                "state": "requested" if requested else "pending",
                "outcome": (
                    _attempt_category(
                        attempt_id,
                        raw_status=raw_status,
                        normalized_ids=normalized_ids,
                        verifier_category=verifier_category,
                    )
                    if requested
                    else None
                ),
                "selected_for_training": selected,
            }

    _publish_jsonl_stream_once(
        run_dir / "attempt_ledger.jsonl",
        attempt_ledger(),
    )

    accepted_count = len(accepted_problem_ids)
    pending_count = len(selected_ids) - len(raw_status)
    dataset_summary = {
        "selected_tasks": len(selected_ids),
        "actual_attempts": len(raw_status),
        "accepted_unique": accepted_count,
        "rejected_tasks": len(selected_ids) - accepted_count,
        "rejected_attempts": len(raw_status) - accepted_count,
        "target": target,
        "accepted_for_target": min(accepted_count, target),
        "target_met": accepted_count >= target,
        "shortfall": max(0, target - accepted_count),
        "pending_attempt_slots": pending_count,
        "unused_attempts_after_pass": 0,
        "duplicate_problem_ids": 0,
        "duplicate_attempt_ids": 0,
    }
    publish_json_once(
        run_dir / "dataset_summary.json",
        dataset_summary,
    )
    failures = Counter(
        _attempt_category(
            attempt_id,
            raw_status=raw_status,
            normalized_ids=normalized_ids,
            verifier_category=verifier_category,
        )
        for attempt_id in raw_status
        if verifier_category.get(attempt_id) != "passed"
    )
    summary = {
        "schema_version": (
            "coding.collection.taco.breadth.summary.v2"
            if dataset_slug == "taco"
            else f"coding.collection.{dataset_slug}.breadth.summary.v1"
        ),
        "dataset_slug": dataset_slug,
        "collection_complete": pending_count == 0,
        "counts": {
            "selected_tasks": len(selected_ids),
            "raw_attempts": len(raw_status),
            "normalized_attempts": len(normalized_ids),
            "verifier_results": len(verifier_category),
            "accepted_unique": accepted_count,
        },
        "dataset": dataset_summary,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "failure_categories": dict(sorted(failures.items())),
        "outputs": {
            "accepted_unique": "accepted_unique.jsonl",
            "attempt_ledger": "attempt_ledger.jsonl",
            "dataset_summary": "dataset_summary.json",
            "rejected_attempts": "rejected_attempts.jsonl",
            "rejected_tasks": "rejected_tasks.jsonl",
        },
    }
    publish_json_once(
        run_dir / "breadth_summary.json",
        summary,
    )
    return summary


def aggregate_attempt_campaign(
    *,
    run_dir: Path,
    selected_tasks_path: Path,
    target: int,
    max_attempts_per_task: int,
) -> dict[str, Any]:
    """Aggregate one-to-three attempt campaigns using offsets, not trace payloads."""

    if max_attempts_per_task == 1:
        return aggregate_single_attempt_campaign(
            run_dir=run_dir,
            selected_tasks_path=selected_tasks_path,
            target=target,
        )
    if max_attempts_per_task not in {2, 3}:
        raise ValueError("max_attempts_per_task must be 1, 2, or 3")
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError("target must be a positive integer")

    run_dir = Path(run_dir)
    selected_tasks_path = Path(selected_tasks_path)
    selected_ids, dataset_slug = _selected_campaign(selected_tasks_path)
    selected_index = {
        problem_id: selection_index
        for selection_index, problem_id in enumerate(selected_ids)
    }
    expected_attempt_ids = {
        make_attempt_id(problem_id, attempt_number)
        for problem_id in selected_ids
        for attempt_number in range(1, max_attempts_per_task + 1)
    }

    raw_status: dict[str, str] = {}
    raw_path = run_dir / "raw_attempts.jsonl"
    for record in _read_jsonl_stream(raw_path):
        attempt_id = _record_id(record, label="raw")
        _validate_multi_attempt_id(attempt_id, expected_attempt_ids)
        if attempt_id in raw_status:
            raise ValueError(f"duplicate raw attempt id {attempt_id!r}")
        status = record.get("status")
        if status not in {"ok", "error"}:
            raise ValueError(f"raw attempt {attempt_id!r} has invalid status")
        raw_status[attempt_id] = status

    verifier_category: dict[str, str] = {}
    verifier_path = run_dir / "verifier_attempts.jsonl"
    for record in _read_jsonl_stream(verifier_path):
        attempt_id = _record_id(record, label="verifier")
        _validate_multi_attempt_id(attempt_id, expected_attempt_ids)
        if attempt_id in verifier_category:
            raise ValueError(f"duplicate verifier attempt id {attempt_id!r}")
        category = record.get("failure_category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"verifier attempt {attempt_id!r} has no failure category")
        verifier_category[attempt_id] = category

    earliest_pass: dict[str, int] = {}
    accepted_by_attempt: Counter[int] = Counter()
    for problem_id in selected_ids:
        for attempt_number in range(1, max_attempts_per_task + 1):
            attempt_id = make_attempt_id(problem_id, attempt_number)
            if verifier_category.get(attempt_id) == "passed":
                earliest_pass[problem_id] = attempt_number
                accepted_by_attempt[attempt_number] += 1
                break

    normalized_path = run_dir / "normalized_attempts.jsonl"
    normalized_ids: set[str] = set()
    normalized_offsets: dict[str, int] = {}
    finish_reasons: Counter[str] = Counter()
    for offset, record in _read_jsonl_with_offsets(normalized_path):
        attempt_id = _record_id(record, label="normalized")
        _validate_multi_attempt_id(attempt_id, expected_attempt_ids)
        if attempt_id in normalized_ids:
            raise ValueError(f"duplicate normalized attempt id {attempt_id!r}")
        normalized_ids.add(attempt_id)
        problem_id, attempt_number = parse_attempt_id(attempt_id)
        if earliest_pass.get(problem_id) == attempt_number:
            normalized_offsets[attempt_id] = offset
        finish_reason = record.get("finish_reason")
        if isinstance(finish_reason, str):
            finish_reasons[finish_reason] += 1

    _validate_trace_indexes(
        raw_status=raw_status,
        normalized_ids=normalized_ids,
        verifier_category=verifier_category,
    )

    def accepted_records() -> Iterator[dict[str, Any]]:
        for problem_id in selected_ids:
            attempt_number = earliest_pass.get(problem_id)
            if attempt_number is None:
                continue
            attempt_id = make_attempt_id(problem_id, attempt_number)
            offset = normalized_offsets.get(attempt_id)
            if offset is None:
                raise ValueError(f"passing attempt {attempt_id!r} has no normalized trace")
            accepted = _read_jsonl_record_at(normalized_path, offset)
            accepted["coding_verification"] = {
                "id": attempt_id,
                "failure_category": "passed",
                "artifact": {"path": "verifier_attempts.jsonl", "id": attempt_id},
            }
            accepted["sampling"] = {
                "problem_id": problem_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "selection_index": selected_index[problem_id],
                "selection": "first_pass_in_seeded_task_order",
            }
            yield accepted

    _publish_jsonl_stream_once(run_dir / "accepted_unique.jsonl", accepted_records())

    def rejected_attempts() -> Iterator[dict[str, Any]]:
        for problem_id in selected_ids:
            chosen_number = earliest_pass.get(problem_id)
            for attempt_number in range(1, max_attempts_per_task + 1):
                attempt_id = make_attempt_id(problem_id, attempt_number)
                if attempt_id not in raw_status or attempt_number == chosen_number:
                    continue
                yield _rejected_attempt_record_multi(
                    dataset_slug=dataset_slug,
                    problem_id=problem_id,
                    attempt_id=attempt_id,
                    attempt_number=attempt_number,
                    selection_index=selected_index[problem_id],
                    failure_category=_attempt_category(
                        attempt_id,
                        raw_status=raw_status,
                        normalized_ids=normalized_ids,
                        verifier_category=verifier_category,
                    ),
                )

    _publish_jsonl_stream_once(run_dir / "rejected_attempts.jsonl", rejected_attempts())

    def rejected_tasks() -> Iterator[dict[str, Any]]:
        for task_record in _read_jsonl_stream(selected_tasks_path):
            problem_id = _record_id(task_record, label="selected task")
            if problem_id in earliest_pass:
                continue
            actual_ids = [
                make_attempt_id(problem_id, attempt_number)
                for attempt_number in range(1, max_attempts_per_task + 1)
                if make_attempt_id(problem_id, attempt_number) in raw_status
            ]
            attempts = [
                _rejected_attempt_record_multi(
                    dataset_slug=dataset_slug,
                    problem_id=problem_id,
                    attempt_id=attempt_id,
                    attempt_number=parse_attempt_id(attempt_id)[1],
                    selection_index=selected_index[problem_id],
                    failure_category=_attempt_category(
                        attempt_id,
                        raw_status=raw_status,
                        normalized_ids=normalized_ids,
                        verifier_category=verifier_category,
                    ),
                )
                for attempt_id in actual_ids
            ]
            yield {
                "schema_version": f"coding.rejected.task.{dataset_slug}.v1",
                "id": problem_id,
                "problem_id": problem_id,
                "selection_index": selected_index[problem_id],
                "task": task_record,
                "attempt_ids": actual_ids,
                "campaign_complete": len(actual_ids) == max_attempts_per_task,
                "attempts": attempts,
            }

    _publish_jsonl_stream_once(run_dir / "rejected_tasks.jsonl", rejected_tasks())

    pending_slots = 0
    not_requested_after_pass = 0

    def attempt_ledger() -> Iterator[dict[str, Any]]:
        nonlocal pending_slots, not_requested_after_pass
        for problem_id in selected_ids:
            chosen_number = earliest_pass.get(problem_id)
            for attempt_number in range(1, max_attempts_per_task + 1):
                attempt_id = make_attempt_id(problem_id, attempt_number)
                if attempt_id in raw_status:
                    state = "requested"
                    outcome: str | None = _attempt_category(
                        attempt_id,
                        raw_status=raw_status,
                        normalized_ids=normalized_ids,
                        verifier_category=verifier_category,
                    )
                elif chosen_number is not None and attempt_number > chosen_number:
                    state = "not_requested_after_pass"
                    outcome = None
                    not_requested_after_pass += 1
                else:
                    state = "pending"
                    outcome = None
                    pending_slots += 1
                yield {
                    "schema_version": f"coding.attempt.ledger.{dataset_slug}.v1",
                    "id": attempt_id,
                    "problem_id": problem_id,
                    "attempt_number": attempt_number,
                    "selection_index": selected_index[problem_id],
                    "state": state,
                    "outcome": outcome,
                    "selected_for_training": attempt_number == chosen_number,
                }

    _publish_jsonl_stream_once(run_dir / "attempt_ledger.jsonl", attempt_ledger())

    accepted_count = len(earliest_pass)
    dataset_summary = {
        "selected_tasks": len(selected_ids),
        "actual_attempts": len(raw_status),
        "accepted_unique": accepted_count,
        "rejected_tasks": len(selected_ids) - accepted_count,
        "rejected_attempts": len(raw_status) - accepted_count,
        "target": target,
        "accepted_for_target": min(accepted_count, target),
        "target_met": accepted_count >= target,
        "shortfall": max(0, target - accepted_count),
        "pending_attempt_slots": pending_slots,
        "unused_attempts_after_pass": 0,
        "not_requested_after_pass": not_requested_after_pass,
        "duplicate_problem_ids": 0,
        "duplicate_attempt_ids": 0,
        "max_attempts_per_task": max_attempts_per_task,
    }
    publish_json_once(run_dir / "dataset_summary.json", dataset_summary)
    failures = Counter(
        _attempt_category(
            attempt_id,
            raw_status=raw_status,
            normalized_ids=normalized_ids,
            verifier_category=verifier_category,
        )
        for attempt_id in raw_status
        if verifier_category.get(attempt_id) != "passed"
    )
    summary = {
        "schema_version": f"coding.collection.{dataset_slug}.rejection.summary.v1",
        "dataset_slug": dataset_slug,
        "collection_complete": pending_slots == 0,
        "counts": {
            "selected_tasks": len(selected_ids),
            "raw_attempts": len(raw_status),
            "normalized_attempts": len(normalized_ids),
            "verifier_results": len(verifier_category),
            "accepted_unique": accepted_count,
            "accepted_by_attempt": {
                str(number): accepted_by_attempt[number]
                for number in sorted(accepted_by_attempt)
            },
        },
        "dataset": dataset_summary,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "failure_categories": dict(sorted(failures.items())),
        "outputs": {
            "accepted_unique": "accepted_unique.jsonl",
            "attempt_ledger": "attempt_ledger.jsonl",
            "dataset_summary": "dataset_summary.json",
            "rejected_attempts": "rejected_attempts.jsonl",
            "rejected_tasks": "rejected_tasks.jsonl",
        },
    }
    publish_json_once(run_dir / "breadth_summary.json", summary)
    return summary


def _selected_campaign(path: Path) -> tuple[list[str], str]:
    selected: list[str] = []
    seen: set[str] = set()
    dataset_slugs: set[str] = set()
    for task in _read_jsonl_stream(path):
        task_id = _record_id(task, label="selected task")
        if task_id in seen:
            raise ValueError(f"duplicate selected task id {task_id!r}")
        make_attempt_id(task_id, 1)
        seen.add(task_id)
        selected.append(task_id)
        schema = task.get("schema_version")
        if schema == MBPP_TASK_SCHEMA_VERSION:
            dataset_slugs.add("mbpp")
        elif schema == TACO_TASK_SCHEMA_VERSION:
            dataset_slugs.add("taco")
        elif schema == MULTISOURCE_TASK_SCHEMA_VERSION:
            dataset_slugs.add(multisource_dataset_slug(task))
        else:
            raise ValueError(
                f"selected task {task_id!r} has an unsupported schema"
            )
    if not selected:
        raise ValueError("selected task file must not be empty")
    if len(dataset_slugs) != 1:
        raise ValueError("one breadth campaign cannot mix benchmark datasets")
    return selected, next(iter(dataset_slugs))


def _validate_attempt_id(attempt_id: str, expected: set[str]) -> None:
    _, attempt_number = parse_attempt_id(attempt_id)
    if attempt_number != 1 or attempt_id not in expected:
        raise ValueError(f"unexpected breadth attempt id {attempt_id!r}")


def _validate_multi_attempt_id(attempt_id: str, expected: set[str]) -> None:
    parse_attempt_id(attempt_id)
    if attempt_id not in expected:
        raise ValueError(f"unexpected campaign attempt id {attempt_id!r}")


def _validate_trace_indexes(
    *,
    raw_status: Mapping[str, str],
    normalized_ids: set[str],
    verifier_category: Mapping[str, str],
) -> None:
    for attempt_id in normalized_ids:
        if raw_status.get(attempt_id) != "ok":
            raise ValueError(
                f"normalized attempt {attempt_id!r} has no successful raw record"
            )
    for attempt_id, category in verifier_category.items():
        if raw_status.get(attempt_id) != "ok":
            raise ValueError(
                f"verifier attempt {attempt_id!r} has no successful raw record"
            )
        if attempt_id not in normalized_ids and category != "malformed_trace":
            raise ValueError(
                f"verifier attempt {attempt_id!r} has no normalized record"
            )
    for attempt_id, category in verifier_category.items():
        if category == "passed" and attempt_id not in normalized_ids:
            raise ValueError(f"passing attempt {attempt_id!r} has no normalized trace")


def _attempt_category(
    attempt_id: str,
    *,
    raw_status: Mapping[str, str],
    normalized_ids: set[str],
    verifier_category: Mapping[str, str],
) -> str:
    status = raw_status.get(attempt_id)
    if status == "error":
        return "api_error"
    category = verifier_category.get(attempt_id)
    if category is not None:
        return category
    if status == "ok" and attempt_id not in normalized_ids:
        return "malformed_trace"
    return "verification_missing"


def _rejected_attempt_record(
    *,
    problem_id: str,
    attempt_id: str,
    selection_index: int,
    failure_category: str,
) -> dict[str, Any]:
    return {
        "schema_version": "coding.rejected.attempt.taco.v1",
        "id": attempt_id,
        "problem_id": problem_id,
        "attempt_number": 1,
        "selection_index": selection_index,
        "failure_category": failure_category,
        "attempt_outcome": failure_category,
        "artifacts": {
            "raw": {"path": "raw_attempts.jsonl", "id": attempt_id},
            "normalized": {
                "path": "normalized_attempts.jsonl",
                "id": attempt_id,
            },
            "verifier": {
                "path": "verifier_attempts.jsonl",
                "id": attempt_id,
            },
        },
    }


def _rejected_attempt_record_multi(
    *,
    dataset_slug: str,
    problem_id: str,
    attempt_id: str,
    attempt_number: int,
    selection_index: int,
    failure_category: str,
) -> dict[str, Any]:
    return {
        "schema_version": f"coding.rejected.attempt.{dataset_slug}.v1",
        "id": attempt_id,
        "problem_id": problem_id,
        "attempt_number": attempt_number,
        "selection_index": selection_index,
        "failure_category": failure_category,
        "attempt_outcome": failure_category,
        "artifacts": {
            "raw": {"path": "raw_attempts.jsonl", "id": attempt_id},
            "normalized": {"path": "normalized_attempts.jsonl", "id": attempt_id},
            "verifier": {"path": "verifier_attempts.jsonl", "id": attempt_id},
        },
    }


def _record_id(record: Mapping[str, Any], *, label: str) -> str:
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"{label} record has no valid id")
    return record_id


def _read_jsonl_stream(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
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
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def _read_jsonl_with_offsets(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if not path.exists():
        return
    with path.open("rb") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield offset, value


def _read_jsonl_record_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline()
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}@{offset}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}@{offset}: expected JSON object")
    return value


def _publish_jsonl_stream_once(
    path: Path,
    records: Iterable[Mapping[str, Any]],
) -> str:
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as existing:
            for record in records:
                expected = _serialize_jsonl(record)
                actual = existing.readline()
                if actual != expected:
                    raise FileExistsError(
                        f"{path} already exists with different content"
                    )
            if existing.read():
                raise FileExistsError(f"{path} already contains extra records")
        return "unchanged"

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(_serialize_jsonl(record))
            handle.flush()
            os.fsync(handle.fileno())
        replace_file(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return "created"


def _serialize_jsonl(record: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(record),
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"
