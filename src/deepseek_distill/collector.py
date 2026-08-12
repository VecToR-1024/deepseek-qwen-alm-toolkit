"""Crash-resumable append-only collection of DeepSeek teacher responses."""

from __future__ import annotations

import json
import os
import copy
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .api import GenerationConfig, build_error_record, build_success_record
from .durable_io import open_text_append
from .mbpp import MBPP_TASK_SCHEMA_VERSION, build_teacher_messages
from .multisource_tasks import (
    MULTISOURCE_TASK_SCHEMA_VERSION,
    build_multisource_teacher_messages,
)
from .taco import (
    TACO_TASK_SCHEMA_VERSION,
    build_teacher_messages as build_taco_teacher_messages,
)
from .taco_retry import TACO_LENGTH_RETRY_SCHEMA_VERSION
from .teacher_prompt import (
    FUNCTION_INTERFACE,
    STDIN_STDOUT_INTERFACE,
    prompt_contract_metadata,
)


class TeacherClient(Protocol):
    def create_completion(self, messages, config: GenerationConfig) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    total: int
    skipped: int
    succeeded: int
    failed: int


class RateLimiter:
    """Thread-safe limiter that spaces request start times evenly."""

    def __init__(
        self,
        requests_per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_minute < 0:
            raise ValueError("requests_per_minute must be non-negative")
        self._interval = 0.0 if requests_per_minute == 0 else 60.0 / requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._next_start: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self._interval == 0:
            return
        with self._lock:
            now = self._clock()
            if self._next_start is None or now >= self._next_start:
                delay = 0.0
                self._next_start = now + self._interval
            else:
                delay = self._next_start - now
                self._next_start += self._interval
        if delay > 0:
            self._sleep(delay)


def collect_records(
    *,
    input_path: Path,
    output_path: Path,
    client: TeacherClient,
    config: GenerationConfig,
    max_workers: int = 1,
    requests_per_minute: float = 60,
    provider: Mapping[str, Any] | None = None,
    duration_clock: Callable[[], float] = time.perf_counter,
) -> CollectionSummary:
    """Collect every input record not already present in the output JSONL."""
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    inputs = _load_input_records(input_path)
    existing_ids = _load_existing_ids(output_path)
    pending = [record for record in inputs if record["id"] not in existing_ids]
    limiter = RateLimiter(requests_per_minute)

    def collect_one(record: dict[str, Any]) -> dict[str, Any]:
        limiter.wait()
        started_at = duration_clock()
        try:
            response = client.create_completion(record["messages"], config)
        except Exception as error:  # noqa: BLE001 - each provider failure must be persisted
            duration = max(0.0, duration_clock() - started_at)
            return build_error_record(
                record_id=record["id"],
                messages=record["messages"],
                config=config,
                error=error,
                task=record.get("task"),
                provider=provider,
                request_duration_seconds=duration,
                prompt_contract=record.get("prompt_contract"),
            )
        duration = max(0.0, duration_clock() - started_at)
        return build_success_record(
            record_id=record["id"],
            messages=record["messages"],
            config=config,
            response=response,
            task=record.get("task"),
            provider=provider,
            request_duration_seconds=duration,
            prompt_contract=record.get("prompt_contract"),
        )

    succeeded = 0
    failed = 0
    if max_workers == 1:
        results = (collect_one(record) for record in pending)
        for result in results:
            _append_jsonl(output_path, result)
            succeeded += result["status"] == "ok"
            failed += result["status"] == "error"
    else:
        for result in _bounded_parallel_results(
            collect_one,
            pending,
            max_workers=max_workers,
        ):
            _append_jsonl(output_path, result)
            succeeded += result["status"] == "ok"
            failed += result["status"] == "error"

    return CollectionSummary(
        total=len(inputs),
        skipped=len(inputs) - len(pending),
        succeeded=succeeded,
        failed=failed,
    )


def _bounded_parallel_results(
    function: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    max_workers: int,
) -> Iterator[Any]:
    """Yield results while retaining at most ``max_workers`` futures."""
    item_iterator = iter(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        active = set()
        for _ in range(max_workers):
            try:
                item = next(item_iterator)
            except StopIteration:
                break
            active.add(executor.submit(function, item))
        while active:
            completed, active = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                result = future.result()
                try:
                    item = next(item_iterator)
                except StopIteration:
                    pass
                else:
                    active.add(executor.submit(function, item))
                yield result


def _load_input_records(path: Path) -> list[dict[str, Any]]:
    records = _read_jsonl(path)
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for line_number, record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
        if record_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate input id {record_id!r}")
        if record.get("schema_version") == MBPP_TASK_SCHEMA_VERSION:
            messages = build_teacher_messages(record)
            task = copy.deepcopy(record)
            prompt_contract = prompt_contract_metadata(FUNCTION_INTERFACE)
        elif record.get("schema_version") in {
            TACO_TASK_SCHEMA_VERSION,
            TACO_LENGTH_RETRY_SCHEMA_VERSION,
        }:
            messages = build_taco_teacher_messages(record)
            task = copy.deepcopy(record)
            prompt_contract = prompt_contract_metadata(STDIN_STDOUT_INTERFACE)
        elif record.get("schema_version") == MULTISOURCE_TASK_SCHEMA_VERSION:
            messages = build_multisource_teacher_messages(record)
            task = copy.deepcopy(record)
            prompt_contract = prompt_contract_metadata(record.get("interface_type"))
        else:
            messages = record.get("messages")
            task = record.get("task")
            prompt_contract = None
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number}: messages must be a non-empty list")
            if task is not None and not isinstance(task, Mapping):
                raise ValueError(f"{path}:{line_number}: task must be an object or null")
        seen.add(record_id)
        normalized.append(
            {
                "id": record_id,
                "messages": copy.deepcopy(messages),
                "task": copy.deepcopy(task),
                "prompt_contract": copy.deepcopy(prompt_contract),
            }
        )
    return normalized


def _load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"{path}:{line_number}: each JSONL value must be an object"
                )
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(
                    f"{path}:{line_number}: existing record id must be a non-empty string"
                )
            if record_id in ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate existing id {record_id!r}"
                )
            ids.add(record_id)
    return ids


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: each JSONL value must be an object")
            records.append((line_number, dict(value)))
    return records


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with open_text_append(path) as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
