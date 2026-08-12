"""Contracts for blind, resumable coding-benchmark rejection sampling."""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .api import GenerationConfig
from .code_verifier import (
    TACO_VERIFIER_SCHEMA_VERSION,
    VERIFIER_SCHEMA_VERSION,
    VerificationSummary,
    verify_jsonl,
)
from .collector import CollectionSummary, TeacherClient, collect_records
from .durable_io import open_text_append, replace_file
from .mbpp import MBPP_CONFIG, MBPP_SPLIT, MBPP_TASK_SCHEMA_VERSION
from .multisource_tasks import (
    MULTISOURCE_TASK_SCHEMA_VERSION,
    multisource_dataset_slug,
)
from .normalize import AppendNormalizeSummary, normalize_jsonl_append
from .taco import TACO_TASK_SCHEMA_VERSION


_PROBLEM_ID = re.compile(
    r"^(?!.*__attempt_[1-3]$)[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$"
)
_ATTEMPT_ID = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._:-]{0,191})__attempt_([1-3])$"
)
_SUPPORTED_TASK_SCHEMAS = frozenset(
    {
        MBPP_TASK_SCHEMA_VERSION,
        TACO_TASK_SCHEMA_VERSION,
        MULTISOURCE_TASK_SCHEMA_VERSION,
    }
)


@dataclass(frozen=True, slots=True)
class RejectionSamplingWaveSummary:
    attempt_number: int
    planned: int
    task_file_status: str
    collection: CollectionSummary
    normalization: AppendNormalizeSummary
    verification: VerificationSummary


@dataclass(frozen=True, slots=True)
class RejectionSamplingRunSummary:
    selected_tasks: int
    raw_attempts: int
    normalized_attempts: int
    verifier_results: int
    waves: tuple[RejectionSamplingWaveSummary, ...]


@dataclass(frozen=True, slots=True)
class RawCollectionRunSummary:
    selected_tasks: int
    raw_attempts: int
    task_file_status: str
    collection: CollectionSummary


@dataclass(frozen=True, slots=True)
class RejectionSamplingDatasetSummary:
    selected_tasks: int
    actual_attempts: int
    accepted_unique: int
    rejected_tasks: int
    rejected_attempts: int
    target: int
    accepted_for_target: int
    target_met: bool
    shortfall: int
    pending_attempt_slots: int
    unused_attempts_after_pass: int
    duplicate_problem_ids: int
    duplicate_attempt_ids: int


@dataclass(frozen=True, slots=True)
class RejectionSamplingDatasets:
    accepted_unique: list[dict[str, Any]]
    accepted_first_target: list[dict[str, Any]]
    rejected_tasks: list[dict[str, Any]]
    rejected_attempts: list[dict[str, Any]]
    attempt_ledger: list[dict[str, Any]]
    summary: RejectionSamplingDatasetSummary


def make_attempt_id(problem_id: str, attempt_number: int) -> str:
    if not isinstance(problem_id, str) or _PROBLEM_ID.fullmatch(problem_id) is None:
        raise ValueError("problem_id must be a supported benchmark task id")
    if isinstance(attempt_number, bool) or attempt_number not in {1, 2, 3}:
        raise ValueError("attempt_number must be 1, 2, or 3")
    return f"{problem_id}__attempt_{attempt_number}"


def parse_attempt_id(attempt_id: str) -> tuple[str, int]:
    if not isinstance(attempt_id, str):
        raise ValueError("attempt_id must be a string")
    match = _ATTEMPT_ID.fullmatch(attempt_id)
    if match is None or _PROBLEM_ID.fullmatch(match.group(1)) is None:
        raise ValueError("attempt_id must contain a supported task id and attempt 1..3")
    return match.group(1), int(match.group(2))


def make_attempt_task(
    task: Mapping[str, Any],
    *,
    attempt_number: int,
    selection_index: int,
) -> dict[str, Any]:
    if (
        not isinstance(task, Mapping)
        or task.get("schema_version") not in _SUPPORTED_TASK_SCHEMAS
    ):
        raise ValueError("task schema_version is not supported for rejection sampling")
    problem_id = task.get("id")
    attempt_id = make_attempt_id(problem_id, attempt_number)
    if (
        isinstance(selection_index, bool)
        or not isinstance(selection_index, int)
        or selection_index < 0
    ):
        raise ValueError("selection_index must be a non-negative integer")
    attempt = copy.deepcopy(dict(task))
    attempt["id"] = attempt_id
    attempt["problem_id"] = problem_id
    attempt["attempt_number"] = attempt_number
    attempt["selection_index"] = selection_index
    return attempt


def validate_campaign_tasks(
    tasks: list[Mapping[str, Any]],
    *,
    expected_count: int,
    expected_revision: str,
    expected_schema_version: str = MBPP_TASK_SCHEMA_VERSION,
    expected_dataset: str = "MBPP",
    expected_config: str | None = MBPP_CONFIG,
    expected_split: str = MBPP_SPLIT,
) -> None:
    """Validate the immutable campaign boundary before any provider request."""
    if len(tasks) != expected_count:
        raise ValueError(f"expected {expected_count} selected tasks, found {len(tasks)}")
    seen_ids: set[str] = set()
    for index, task in enumerate(tasks):
        if task.get("schema_version") != expected_schema_version:
            raise ValueError(
                f"selected task {index} schema_version must be "
                f"{expected_schema_version!r}"
            )
        task_id = task.get("id")
        source = task.get("source")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"selected task {index} id must be a non-empty string")
        if task_id in seen_ids:
            raise ValueError(f"duplicate selected task id {task_id!r}")
        seen_ids.add(task_id)
        if not isinstance(source, Mapping):
            raise ValueError(f"selected task {task_id!r} source metadata is missing")
        if source.get("dataset") != expected_dataset or source.get("split") != expected_split:
            location = (
                f"{expected_config}/{expected_split}"
                if expected_config is not None
                else expected_split
            )
            raise ValueError(
                f"selected task {task_id!r} must come from "
                f"{expected_dataset} {location}"
            )
        if expected_config is not None and source.get("config") != expected_config:
            raise ValueError(
                f"selected task {task_id!r} must come from "
                f"{expected_dataset} {expected_config}/{expected_split}"
            )
        if source.get("revision") != expected_revision:
            raise ValueError(
                f"selected task {task_id!r} revision must be {expected_revision!r}"
            )
        if expected_schema_version == MBPP_TASK_SCHEMA_VERSION:
            original_id = source.get("original_id")
            expected_task_id = f"mbpp_{original_id}"
            source_field = "original_id"
        elif expected_schema_version == TACO_TASK_SCHEMA_VERSION:
            original_id = source.get("original_index")
            expected_task_id = (
                f"taco_train_{original_id:06d}"
                if isinstance(original_id, int) and not isinstance(original_id, bool)
                else None
            )
            source_field = "original_index"
        elif expected_schema_version == MULTISOURCE_TASK_SCHEMA_VERSION:
            multisource_dataset_slug(task)
            if _PROBLEM_ID.fullmatch(task_id) is None:
                raise ValueError(
                    f"selected task {task_id!r} is not a safe benchmark task id"
                )
            expected_task_id = task_id
            source_field = "source identity"
        else:
            raise ValueError(
                f"unsupported expected task schema {expected_schema_version!r}"
            )
        if task_id != expected_task_id:
            raise ValueError(
                f"selected task {task_id!r} does not match its {source_field}"
            )


def build_rejection_sampling_datasets(
    *,
    selected_tasks: list[Mapping[str, Any]],
    raw_records: list[Mapping[str, Any]],
    normalized_records: list[Mapping[str, Any]],
    verifier_records: list[Mapping[str, Any]],
    target: int = 200,
    embed_rejected_records: bool = True,
    max_attempts_per_task: int = 3,
) -> RejectionSamplingDatasets:
    """Select earliest passes mechanically and preserve every other attempt."""
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError("target must be a positive integer")
    attempt_numbers = _attempt_numbers(max_attempts_per_task)
    selected_ids: list[str] = []
    selected_by_id: dict[str, Mapping[str, Any]] = {}
    selected_schemas: set[str] = set()
    for index, task in enumerate(selected_tasks):
        problem_id = task.get("id")
        if not isinstance(problem_id, str):
            raise ValueError(f"selected task {index} has no valid id")
        if problem_id in selected_by_id:
            raise ValueError(f"duplicate selected task id {problem_id!r}")
        parseable = make_attempt_id(problem_id, 1)
        assert parseable
        selected_ids.append(problem_id)
        selected_by_id[problem_id] = task
        schema = task.get("schema_version")
        if schema not in _SUPPORTED_TASK_SCHEMAS:
            raise ValueError(f"selected task {problem_id!r} has unsupported schema")
        selected_schemas.add(schema)
    if len(selected_schemas) != 1:
        raise ValueError("selected tasks must all use the same benchmark schema")
    dataset_slug = _dataset_slug(selected_tasks, selected_schemas)

    raw_by_id = _index_attempt_records(raw_records, label="raw")
    normalized_by_id = _index_attempt_records(normalized_records, label="normalized")
    verifier_by_id = _index_attempt_records(verifier_records, label="verifier")
    allowed_ids = {
        make_attempt_id(problem_id, number)
        for problem_id in selected_ids
        for number in attempt_numbers
    }
    for label, indexed in (
        ("raw", raw_by_id),
        ("normalized", normalized_by_id),
        ("verifier", verifier_by_id),
    ):
        unknown = set(indexed) - allowed_ids
        if unknown:
            raise ValueError(f"{label} records contain unknown attempt ids: {sorted(unknown)!r}")
    for attempt_id in normalized_by_id:
        raw = raw_by_id.get(attempt_id)
        if raw is None or raw.get("status") != "ok":
            raise ValueError(f"normalized record {attempt_id!r} has no successful raw attempt")
    for attempt_id in verifier_by_id:
        verification = verifier_by_id[attempt_id]
        raw = raw_by_id.get(attempt_id)
        if raw is None or raw.get("status") != "ok":
            raise ValueError(f"verifier record {attempt_id!r} has no successful raw attempt")
        if (
            attempt_id not in normalized_by_id
            and verification.get("failure_category") != "malformed_trace"
        ):
            raise ValueError(f"verifier record {attempt_id!r} has no normalized trace")

    accepted_unique: list[dict[str, Any]] = []
    rejected_tasks: list[dict[str, Any]] = []
    rejected_attempts: list[dict[str, Any]] = []
    attempt_ledger: list[dict[str, Any]] = []
    pending_slots = 0
    unused_after_pass = 0
    for selection_index, problem_id in enumerate(selected_ids):
        actual_attempt_ids = [
            make_attempt_id(problem_id, attempt_number)
            for attempt_number in attempt_numbers
            if make_attempt_id(problem_id, attempt_number) in raw_by_id
        ]
        passing_numbers = [
            attempt_number
            for attempt_number in attempt_numbers
            if verifier_by_id.get(make_attempt_id(problem_id, attempt_number), {}).get(
                "failure_category"
            )
            == "passed"
        ]
        earliest_pass = min(passing_numbers) if passing_numbers else None
        chosen_attempt_id = (
            make_attempt_id(problem_id, earliest_pass)
            if earliest_pass is not None
            else None
        )
        if chosen_attempt_id is not None:
            normalized = copy.deepcopy(dict(normalized_by_id[chosen_attempt_id]))
            verification = copy.deepcopy(dict(verifier_by_id[chosen_attempt_id]))
            normalized["coding_verification"] = verification
            normalized["sampling"] = {
                "problem_id": problem_id,
                "attempt_id": chosen_attempt_id,
                "attempt_number": earliest_pass,
                "selection_index": selection_index,
                "selection": "first_pass_in_seeded_task_order",
            }
            accepted_unique.append(normalized)

        problem_rejections: list[dict[str, Any]] = []
        for attempt_number in attempt_numbers:
            attempt_id = make_attempt_id(problem_id, attempt_number)
            raw = raw_by_id.get(attempt_id)
            if raw is None:
                if earliest_pass is not None and attempt_number > earliest_pass:
                    state = "not_requested_after_pass"
                    outcome = None
                else:
                    state = "pending"
                    outcome = None
                    pending_slots += 1
            else:
                outcome = _attempt_outcome(
                    raw,
                    normalized_by_id.get(attempt_id),
                    verifier_by_id.get(attempt_id),
                )
                if earliest_pass is not None and attempt_number > earliest_pass:
                    state = "unused_after_pass"
                    unused_after_pass += 1
                else:
                    state = "requested"
            attempt_ledger.append(
                {
                    "schema_version": f"coding.attempt.ledger.{dataset_slug}.v1",
                    "id": attempt_id,
                    "problem_id": problem_id,
                    "attempt_number": attempt_number,
                    "selection_index": selection_index,
                    "state": state,
                    "outcome": outcome,
                    "selected_for_training": attempt_id == chosen_attempt_id,
                }
            )
            if raw is not None and attempt_id != chosen_attempt_id:
                rejection: dict[str, Any] = {
                    "schema_version": f"coding.rejected.attempt.{dataset_slug}.v1",
                    "id": attempt_id,
                    "problem_id": problem_id,
                    "attempt_number": attempt_number,
                    "selection_index": selection_index,
                    "failure_category": (
                        "unused_after_pass" if state == "unused_after_pass" else outcome
                    ),
                    "attempt_outcome": outcome,
                }
                if embed_rejected_records:
                    rejection.update(
                        {
                            "raw_record": copy.deepcopy(dict(raw)),
                            "normalized_record": (
                                copy.deepcopy(dict(normalized_by_id[attempt_id]))
                                if attempt_id in normalized_by_id
                                else None
                            ),
                            "verifier_result": (
                                copy.deepcopy(dict(verifier_by_id[attempt_id]))
                                if attempt_id in verifier_by_id
                                else None
                            ),
                        }
                    )
                else:
                    rejection["artifacts"] = {
                        "raw": {"path": "raw_attempts.jsonl", "id": attempt_id},
                        "normalized": (
                            {"path": "normalized_attempts.jsonl", "id": attempt_id}
                            if attempt_id in normalized_by_id
                            else None
                        ),
                        "verifier": (
                            {"path": "verifier_attempts.jsonl", "id": attempt_id}
                            if attempt_id in verifier_by_id
                            else None
                        ),
                    }
                rejected_attempts.append(rejection)
                problem_rejections.append(rejection)
        if earliest_pass is None:
            rejected_tasks.append(
                {
                    "schema_version": f"coding.rejected.task.{dataset_slug}.v1",
                    "id": problem_id,
                    "problem_id": problem_id,
                    "selection_index": selection_index,
                    "task": copy.deepcopy(dict(selected_by_id[problem_id])),
                    "attempt_ids": actual_attempt_ids,
                    "campaign_complete": (
                        len(actual_attempt_ids) == max_attempts_per_task
                    ),
                    "attempts": problem_rejections,
                }
            )

    accepted_first_target = copy.deepcopy(accepted_unique[:target])
    accepted_count = len(accepted_unique)
    summary = RejectionSamplingDatasetSummary(
        selected_tasks=len(selected_tasks),
        actual_attempts=len(raw_records),
        accepted_unique=accepted_count,
        rejected_tasks=len(rejected_tasks),
        rejected_attempts=len(rejected_attempts),
        target=target,
        accepted_for_target=len(accepted_first_target),
        target_met=accepted_count >= target,
        shortfall=max(0, target - accepted_count),
        pending_attempt_slots=pending_slots,
        unused_attempts_after_pass=unused_after_pass,
        duplicate_problem_ids=0,
        duplicate_attempt_ids=0,
    )
    return RejectionSamplingDatasets(
        accepted_unique=accepted_unique,
        accepted_first_target=accepted_first_target,
        rejected_tasks=rejected_tasks,
        rejected_attempts=rejected_attempts,
        attempt_ledger=attempt_ledger,
        summary=summary,
    )


def attempt_tasks_for_wave(
    selected_tasks: list[Mapping[str, Any]],
    *,
    raw_records: list[Mapping[str, Any]],
    verifier_records: list[Mapping[str, Any]],
    attempt_number: int,
) -> list[dict[str, Any]]:
    """Return the deterministic attempt wave for tasks not passed in earlier waves."""
    if attempt_number not in {1, 2, 3}:
        raise ValueError("attempt_number must be 1, 2, or 3")
    selected_ids: list[str] = []
    for index, task in enumerate(selected_tasks):
        task_id = task.get("id")
        if not isinstance(task_id, str):
            raise ValueError(f"selected task {index} has no valid id")
        if task_id in selected_ids:
            raise ValueError(f"duplicate selected task id {task_id!r}")
        selected_ids.append(task_id)

    raw_by_id = _index_attempt_records(raw_records, label="raw")
    verifier_by_id = _index_attempt_records(verifier_records, label="verifier")
    unknown_ids = (set(raw_by_id) | set(verifier_by_id)) - {
        make_attempt_id(problem_id, number)
        for problem_id in selected_ids
        for number in (1, 2, 3)
    }
    if unknown_ids:
        raise ValueError(f"attempt history contains unknown ids: {sorted(unknown_ids)!r}")

    for attempt_id, verification in verifier_by_id.items():
        raw = raw_by_id.get(attempt_id)
        if raw is None:
            raise ValueError(f"verifier record {attempt_id!r} has no raw record")
        if raw.get("status") != "ok":
            raise ValueError(f"verifier record {attempt_id!r} does not refer to raw success")
        task = verification.get("task")
        if isinstance(task, Mapping) and task.get("id") != attempt_id:
            raise ValueError(f"verifier task id does not match {attempt_id!r}")

    wave: list[dict[str, Any]] = []
    for selection_index, (problem_id, task) in enumerate(zip(selected_ids, selected_tasks)):
        passed_earlier = False
        for earlier_number in range(1, attempt_number):
            earlier_id = make_attempt_id(problem_id, earlier_number)
            raw = raw_by_id.get(earlier_id)
            if raw is None:
                raise ValueError(
                    f"{problem_id}: attempt {earlier_number} is missing before attempt "
                    f"{attempt_number}"
                )
            if raw.get("status") not in {"ok", "error"}:
                raise ValueError(f"raw attempt {earlier_id!r} has invalid status")
            verification = verifier_by_id.get(earlier_id)
            if raw.get("status") == "ok" and verification is None:
                raise ValueError(f"raw success {earlier_id!r} has no verifier result")
            if verification is not None and verification.get("failure_category") == "passed":
                passed_earlier = True
                break
        if not passed_earlier:
            wave.append(
                make_attempt_task(
                    task,
                    attempt_number=attempt_number,
                    selection_index=selection_index,
                )
            )
    return wave


def publish_jsonl_once(path: Path, records: list[Mapping[str, Any]]) -> str:
    """Create an immutable JSONL artifact, or verify an identical prior artifact."""
    path = Path(path)
    materialized = [copy.deepcopy(dict(record)) for record in records]
    if path.exists():
        existing = _read_jsonl(path)
        if existing != materialized:
            raise FileExistsError(f"{path} already exists with different content")
        return "unchanged"

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in materialized:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
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


def publish_json_once(path: Path, value: Mapping[str, Any]) -> str:
    """Create an immutable JSON object artifact, or verify exact equivalence."""
    path = Path(path)
    materialized = copy.deepcopy(dict(value))
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
        if existing != materialized:
            raise FileExistsError(f"{path} already exists with different content")
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(materialized, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
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


def write_rejection_sampling_outputs(
    run_dir: Path,
    datasets: RejectionSamplingDatasets,
) -> dict[str, str]:
    """Publish every deterministic aggregation artifact without overwriting."""
    run_dir = Path(run_dir)
    target = datasets.summary.target
    statuses = {
        "accepted_unique": publish_jsonl_once(
            run_dir / "accepted_unique.jsonl", datasets.accepted_unique
        ),
        "accepted_first_target": publish_jsonl_once(
            run_dir / f"accepted_first_{target}.jsonl", datasets.accepted_first_target
        ),
        "rejected_tasks": publish_jsonl_once(
            run_dir / "rejected_tasks.jsonl", datasets.rejected_tasks
        ),
        "rejected_attempts": publish_jsonl_once(
            run_dir / "rejected_attempts.jsonl", datasets.rejected_attempts
        ),
        "attempt_ledger": publish_jsonl_once(
            run_dir / "attempt_ledger.jsonl", datasets.attempt_ledger
        ),
        "dataset_summary": publish_json_once(
            run_dir / "dataset_summary.json", asdict(datasets.summary)
        ),
    }
    return statuses


def run_rejection_sampling(
    *,
    selected_tasks_path: Path,
    run_dir: Path,
    client: TeacherClient,
    config: GenerationConfig,
    max_workers: int = 1,
    requests_per_minute: float = 60,
    provider: Mapping[str, Any] | None = None,
    phase_timeout_seconds: float = 5.0,
    max_output_characters: int = 65_536,
    verification_workers: int = 1,
    max_attempts_per_task: int = 3,
    streaming_pipeline: bool = False,
    progress: Callable[[RejectionSamplingWaveSummary], None] | None = None,
) -> RejectionSamplingRunSummary:
    """Run one to three append-only blind collection/normalize/verify waves."""
    if (
        isinstance(verification_workers, bool)
        or not isinstance(verification_workers, int)
        or verification_workers <= 0
    ):
        raise ValueError("verification_workers must be a positive integer")
    if not isinstance(streaming_pipeline, bool):
        raise ValueError("streaming_pipeline must be a boolean")
    selected_tasks_path = Path(selected_tasks_path)
    run_dir = Path(run_dir)
    tasks = _read_jsonl(selected_tasks_path)
    if not tasks:
        raise ValueError("selected task file must contain at least one task")
    attempt_numbers = _attempt_numbers(max_attempts_per_task)
    for index, task in enumerate(tasks):
        if task.get("schema_version") not in _SUPPORTED_TASK_SCHEMAS:
            raise ValueError(
                f"{selected_tasks_path}: task {index} schema_version is unsupported"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw_attempts.jsonl"
    normalized_path = run_dir / "normalized_attempts.jsonl"
    normalization_errors_path = run_dir / "normalization_errors.jsonl"
    verifier_path = run_dir / "verifier_attempts.jsonl"
    wave_summaries: list[RejectionSamplingWaveSummary] = []
    existing_wave_numbers = [
        attempt_number
        for attempt_number in attempt_numbers
        if (run_dir / f"attempt_{attempt_number}_tasks.jsonl").exists()
    ]
    resume_attempt = (
        max(existing_wave_numbers)
        if streaming_pipeline and any(number > 1 for number in existing_wave_numbers)
        else 1
    )
    for attempt_number in attempt_numbers:
        if attempt_number < resume_attempt:
            continue
        use_streaming_pipeline = streaming_pipeline and attempt_number == 1
        wave_path = run_dir / f"attempt_{attempt_number}_tasks.jsonl"
        if wave_path.exists():
            # A wave manifest is immutable and was published only after the prior
            # wave became terminal. Reuse it verbatim on crash recovery instead
            # of deriving a smaller manifest from partial current-wave output.
            wave_tasks = _read_jsonl(wave_path)
        elif use_streaming_pipeline:
            # The durable pipeline validates the complete first-wave queue itself.
            # Avoid reparsing multi-GB top-k traces before it can start consuming.
            raw_records = []
            verifier_records = []
            wave_tasks = attempt_tasks_for_wave(
                tasks,
                raw_records=raw_records,
                verifier_records=verifier_records,
                attempt_number=attempt_number,
            )
        else:
            raw_records = _read_jsonl_projection_if_exists(
                raw_path,
                fields=("id", "status"),
            )
            verifier_records = _read_jsonl_projection_if_exists(
                verifier_path,
                fields=("id", "failure_category"),
            )
            wave_tasks = attempt_tasks_for_wave(
                tasks,
                raw_records=raw_records,
                verifier_records=verifier_records,
                attempt_number=attempt_number,
            )
        task_file_status = publish_jsonl_once(wave_path, wave_tasks)
        if use_streaming_pipeline:
            from .durable_pipeline import run_durable_collection_pipeline

            pipeline = run_durable_collection_pipeline(
                input_path=wave_path,
                raw_path=raw_path,
                normalized_path=normalized_path,
                normalization_errors_path=normalization_errors_path,
                verifier_path=verifier_path,
                state_path=run_dir / "pipeline_state.json",
                client=client,
                config=config,
                collection_workers=max_workers,
                verification_workers=verification_workers,
                requests_per_minute=requests_per_minute,
                provider=provider,
                phase_timeout_seconds=phase_timeout_seconds,
                max_output_characters=max_output_characters,
            )
            collection = pipeline.collection
            normalization = pipeline.normalization
            verification = pipeline.verification
        else:
            collection = collect_records(
                input_path=wave_path,
                output_path=raw_path,
                client=client,
                config=config,
                max_workers=max_workers,
                requests_per_minute=requests_per_minute,
                provider=provider,
            )
            normalization = normalize_jsonl_append(
                raw_path,
                normalized_path,
                error_output_path=normalization_errors_path,
            )
        _append_normalization_failures_to_verifier(
            raw_path=raw_path,
            normalization_errors_path=normalization_errors_path,
            verifier_path=verifier_path,
        )
        if not use_streaming_pipeline:
            normalized_path.touch(exist_ok=True)
            verification = verify_jsonl(
                input_path=normalized_path,
                output_path=verifier_path,
                phase_timeout_seconds=phase_timeout_seconds,
                max_output_characters=max_output_characters,
                max_workers=verification_workers,
            )
        wave_summary = RejectionSamplingWaveSummary(
            attempt_number=attempt_number,
            planned=len(wave_tasks),
            task_file_status=task_file_status,
            collection=collection,
            normalization=normalization,
            verification=verification,
        )
        wave_summaries.append(wave_summary)
        if progress is not None:
            progress(wave_summary)

    return RejectionSamplingRunSummary(
        selected_tasks=len(tasks),
        raw_attempts=_count_jsonl_records(raw_path),
        normalized_attempts=_count_jsonl_records(normalized_path),
        verifier_results=_count_jsonl_records(verifier_path),
        waves=tuple(wave_summaries),
    )


def collect_rejection_sampling_raw(
    *,
    selected_tasks_path: Path,
    run_dir: Path,
    client: TeacherClient,
    config: GenerationConfig,
    max_workers: int = 1,
    requests_per_minute: float = 60,
    provider: Mapping[str, Any] | None = None,
) -> RawCollectionRunSummary:
    """Collect only the durable first-attempt raw queue for a later consumer."""
    selected_tasks_path = Path(selected_tasks_path)
    run_dir = Path(run_dir)
    tasks = _read_jsonl(selected_tasks_path)
    if not tasks:
        raise ValueError("selected task file must contain at least one task")
    for index, task in enumerate(tasks):
        if task.get("schema_version") not in _SUPPORTED_TASK_SCHEMAS:
            raise ValueError(
                f"{selected_tasks_path}: task {index} schema_version is unsupported"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw_attempts.jsonl"
    verifier_path = run_dir / "verifier_attempts.jsonl"
    wave_tasks = attempt_tasks_for_wave(
        tasks,
        raw_records=_read_jsonl_projection_if_exists(
            raw_path,
            fields=("id", "status"),
        ),
        verifier_records=_read_jsonl_projection_if_exists(
            verifier_path,
            fields=("id", "failure_category"),
        ),
        attempt_number=1,
    )
    wave_path = run_dir / "attempt_1_tasks.jsonl"
    task_file_status = publish_jsonl_once(wave_path, wave_tasks)
    collection = collect_records(
        input_path=wave_path,
        output_path=raw_path,
        client=client,
        config=config,
        max_workers=max_workers,
        requests_per_minute=requests_per_minute,
        provider=provider,
    )
    return RawCollectionRunSummary(
        selected_tasks=len(tasks),
        raw_attempts=_count_jsonl_records(raw_path),
        task_file_status=task_file_status,
        collection=collection,
    )


def _attempt_numbers(max_attempts_per_task: int) -> range:
    if (
        isinstance(max_attempts_per_task, bool)
        or not isinstance(max_attempts_per_task, int)
        or max_attempts_per_task not in {1, 2, 3}
    ):
        raise ValueError("max_attempts_per_task must be 1, 2, or 3")
    return range(1, max_attempts_per_task + 1)


def _dataset_slug(
    selected_tasks: list[Mapping[str, Any]],
    selected_schemas: set[str],
) -> str:
    if selected_schemas == {MBPP_TASK_SCHEMA_VERSION}:
        return "mbpp"
    if selected_schemas == {TACO_TASK_SCHEMA_VERSION}:
        return "taco"
    if selected_schemas != {MULTISOURCE_TASK_SCHEMA_VERSION}:
        raise ValueError("selected tasks use an unsupported benchmark schema")
    slugs = {multisource_dataset_slug(task) for task in selected_tasks}
    if len(slugs) != 1:
        raise ValueError(
            "one rejection-sampling campaign cannot mix multi-source datasets"
        )
    return next(iter(slugs))


def _index_attempt_records(
    records: list[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} record {position} must be an object")
        record_id = record.get("id")
        parse_attempt_id(record_id)
        if record_id in indexed:
            raise ValueError(f"duplicate {label} attempt id {record_id!r}")
        indexed[record_id] = record
    return indexed


def _attempt_outcome(
    raw: Mapping[str, Any],
    normalized: Mapping[str, Any] | None,
    verifier: Mapping[str, Any] | None,
) -> str:
    status = raw.get("status")
    if status == "error":
        return "api_error"
    if status != "ok" or normalized is None:
        return "malformed_trace"
    if verifier is None:
        return "verification_missing"
    category = verifier.get("failure_category")
    return category if isinstance(category, str) and category else "verification_missing"


def _append_normalization_failures_to_verifier(
    *,
    raw_path: Path,
    normalization_errors_path: Path,
    verifier_path: Path,
) -> int:
    if not normalization_errors_path.exists():
        return 0
    existing = {
        record["id"]
        for record in _read_jsonl_if_exists(verifier_path)
        if isinstance(record.get("id"), str)
    }
    failures = _read_jsonl(normalization_errors_path)
    pending_ids = {
        failure.get("id")
        for failure in failures
        if isinstance(failure.get("id"), str) and failure.get("id") not in existing
    }
    raw_by_id: dict[str, dict[str, Any]] = {}
    if pending_ids:
        for record in _iter_jsonl(raw_path):
            record_id = record.get("id")
            if record_id in pending_ids:
                raw_by_id[record_id] = record
                if raw_by_id.keys() >= pending_ids:
                    break
    appended = 0
    for failure in failures:
        attempt_id = failure.get("id")
        if attempt_id in existing:
            continue
        raw = raw_by_id.get(attempt_id)
        if raw is None:
            raise ValueError(f"normalization failure {attempt_id!r} has no raw record")
        result = {
            "schema_version": (
                TACO_VERIFIER_SCHEMA_VERSION
                if isinstance(raw.get("task"), Mapping)
                and (
                    raw["task"].get("schema_version") == TACO_TASK_SCHEMA_VERSION
                    or raw["task"].get("interface_type") == "stdin_stdout"
                )
                else VERIFIER_SCHEMA_VERSION
            ),
            "id": attempt_id,
            "status": "rejected",
            "failure_category": "malformed_trace",
            "task": copy.deepcopy(raw.get("task")),
            "teacher_response": _raw_response_text(raw),
            "extracted_source": None,
            "source_extraction": None,
            "trace_validation": {
                "valid": False,
                "error": copy.deepcopy(failure.get("error")),
            },
            "phases": [],
            "error": (
                failure.get("error", {}).get("message")
                if isinstance(failure.get("error"), Mapping)
                else "normalization failed"
            ),
        }
        _append_jsonl_durable(verifier_path, result)
        existing.add(attempt_id)
        appended += 1
    return appended


def _raw_response_text(raw: Mapping[str, Any]) -> str | None:
    response = raw.get("response")
    if not isinstance(response, Mapping):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _append_jsonl_durable(path: Path, record: Mapping[str, Any]) -> None:
    with open_text_append(path) as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL value must be an object")
            yield value


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path) if path.exists() else []


def _read_jsonl_projection_if_exists(
    path: Path,
    *,
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    projected: list[dict[str, Any]] = []
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
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            projected.append({field: value.get(field) for field in fields})
    return projected


def _count_jsonl_records(path: Path) -> int:
    """Count durable JSONL records without retaining their multi-MB payloads."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)
