"""Durable streaming collection, normalization, and verification."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .api import GenerationConfig
from .code_verifier import (
    VerificationSummary,
    verify_normalized_record,
)
from .collector import CollectionSummary, TeacherClient, collect_records
from .durable_io import open_text_append, replace_file
from .normalize import (
    NORMALIZATION_ERROR_SCHEMA_VERSION,
    AppendNormalizeSummary,
)
from .records import (
    NORMALIZED_SCHEMA_VERSION,
    RAW_SCHEMA_VERSION,
    RecordValidationError,
    normalize_raw_record,
)


PIPELINE_STATE_SCHEMA_VERSION = "deepseek.durable.pipeline.state.v1"


@dataclass(frozen=True, slots=True)
class DurablePipelineSummary:
    collection: CollectionSummary
    normalization: AppendNormalizeSummary
    verification: VerificationSummary
    peak_verifier_in_flight: int


def run_durable_collection_pipeline(
    *,
    input_path: Path,
    raw_path: Path,
    normalized_path: Path,
    normalization_errors_path: Path,
    verifier_path: Path,
    state_path: Path,
    client: TeacherClient,
    config: GenerationConfig,
    collection_workers: int,
    verification_workers: int,
    requests_per_minute: float,
    provider: Mapping[str, Any] | None,
    phase_timeout_seconds: float,
    max_output_characters: int,
    poll_interval_seconds: float = 0.02,
    state_interval_seconds: float = 1.0,
) -> DurablePipelineSummary:
    """Run three resumable stages against fsynced append-only JSONL queues."""
    if collection_workers <= 0:
        raise ValueError("collection_workers must be positive")
    if verification_workers <= 0:
        raise ValueError("verification_workers must be positive")
    if poll_interval_seconds <= 0 or state_interval_seconds <= 0:
        raise ValueError("pipeline intervals must be positive")

    raw_path = Path(raw_path)
    normalized_path = Path(normalized_path)
    normalization_errors_path = Path(normalization_errors_path)
    verifier_path = Path(verifier_path)
    state_path = Path(state_path)
    for path in (
        raw_path,
        normalized_path,
        normalization_errors_path,
        verifier_path,
        state_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.touch(exist_ok=True)

    expected_raw_ids = _load_ids(Path(input_path))
    preexisting_raw_lines = _count_complete_jsonl_lines(raw_path)
    if preexisting_raw_lines > len(expected_raw_ids):
        raise ValueError("raw queue contains more records than selected inputs")
    normalized_ids = _load_ids(
        normalized_path,
        expected_schema=NORMALIZED_SCHEMA_VERSION,
    )
    normalization_error_ids = _load_ids(
        normalization_errors_path,
        expected_schema=NORMALIZATION_ERROR_SCHEMA_VERSION,
    )
    verifier_ids = _load_ids(verifier_path)
    if normalized_ids & normalization_error_ids:
        raise ValueError("an id cannot be both normalized and a normalization error")
    initial_normalized_terminal = normalized_ids | normalization_error_ids
    initial_verifier_ids = set(verifier_ids)

    raw_reader = _JsonlTail(raw_path)
    normalized_reader = _JsonlTail(normalized_path)
    raw_ids: set[str] = set()
    successful_raw_ids: set[str] = set()
    api_error_ids: set[str] = set()
    normalized_reader_ids: set[str] = set()
    new_normalized = 0
    new_malformed = 0
    new_warnings = 0
    new_verifier_passed = 0
    new_verifier_failed = 0
    new_failure_counts: Counter[str] = Counter()
    peak_in_flight = 0
    normalization_burst = max(1, min(8, verification_workers // 2))
    in_flight: dict[Future[dict[str, Any]], str] = {}
    collection: CollectionSummary | None = (
        CollectionSummary(
            total=len(expected_raw_ids),
            skipped=len(expected_raw_ids),
            succeeded=0,
            failed=0,
        )
        if preexisting_raw_lines == len(expected_raw_ids)
        else None
    )
    collection_error: BaseException | None = None
    last_state_write = 0.0

    def publish_state(phase: str, error: BaseException | None = None) -> None:
        nonlocal last_state_write
        state = {
            "schema_version": PIPELINE_STATE_SCHEMA_VERSION,
            "phase": phase,
            "updated_at": datetime.now(UTC).isoformat(),
            "workers": {
                "collector": collection_workers,
                "verifier": verification_workers,
            },
            "queues": {
                "raw": len(raw_ids),
                "normalized": len(normalized_ids),
                "normalization_errors": len(normalization_error_ids),
                "verifier": len(verifier_ids),
                "raw_to_normalized_lag": len(
                    successful_raw_ids - normalized_ids - normalization_error_ids
                ),
                "normalized_to_verifier_lag": len(normalized_ids - verifier_ids),
            },
            "runtime": {
                "collector_done": collection is not None or collection_error is not None,
                "verifier_in_flight": len(in_flight),
                "peak_verifier_in_flight": peak_in_flight,
            },
            "error": (
                {"type": type(error).__name__, "message": str(error)}
                if error is not None
                else None
            ),
        }
        _publish_json_atomic(state_path, state)
        last_state_write = time.monotonic()

    publish_state("running")
    try:
        with (
            ThreadPoolExecutor(max_workers=1) as collector_executor,
            ThreadPoolExecutor(max_workers=verification_workers) as verifier_executor,
        ):
            collector_future = (
                collector_executor.submit(
                    collect_records,
                    input_path=Path(input_path),
                    output_path=raw_path,
                    client=client,
                    config=config,
                    max_workers=collection_workers,
                    requests_per_minute=requests_per_minute,
                    provider=provider,
                )
                if collection is None
                else None
            )
            while True:
                progressed = False

                for future in [item for item in in_flight if item.done()]:
                    attempt_id = in_flight.pop(future)
                    result = future.result()
                    if result.get("id") != attempt_id:
                        raise ValueError("verifier result id does not match queued id")
                    if attempt_id in verifier_ids:
                        raise ValueError(f"duplicate verifier id {attempt_id!r}")
                    _append_jsonl(verifier_path, result)
                    verifier_ids.add(attempt_id)
                    category = result.get("failure_category")
                    if category == "passed":
                        new_verifier_passed += 1
                    else:
                        new_verifier_failed += 1
                        new_failure_counts[str(category or "unknown")] += 1
                    progressed = True

                while len(in_flight) < verification_workers:
                    normalized = normalized_reader.read()
                    if normalized is None:
                        break
                    attempt_id = _record_id(
                        normalized_path,
                        normalized_reader.line_number,
                        normalized,
                    )
                    if attempt_id in normalized_reader_ids:
                        raise ValueError(f"duplicate normalized id {attempt_id!r}")
                    normalized_reader_ids.add(attempt_id)
                    if normalized.get("schema_version") != NORMALIZED_SCHEMA_VERSION:
                        raise ValueError(
                            f"{normalized_path}:{normalized_reader.line_number}: "
                            "invalid normalized schema_version"
                        )
                    if attempt_id not in verifier_ids:
                        in_flight[
                            verifier_executor.submit(
                                verify_normalized_record,
                                normalized,
                                phase_timeout_seconds=phase_timeout_seconds,
                                max_output_characters=max_output_characters,
                            )
                        ] = attempt_id
                        peak_in_flight = max(peak_in_flight, len(in_flight))
                    progressed = True

                for _ in range(normalization_burst):
                    raw = raw_reader.read()
                    if raw is None:
                        break
                    attempt_id = _validate_raw_record(
                        raw_path,
                        raw_reader.line_number,
                        raw,
                    )
                    if attempt_id in raw_ids:
                        raise ValueError(f"duplicate raw id {attempt_id!r}")
                    if attempt_id not in expected_raw_ids:
                        raise ValueError(f"raw queue contains unknown id {attempt_id!r}")
                    raw_ids.add(attempt_id)
                    if raw["status"] == "error":
                        api_error_ids.add(attempt_id)
                        progressed = True
                        continue
                    successful_raw_ids.add(attempt_id)
                    if attempt_id in normalized_ids or attempt_id in normalization_error_ids:
                        progressed = True
                        continue
                    try:
                        normalized = normalize_raw_record(raw)
                    except RecordValidationError as error:
                        failure = {
                            "schema_version": NORMALIZATION_ERROR_SCHEMA_VERSION,
                            "id": attempt_id,
                            "failure_category": "malformed_trace",
                            "raw_line_number": raw_reader.line_number,
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                        }
                        _append_jsonl(normalization_errors_path, failure)
                        normalization_error_ids.add(attempt_id)
                        new_malformed += 1
                    else:
                        _append_jsonl(normalized_path, normalized)
                        normalized_ids.add(attempt_id)
                        new_normalized += 1
                        new_warnings += len(normalized["validation"]["warnings"])
                    progressed = True

                if (
                    collector_future is not None
                    and collector_future.done()
                    and collection is None
                    and collection_error is None
                ):
                    try:
                        collection = collector_future.result()
                    except BaseException as error:  # drain durable queues before surfacing it
                        collection_error = error

                collector_done = collection is not None or collection_error is not None
                if collector_done and raw_reader.has_partial_line:
                    raise ValueError(f"{raw_path}: incomplete trailing JSONL record")
                if collector_done and normalized_reader.has_partial_line:
                    raise ValueError(f"{normalized_path}: incomplete trailing JSONL record")
                if (
                    collector_done
                    and raw_reader.at_eof
                    and normalized_reader.at_eof
                    and not in_flight
                    and (
                        collection_error is not None
                        or expected_raw_ids <= raw_ids
                    )
                    and normalized_ids <= verifier_ids
                ):
                    break

                now = time.monotonic()
                if now - last_state_write >= state_interval_seconds:
                    publish_state("running")
                if not progressed:
                    time.sleep(poll_interval_seconds)

        if collection_error is not None:
            raise collection_error
        assert collection is not None
        if raw_ids != expected_raw_ids:
            raise ValueError("raw queue does not contain exactly the selected input ids")
        if not normalized_ids <= successful_raw_ids:
            raise ValueError("normalized queue contains an id without successful raw input")
        if not normalization_error_ids <= successful_raw_ids:
            raise ValueError("normalization error queue contains an id without raw input")
        if not verifier_ids <= normalized_ids | normalization_error_ids:
            raise ValueError("verifier queue contains an id without normalized input")

        publish_state("completed")
        normalization = AppendNormalizeSummary(
            total=len(raw_ids),
            skipped=len(successful_raw_ids & initial_normalized_terminal),
            normalized=new_normalized,
            api_errors=len(api_error_ids),
            malformed=new_malformed,
            warnings=new_warnings,
        )
        verification = VerificationSummary(
            total=len(normalized_ids),
            skipped=len(normalized_ids & initial_verifier_ids),
            passed=new_verifier_passed,
            failed=new_verifier_failed,
            failure_counts=dict(sorted(new_failure_counts.items())),
        )
        return DurablePipelineSummary(
            collection=collection,
            normalization=normalization,
            verification=verification,
            peak_verifier_in_flight=peak_in_flight,
        )
    except BaseException as error:
        publish_state("failed", error)
        raise


class _JsonlTail:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle = None
        self.line_number = 0
        self.at_eof = True
        self.has_partial_line = False

    def read(self) -> dict[str, Any] | None:
        if self._handle is None:
            if not self.path.exists():
                self.at_eof = True
                return None
            self._handle = self.path.open("r", encoding="utf-8", newline="")
        position = self._handle.tell()
        line = self._handle.readline()
        if not line:
            self.at_eof = True
            self.has_partial_line = False
            return None
        if not line.endswith("\n"):
            self._handle.seek(position)
            self.at_eof = False
            self.has_partial_line = True
            return None
        self.line_number += 1
        self.at_eof = False
        self.has_partial_line = False
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{self.path}:{self.line_number}: invalid JSON: {error.msg}"
            ) from error
        if not isinstance(value, Mapping):
            raise ValueError(f"{self.path}:{self.line_number}: expected a JSON object")
        return dict(value)


def _validate_raw_record(path: Path, line_number: int, record: Mapping[str, Any]) -> str:
    attempt_id = _record_id(path, line_number, record)
    if record.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError(f"{path}:{line_number}: invalid raw schema_version")
    status = record.get("status")
    if status not in {"ok", "error"}:
        raise ValueError(f"{path}:{line_number}: raw status must be 'ok' or 'error'")
    if status == "error":
        request = record.get("request")
        error = record.get("error")
        if not isinstance(request, Mapping) or not isinstance(error, Mapping):
            raise ValueError(
                f"{path}:{line_number}: raw API error metadata is malformed"
            )
        if not isinstance(request.get("messages"), list) or not request["messages"]:
            raise ValueError(
                f"{path}:{line_number}: raw API error request messages are missing"
            )
        if not isinstance(request.get("generation_config"), Mapping):
            raise ValueError(
                f"{path}:{line_number}: raw API error generation config is missing"
            )
        if not isinstance(error.get("type"), str) or not isinstance(
            error.get("message"), str
        ):
            raise ValueError(
                f"{path}:{line_number}: raw API error details are malformed"
            )
    return attempt_id


def _record_id(path: Path, line_number: int, record: Mapping[str, Any]) -> str:
    attempt_id = record.get("id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
    return attempt_id


def _load_ids(path: Path, *, expected_schema: str | None = None) -> set[str]:
    if not path.exists():
        return set()
    identifiers: set[str] = set()
    reader = _JsonlTail(path)
    while (record := reader.read()) is not None:
        attempt_id = _record_id(path, reader.line_number, record)
        if attempt_id in identifiers:
            raise ValueError(f"{path}:{reader.line_number}: duplicate id {attempt_id!r}")
        if expected_schema is not None and record.get("schema_version") != expected_schema:
            raise ValueError(f"{path}:{reader.line_number}: invalid schema_version")
        identifiers.add(attempt_id)
    if reader.has_partial_line:
        raise ValueError(f"{path}: incomplete trailing JSONL record")
    return identifiers


def _count_complete_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    lines = 0
    last_byte = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            lines += block.count(b"\n")
            last_byte = block[-1:]
    if last_byte and last_byte != b"\n":
        raise ValueError(f"{path}: incomplete trailing JSONL record")
    return lines


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with open_text_append(path) as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _publish_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
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
