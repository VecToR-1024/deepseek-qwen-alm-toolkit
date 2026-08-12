from __future__ import annotations

import errno
import json
import os
import threading
import time
from pathlib import Path

from deepseek_distill.api import GenerationConfig
from deepseek_distill.code_verifier import VERIFIER_SCHEMA_VERSION
from deepseek_distill.collector import collect_records
from deepseek_distill.durable_pipeline import (
    _publish_json_atomic,
    run_durable_collection_pipeline,
)
from deepseek_distill.rejection_sampling import make_attempt_task, run_rejection_sampling
from deepseek_distill.records import normalize_raw_record as real_normalize_raw_record
from tests.test_rejection_sampling import base_task, raw_response


def test_pipeline_state_publish_retries_a_transient_windows_file_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "pipeline_state.json"
    original_replace = os.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(errno.EACCES, "temporarily locked", str(destination))
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", flaky_replace)

    _publish_json_atomic(state_path, {"phase": "running"})

    assert attempts == 3
    assert json.loads(state_path.read_text(encoding="utf-8")) == {"phase": "running"}


def _write_tasks(path: Path, count: int) -> None:
    tasks = []
    for offset in range(count):
        original_id = 700 + offset
        task = base_task() | {
            "id": f"mbpp_{original_id}",
            "source": {
                "dataset": "MBPP",
                "split": "train",
                "original_id": original_id,
            },
        }
        tasks.append(task)
    path.write_text(
        "".join(json.dumps(task) + "\n" for task in tasks),
        encoding="utf-8",
    )


class SlowTeacherClient:
    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self.calls = 0
        self.lock = threading.Lock()

    def create_completion(self, messages, config):
        time.sleep(0.04)
        with self.lock:
            self.calls += 1
            self.completed += 1
        return raw_response("def identity(value):\n    return value\n")


def _passing_verifier(record, **kwargs) -> dict:
    return {
        "schema_version": VERIFIER_SCHEMA_VERSION,
        "id": record["id"],
        "status": "accepted",
        "failure_category": "passed",
        "task": record["task"],
        "teacher_response": record["response_text"],
        "extracted_source": record["response_text"],
        "source_extraction": {
            "status": "passed",
            "removed_markdown_fence": False,
            "error": None,
        },
        "trace_validation": {"valid": True, "error": None},
        "phases": [],
    }


def test_streaming_pipeline_verifies_before_api_collection_finishes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path, 8)
    client = SlowTeacherClient(total=8)
    overlap_observed = False

    def verify(record, **kwargs):
        nonlocal overlap_observed
        with client.lock:
            overlap_observed |= client.completed < client.total
        return _passing_verifier(record)

    monkeypatch.setattr(
        "deepseek_distill.durable_pipeline.verify_normalized_record",
        verify,
    )
    summary = run_rejection_sampling(
        selected_tasks_path=tasks_path,
        run_dir=tmp_path / "run",
        client=client,
        config=GenerationConfig(top_logprobs=1),
        max_workers=2,
        requests_per_minute=0,
        verification_workers=2,
        max_attempts_per_task=1,
        streaming_pipeline=True,
    )

    assert overlap_observed is True
    assert summary.raw_attempts == 8
    assert summary.normalized_attempts == 8
    assert summary.verifier_results == 8
    state = json.loads((tmp_path / "run" / "pipeline_state.json").read_text())
    assert state["phase"] == "completed"
    assert state["queues"] == {
        "raw": 8,
        "normalized": 8,
        "normalization_errors": 0,
        "verifier": 8,
        "raw_to_normalized_lag": 0,
        "normalized_to_verifier_lag": 0,
    }


def test_streaming_pipeline_resume_skips_every_durable_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path, 3)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        "deepseek_distill.durable_pipeline.verify_normalized_record",
        _passing_verifier,
    )
    first_client = SlowTeacherClient(total=3)
    run_rejection_sampling(
        selected_tasks_path=tasks_path,
        run_dir=run_dir,
        client=first_client,
        config=GenerationConfig(top_logprobs=1),
        max_workers=2,
        requests_per_minute=0,
        verification_workers=2,
        max_attempts_per_task=1,
        streaming_pipeline=True,
    )
    before = {
        name: (run_dir / name).read_bytes()
        for name in (
            "raw_attempts.jsonl",
            "normalized_attempts.jsonl",
            "verifier_attempts.jsonl",
        )
    }
    resumed_client = SlowTeacherClient(total=0)

    resumed = run_rejection_sampling(
        selected_tasks_path=tasks_path,
        run_dir=run_dir,
        client=resumed_client,
        config=GenerationConfig(top_logprobs=1),
        max_workers=2,
        requests_per_minute=0,
        verification_workers=2,
        max_attempts_per_task=1,
        streaming_pipeline=True,
    )

    assert resumed_client.calls == 0
    assert resumed.waves[0].collection.skipped == 3
    assert all((run_dir / name).read_bytes() == contents for name, contents in before.items())


def test_streaming_rejection_sampling_uses_append_pipeline_for_later_waves(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path, 1)
    client = SlowTeacherClient(total=2)

    def fail_then_pass(record, **kwargs):
        result = _passing_verifier(record)
        if record["id"].endswith("__attempt_1"):
            result["status"] = "rejected"
            result["failure_category"] = "assertion_failure"
        return result

    monkeypatch.setattr(
        "deepseek_distill.durable_pipeline.verify_normalized_record",
        fail_then_pass,
    )
    monkeypatch.setattr(
        "deepseek_distill.code_verifier.verify_normalized_record",
        fail_then_pass,
    )

    summary = run_rejection_sampling(
        selected_tasks_path=tasks_path,
        run_dir=tmp_path / "run",
        client=client,
        config=GenerationConfig(top_logprobs=1),
        max_workers=1,
        requests_per_minute=0,
        verification_workers=1,
        max_attempts_per_task=3,
        streaming_pipeline=True,
    )

    assert [wave.planned for wave in summary.waves] == [1, 1, 0]
    assert summary.raw_attempts == 2
    assert summary.normalized_attempts == 2
    assert summary.verifier_results == 2

    resumed_client = SlowTeacherClient(total=0)
    resumed = run_rejection_sampling(
        selected_tasks_path=tasks_path,
        run_dir=tmp_path / "run",
        client=resumed_client,
        config=GenerationConfig(top_logprobs=1),
        max_workers=1,
        requests_per_minute=0,
        verification_workers=1,
        max_attempts_per_task=3,
        streaming_pipeline=True,
    )

    assert resumed_client.calls == 0
    assert resumed.raw_attempts == 2
    assert resumed.verifier_results == 2


def test_pipeline_drains_a_preexisting_raw_queue_without_new_api_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tasks = []
    for offset in range(4):
        original_id = 800 + offset
        task = base_task() | {
            "id": f"mbpp_{original_id}",
            "source": {
                "dataset": "MBPP",
                "split": "train",
                "original_id": original_id,
            },
        }
        tasks.append(make_attempt_task(task, attempt_number=1, selection_index=offset))
    input_path = tmp_path / "attempt_1_tasks.jsonl"
    input_path.write_text(
        "".join(json.dumps(task) + "\n" for task in tasks),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    raw_path = run_dir / "raw_attempts.jsonl"
    initial_client = SlowTeacherClient(total=4)
    collect_records(
        input_path=input_path,
        output_path=raw_path,
        client=initial_client,
        config=GenerationConfig(top_logprobs=1),
        max_workers=2,
        requests_per_minute=0,
    )
    monkeypatch.setattr(
        "deepseek_distill.durable_pipeline.verify_normalized_record",
        _passing_verifier,
    )

    def reject_redundant_collector(**kwargs):
        raise AssertionError("a complete durable raw queue must not be rescanned")

    monkeypatch.setattr(
        "deepseek_distill.durable_pipeline.collect_records",
        reject_redundant_collector,
    )
    resumed_client = SlowTeacherClient(total=0)

    summary = run_durable_collection_pipeline(
        input_path=input_path,
        raw_path=raw_path,
        normalized_path=run_dir / "normalized_attempts.jsonl",
        normalization_errors_path=run_dir / "normalization_errors.jsonl",
        verifier_path=run_dir / "verifier_attempts.jsonl",
        state_path=run_dir / "pipeline_state.json",
        client=resumed_client,
        config=GenerationConfig(top_logprobs=1),
        collection_workers=2,
        verification_workers=2,
        requests_per_minute=0,
        provider=None,
        phase_timeout_seconds=2,
        max_output_characters=4096,
    )

    assert resumed_client.calls == 0
    assert summary.collection.skipped == 4
    assert summary.normalization.normalized == 4
    assert summary.verification.passed == 4


def test_streaming_pipeline_uses_the_real_isolated_verifier(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path, 1)

    summary = run_rejection_sampling(
        selected_tasks_path=tasks_path,
        run_dir=tmp_path / "run",
        client=SlowTeacherClient(total=1),
        config=GenerationConfig(top_logprobs=1),
        max_workers=1,
        requests_per_minute=0,
        verification_workers=1,
        phase_timeout_seconds=2,
        max_attempts_per_task=1,
        streaming_pipeline=True,
    )

    assert summary.verifier_results == 1
    verifier = json.loads(
        (tmp_path / "run" / "verifier_attempts.jsonl").read_text(encoding="utf-8")
    )
    assert verifier["failure_category"] == "passed"


def test_streaming_first_wave_does_not_preparse_the_complete_raw_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = base_task() | {
        "id": "mbpp_900",
        "source": {
            "dataset": "MBPP",
            "split": "train",
            "original_id": 900,
        },
    }
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
    attempt = make_attempt_task(task, attempt_number=1, selection_index=0)
    attempt_path = tmp_path / "preexisting_attempt.jsonl"
    attempt_path.write_text(json.dumps(attempt) + "\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    collect_records(
        input_path=attempt_path,
        output_path=run_dir / "raw_attempts.jsonl",
        client=SlowTeacherClient(total=1),
        config=GenerationConfig(top_logprobs=1),
        requests_per_minute=0,
    )
    monkeypatch.setattr(
        "deepseek_distill.durable_pipeline.verify_normalized_record",
        _passing_verifier,
    )

    def reject_projection_reads(*args, **kwargs):
        raise AssertionError("streaming attempt 1 must consume the raw queue directly")

    monkeypatch.setattr(
        "deepseek_distill.rejection_sampling._read_jsonl_projection_if_exists",
        reject_projection_reads,
    )

    summary = run_rejection_sampling(
        selected_tasks_path=tasks_path,
        run_dir=run_dir,
        client=SlowTeacherClient(total=0),
        config=GenerationConfig(top_logprobs=1),
        requests_per_minute=0,
        max_attempts_per_task=1,
        streaming_pipeline=True,
    )

    assert summary.raw_attempts == 1
    assert summary.verifier_results == 1


def test_normalizer_yields_before_starving_ready_verifier_workers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tasks = []
    for offset in range(40):
        original_id = 1000 + offset
        task = base_task() | {
            "id": f"mbpp_{original_id}",
            "source": {
                "dataset": "MBPP",
                "split": "train",
                "original_id": original_id,
            },
        }
        tasks.append(make_attempt_task(task, attempt_number=1, selection_index=offset))
    input_path = tmp_path / "attempt_1_tasks.jsonl"
    input_path.write_text(
        "".join(json.dumps(task) + "\n" for task in tasks),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    collect_records(
        input_path=input_path,
        output_path=run_dir / "raw_attempts.jsonl",
        client=SlowTeacherClient(total=40),
        config=GenerationConfig(top_logprobs=1),
        max_workers=8,
        requests_per_minute=0,
    )
    normalized_count = 0
    lock = threading.Lock()
    verifier_start_counts: list[int] = []

    def tracked_normalize(record):
        nonlocal normalized_count
        normalized = real_normalize_raw_record(record)
        with lock:
            normalized_count += 1
        return normalized

    def tracked_verify(record, **kwargs):
        with lock:
            verifier_start_counts.append(normalized_count)
        return _passing_verifier(record)

    monkeypatch.setattr(
        "deepseek_distill.durable_pipeline.normalize_raw_record",
        tracked_normalize,
    )
    monkeypatch.setattr(
        "deepseek_distill.durable_pipeline.verify_normalized_record",
        tracked_verify,
    )

    run_durable_collection_pipeline(
        input_path=input_path,
        raw_path=run_dir / "raw_attempts.jsonl",
        normalized_path=run_dir / "normalized_attempts.jsonl",
        normalization_errors_path=run_dir / "normalization_errors.jsonl",
        verifier_path=run_dir / "verifier_attempts.jsonl",
        state_path=run_dir / "pipeline_state.json",
        client=SlowTeacherClient(total=0),
        config=GenerationConfig(top_logprobs=1),
        collection_workers=8,
        verification_workers=2,
        requests_per_minute=0,
        provider=None,
        phase_timeout_seconds=2,
        max_output_characters=4096,
    )

    assert verifier_start_counts
    assert min(verifier_start_counts) <= 16
