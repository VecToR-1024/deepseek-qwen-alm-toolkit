from __future__ import annotations

import copy
import json

import pytest

import deepseek_distill.rejection_sampling as rejection_sampling

from deepseek_distill.api import GenerationConfig, build_error_record, build_success_record
from deepseek_distill.mbpp import build_teacher_messages
from deepseek_distill.multisource_tasks import (
    MULTISOURCE_TASK_SCHEMA_VERSION,
    make_multisource_task,
)
from deepseek_distill.normalize import normalize_jsonl_append
from deepseek_distill.offline_teacher import OfflineTeacherTraceProvider
from deepseek_distill.records import normalize_raw_record
from deepseek_distill.taco import TACO_REVISION, TACO_TASK_SCHEMA_VERSION
from deepseek_distill.rejection_sampling import (
    _append_normalization_failures_to_verifier,
    _count_jsonl_records,
    attempt_tasks_for_wave,
    build_rejection_sampling_datasets,
    collect_rejection_sampling_raw,
    make_attempt_id,
    make_attempt_task,
    parse_attempt_id,
    publish_jsonl_once,
    run_rejection_sampling,
    validate_campaign_tasks,
    write_rejection_sampling_outputs,
)


def test_normalization_failure_projection_does_not_materialize_raw_trace_file(
    tmp_path,
    monkeypatch,
) -> None:
    raw_path = tmp_path / "raw_attempts.jsonl"
    errors_path = tmp_path / "normalization_errors.jsonl"
    verifier_path = tmp_path / "verifier_attempts.jsonl"
    attempt_id = "mbpp_601__attempt_1"
    write_jsonl(
        raw_path,
        [
            {
                "id": attempt_id,
                "status": "ok",
                "task": base_task() | {"id": attempt_id},
                "response": {
                    "choices": [{"message": {"content": "def identity(x): return x"}}]
                },
            }
        ],
    )
    write_jsonl(
        errors_path,
        [
            {
                "id": attempt_id,
                "error": {"type": "RecordValidationError", "message": "bad trace"},
            }
        ],
    )
    original_read_jsonl = rejection_sampling._read_jsonl

    def guarded_read_jsonl(path):
        if path == raw_path:
            raise AssertionError("raw top-k traces must not be materialized as a list")
        return original_read_jsonl(path)

    monkeypatch.setattr(rejection_sampling, "_read_jsonl", guarded_read_jsonl)

    assert (
        _append_normalization_failures_to_verifier(
            raw_path=raw_path,
            normalization_errors_path=errors_path,
            verifier_path=verifier_path,
        )
        == 1
    )
    assert read_jsonl(verifier_path)[0]["failure_category"] == "malformed_trace"


def base_task() -> dict:
    return {
        "schema_version": "coding.task.mbpp.v1",
        "id": "mbpp_601",
        "source": {
            "dataset": "MBPP",
            "split": "train",
            "original_id": 601,
        },
        "problem_text": "Return the input unchanged.",
        "function_name": "identity",
        "function_signature": "identity(value)",
        "supporting_interfaces": [],
        "tests": ["assert identity(1) == 1"],
        "metadata": {
            "reference_code": "def identity(value): return value",
            "test_setup_code": "",
            "challenge_tests": [],
        },
    }


def test_jsonl_counting_does_not_materialize_records(tmp_path) -> None:
    path = tmp_path / "large.jsonl"
    path.write_text('{"large":"payload"}\n\n{"second":true}\n', encoding="utf-8")

    assert _count_jsonl_records(path) == 2
    assert _count_jsonl_records(tmp_path / "missing.jsonl") == 0


def taco_base_task() -> dict:
    return {
        "schema_version": TACO_TASK_SCHEMA_VERSION,
        "id": "taco_train_000017",
        "source": {
            "dataset": "BAAI/TACO",
            "split": "train",
            "original_index": 17,
            "revision": TACO_REVISION,
        },
        "problem_text": "Read an integer and print it.",
        "interface_type": "stdin_stdout",
        "tests": [{"input": "3\n", "output": "3\n"}],
        "metadata": {},
    }


def apps_base_task() -> dict:
    return make_multisource_task(
        task_id="apps_train_000017",
        source={
            "dataset": "codeparrot/apps",
            "config": "all",
            "split": "train",
            "original_id": 17,
            "revision": "21e74ddf8de1a21436da12e3e653065c5213e9d1",
            "license": "MIT",
            "provenance": "https://github.com/hendrycks/apps",
            "mirror": "https://huggingface.co/datasets/codeparrot/apps",
        },
        problem_text="Read an integer and print it.",
        interface_type="stdin_stdout",
        required_interface=(
            "Complete Python program using standard input and standard output."
        ),
        tests=[{"input": "3\n", "output": "3\n"}],
        metadata={},
    )


def raw_response(text: str) -> dict:
    encoded = list(text.encode("utf-8"))
    return {
        "id": "response-test",
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": text,
                    "reasoning_content": None,
                },
                "logprobs": {
                    "content": [
                        {
                            "token": text,
                            "bytes": encoded,
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": text, "bytes": encoded, "logprob": -0.1}
                            ],
                        }
                    ],
                    "reasoning_content": None,
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }


def write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_attempt_ids_and_metadata_are_stable_and_parseable() -> None:
    task = base_task()

    attempt = make_attempt_task(task, attempt_number=2, selection_index=17)

    assert make_attempt_id("mbpp_601", 2) == "mbpp_601__attempt_2"
    assert parse_attempt_id(attempt["id"]) == ("mbpp_601", 2)
    assert attempt["id"] == "mbpp_601__attempt_2"
    assert attempt["problem_id"] == "mbpp_601"
    assert attempt["attempt_number"] == 2
    assert attempt["selection_index"] == 17
    assert attempt["source"] == task["source"]


def test_taco_attempt_ids_and_campaign_contract_are_supported() -> None:
    task = taco_base_task()
    attempt = make_attempt_task(task, attempt_number=3, selection_index=4)

    assert attempt["id"] == "taco_train_000017__attempt_3"
    assert parse_attempt_id(attempt["id"]) == ("taco_train_000017", 3)
    validate_campaign_tasks(
        [task],
        expected_count=1,
        expected_revision=TACO_REVISION,
        expected_schema_version=TACO_TASK_SCHEMA_VERSION,
        expected_dataset="BAAI/TACO",
        expected_config=None,
        expected_split="train",
    )


def test_multisource_attempt_ids_and_campaign_contract_are_supported() -> None:
    task = apps_base_task()
    attempt = make_attempt_task(task, attempt_number=2, selection_index=8)

    assert attempt["id"] == "apps_train_000017__attempt_2"
    assert parse_attempt_id(attempt["id"]) == ("apps_train_000017", 2)
    validate_campaign_tasks(
        [task],
        expected_count=1,
        expected_revision=task["source"]["revision"],
        expected_schema_version=MULTISOURCE_TASK_SCHEMA_VERSION,
        expected_dataset="codeparrot/apps",
        expected_config="all",
        expected_split="train",
    )


def test_attempt_ids_reject_paths_and_reserved_attempt_suffixes() -> None:
    with pytest.raises(ValueError, match="supported benchmark task id"):
        make_attempt_id("../apps_train_17", 1)
    with pytest.raises(ValueError, match="supported benchmark task id"):
        make_attempt_id("apps_train_17__attempt_2", 1)


def test_multisource_rejection_outputs_use_source_specific_schema_names() -> None:
    task = apps_base_task()
    attempt = make_attempt_task(task, attempt_number=1, selection_index=0)

    datasets = build_rejection_sampling_datasets(
        selected_tasks=[task],
        raw_records=[{"id": attempt["id"], "status": "error"}],
        normalized_records=[],
        verifier_records=[],
        target=1,
        max_attempts_per_task=1,
    )

    assert datasets.attempt_ledger[0]["schema_version"] == (
        "coding.attempt.ledger.apps.v1"
    )
    assert datasets.rejected_tasks[0]["schema_version"] == (
        "coding.rejected.task.apps.v1"
    )


def test_attempt_prompts_are_identical_and_blind_across_attempts() -> None:
    task = base_task()
    first = make_attempt_task(task, attempt_number=1, selection_index=0)
    third = make_attempt_task(task, attempt_number=3, selection_index=0)

    first_messages = build_teacher_messages(first)
    third_messages = build_teacher_messages(third)

    assert first_messages == third_messages
    assert "Task ID: mbpp_601" in first_messages[1]["content"]
    assert "attempt" not in first_messages[1]["content"]
    assert task["tests"][0] not in repr(first_messages)
    assert task["metadata"]["reference_code"] not in repr(first_messages)


def test_append_normalizer_adds_only_new_successful_attempts_and_resumes(tmp_path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    normalized_path = tmp_path / "normalized.jsonl"
    config = GenerationConfig(top_logprobs=1)
    first_task = make_attempt_task(base_task(), attempt_number=1, selection_index=0)
    second_task = make_attempt_task(base_task(), attempt_number=2, selection_index=0)
    messages = build_teacher_messages(first_task)
    first = build_success_record(
        record_id=first_task["id"],
        messages=messages,
        config=config,
        response=raw_response("def identity(value):\n    return value\n"),
        task=first_task,
    )
    api_error = build_error_record(
        record_id=second_task["id"],
        messages=messages,
        config=config,
        error=RuntimeError("offline"),
        task=second_task,
    )
    write_jsonl(raw_path, [first, api_error])

    initial = normalize_jsonl_append(raw_path, normalized_path)
    resumed = normalize_jsonl_append(raw_path, normalized_path)

    assert initial.total == 2
    assert initial.normalized == 1
    assert initial.api_errors == 1
    assert initial.skipped == 0
    assert resumed.normalized == 0
    assert resumed.skipped == 1
    assert resumed.api_errors == 1
    assert [record["id"] for record in read_jsonl(normalized_path)] == [first_task["id"]]


def test_append_normalizer_persists_malformed_trace_and_continues(tmp_path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    normalized_path = tmp_path / "normalized.jsonl"
    errors_path = tmp_path / "normalization_errors.jsonl"
    task = make_attempt_task(base_task(), attempt_number=1, selection_index=0)
    response = raw_response("def identity(value):\n    return value\n")
    response["choices"][0]["logprobs"]["content"][0]["logprob"] = "not-a-number"
    raw = build_success_record(
        record_id=task["id"],
        messages=build_teacher_messages(task),
        config=GenerationConfig(top_logprobs=1),
        response=response,
        task=task,
    )
    write_jsonl(raw_path, [raw])

    first = normalize_jsonl_append(
        raw_path,
        normalized_path,
        error_output_path=errors_path,
    )
    resumed = normalize_jsonl_append(
        raw_path,
        normalized_path,
        error_output_path=errors_path,
    )

    assert first.malformed == 1
    assert resumed.malformed == 0
    assert resumed.skipped == 1
    assert read_jsonl(errors_path)[0]["failure_category"] == "malformed_trace"
    assert not normalized_path.exists()


def test_append_normalizer_does_not_retain_all_raw_records(
    tmp_path, monkeypatch
) -> None:
    import deepseek_distill.normalize as normalize_module

    raw_path = tmp_path / "raw.jsonl"
    normalized_path = tmp_path / "normalized.jsonl"
    task = make_attempt_task(base_task(), attempt_number=1, selection_index=0)
    template = build_success_record(
        record_id=task["id"],
        messages=build_teacher_messages(task),
        config=GenerationConfig(top_logprobs=1),
        response=raw_response("def identity(value):\n    return value\n"),
        task=task,
    )
    records = []
    for index in range(12):
        record = copy.deepcopy(template)
        record["id"] = f"mbpp_{700 + index}__attempt_1"
        records.append(record)
    write_jsonl(raw_path, records)

    original_parse = normalize_module._parse_object

    class TrackedRecord(dict):
        live = 0
        peak = 0

        def __init__(self, value):
            super().__init__(value)
            type(self).live += 1
            type(self).peak = max(type(self).peak, type(self).live)

        def __del__(self):
            type(self).live -= 1

    def tracked_parse(path, line_number, line):
        return TrackedRecord(original_parse(path, line_number, line))

    monkeypatch.setattr(normalize_module, "_parse_object", tracked_parse)

    summary = normalize_jsonl_append(raw_path, normalized_path)

    assert summary.normalized == 12
    assert TrackedRecord.peak <= 3


def test_wave_plan_retries_only_tasks_not_previously_accepted() -> None:
    first = base_task()
    second = base_task() | {
        "id": "mbpp_602",
        "source": {"dataset": "MBPP", "split": "train", "original_id": 602},
    }
    third = base_task() | {
        "id": "mbpp_603",
        "source": {"dataset": "MBPP", "split": "train", "original_id": 603},
    }
    verifier_records = [
        {
            "id": "mbpp_601__attempt_1",
            "failure_category": "passed",
            "task": make_attempt_task(first, attempt_number=1, selection_index=0),
        },
        {
            "id": "mbpp_602__attempt_1",
            "failure_category": "assertion_failure",
            "task": make_attempt_task(second, attempt_number=1, selection_index=1),
        },
    ]
    raw_records = [
        {"id": "mbpp_601__attempt_1", "status": "ok"},
        {"id": "mbpp_602__attempt_1", "status": "ok"},
        {"id": "mbpp_603__attempt_1", "status": "error"},
    ]

    second_wave = attempt_tasks_for_wave(
        [first, second, third],
        raw_records=raw_records,
        verifier_records=verifier_records,
        attempt_number=2,
    )

    assert [record["id"] for record in second_wave] == [
        "mbpp_602__attempt_2",
        "mbpp_603__attempt_2",
    ]
    assert [record["selection_index"] for record in second_wave] == [1, 2]


def test_wave_plan_rejects_out_of_order_or_mismatched_attempt_history() -> None:
    task = base_task()
    with pytest.raises(ValueError, match="attempt 1 is missing"):
        attempt_tasks_for_wave(
            [task],
            raw_records=[{"id": "mbpp_601__attempt_2", "status": "ok"}],
            verifier_records=[
                {
                    "id": "mbpp_601__attempt_2",
                    "failure_category": "assertion_failure",
                    "task": make_attempt_task(task, attempt_number=2, selection_index=0),
                }
            ],
            attempt_number=3,
        )


def test_publish_jsonl_once_is_idempotent_but_refuses_changed_content(tmp_path) -> None:
    output = tmp_path / "attempt_1_tasks.jsonl"
    records = [make_attempt_task(base_task(), attempt_number=1, selection_index=0)]

    assert publish_jsonl_once(output, records) == "created"
    assert publish_jsonl_once(output, records) == "unchanged"

    changed = [dict(records[0], problem_text="changed")]
    with pytest.raises(FileExistsError, match="different content"):
        publish_jsonl_once(output, changed)


class SequencedTeacherClient:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self.calls_by_problem: dict[str, int] = {}

    def create_completion(self, messages, config):
        copied = [dict(message) for message in messages]
        self.calls.append(copied)
        user_prompt = copied[1]["content"]
        problem_id = next(
            line.removeprefix("Task ID: ")
            for line in user_prompt.splitlines()
            if line.startswith("Task ID: ")
        )
        attempt = self.calls_by_problem.get(problem_id, 0) + 1
        self.calls_by_problem[problem_id] = attempt
        if problem_id == "mbpp_601" or (problem_id == "mbpp_602" and attempt >= 2):
            code = "def identity(value):\n    return value\n"
        else:
            code = "def identity(value):\n    return value + 1\n"
        return raw_response(code)


def test_rejection_sampling_runs_three_blind_waves_and_resumes(
    tmp_path, monkeypatch
) -> None:
    tasks = []
    for offset in range(3):
        original_id = 601 + offset
        tasks.append(
            base_task()
            | {
                "id": f"mbpp_{original_id}",
                "source": {
                    "dataset": "MBPP",
                    "split": "train",
                    "original_id": original_id,
                },
            }
        )
    tasks_path = tmp_path / "selected_tasks.jsonl"
    write_jsonl(tasks_path, tasks)
    client = SequencedTeacherClient()

    summary = run_rejection_sampling(
        selected_tasks_path=tasks_path,
        run_dir=tmp_path / "run",
        client=client,
        config=GenerationConfig(top_logprobs=1),
        max_workers=1,
        requests_per_minute=0,
        phase_timeout_seconds=2,
    )

    assert [wave.planned for wave in summary.waves] == [3, 2, 1]
    assert client.calls_by_problem == {"mbpp_601": 1, "mbpp_602": 2, "mbpp_603": 3}
    assert len(read_jsonl(tmp_path / "run" / "raw_attempts.jsonl")) == 6
    verifier = read_jsonl(tmp_path / "run" / "verifier_attempts.jsonl")
    assert [record["failure_category"] for record in verifier].count("passed") == 2
    assert all(task["tests"][0] not in repr(messages) for task in tasks for messages in client.calls)
    assert all("assertion_failure" not in repr(messages) for messages in client.calls)

    import deepseek_distill.rejection_sampling as sampling_module

    original_read_jsonl = sampling_module._read_jsonl

    def reject_full_trace_reads(path):
        if path.name in {"raw_attempts.jsonl", "verifier_attempts.jsonl"}:
            raise AssertionError("attempt history must be read as a light projection")
        return original_read_jsonl(path)

    monkeypatch.setattr(sampling_module, "_read_jsonl", reject_full_trace_reads)
    resumed_client = SequencedTeacherClient()
    resumed = run_rejection_sampling(
        selected_tasks_path=tasks_path,
        run_dir=tmp_path / "run",
        client=resumed_client,
        config=GenerationConfig(top_logprobs=1),
        max_workers=1,
        requests_per_minute=0,
        phase_timeout_seconds=2,
    )

    assert resumed_client.calls == []
    assert [wave.collection.skipped for wave in resumed.waves] == [3, 2, 1]
    assert len(read_jsonl(tmp_path / "run" / "raw_attempts.jsonl")) == 6


def test_rejection_sampling_can_stop_after_one_blind_attempt(tmp_path) -> None:
    task = base_task() | {
        "id": "mbpp_603",
        "source": {
            "dataset": "MBPP",
            "split": "train",
            "original_id": 603,
        },
    }
    tasks_path = tmp_path / "selected_tasks.jsonl"
    write_jsonl(tasks_path, [task])
    client = SequencedTeacherClient()

    summary = run_rejection_sampling(
        selected_tasks_path=tasks_path,
        run_dir=tmp_path / "run",
        client=client,
        config=GenerationConfig(top_logprobs=1),
        max_workers=1,
        requests_per_minute=0,
        phase_timeout_seconds=2,
        max_attempts_per_task=1,
    )

    assert [wave.planned for wave in summary.waves] == [1]
    assert client.calls_by_problem == {"mbpp_603": 1}
    assert not (tmp_path / "run" / "attempt_2_tasks.jsonl").exists()

    datasets = build_rejection_sampling_datasets(
        selected_tasks=[task],
        raw_records=read_jsonl(tmp_path / "run" / "raw_attempts.jsonl"),
        normalized_records=read_jsonl(tmp_path / "run" / "normalized_attempts.jsonl"),
        verifier_records=read_jsonl(tmp_path / "run" / "verifier_attempts.jsonl"),
        target=1,
        max_attempts_per_task=1,
    )
    assert datasets.summary.pending_attempt_slots == 0
    assert datasets.rejected_tasks[0]["campaign_complete"] is True
    assert [entry["attempt_number"] for entry in datasets.attempt_ledger] == [1]


def test_rejection_sampling_rejects_invalid_verifier_workers_before_api_call(
    tmp_path,
) -> None:
    tasks_path = tmp_path / "selected_tasks.jsonl"
    write_jsonl(tasks_path, [base_task()])
    client = SequencedTeacherClient()

    with pytest.raises(ValueError, match="verification_workers"):
        run_rejection_sampling(
            selected_tasks_path=tasks_path,
            run_dir=tmp_path / "run",
            client=client,
            config=GenerationConfig(top_logprobs=1),
            requests_per_minute=0,
            verification_workers=0,
            max_attempts_per_task=1,
        )

    assert client.calls == []


class MalformedThenPassClient:
    def __init__(self) -> None:
        self.calls = 0

    def create_completion(self, messages, config):
        self.calls += 1
        response = raw_response("def identity(value):\n    return value\n")
        if self.calls == 1:
            response["choices"][0]["logprobs"]["content"][0]["logprob"] = "bad"
        return response


def test_rejection_sampling_retries_structurally_malformed_trace(tmp_path) -> None:
    tasks_path = tmp_path / "selected_tasks.jsonl"
    write_jsonl(tasks_path, [base_task()])
    client = MalformedThenPassClient()

    summary = run_rejection_sampling(
        selected_tasks_path=tasks_path,
        run_dir=tmp_path / "run",
        client=client,
        config=GenerationConfig(top_logprobs=1),
        requests_per_minute=0,
        phase_timeout_seconds=2,
    )

    assert [wave.planned for wave in summary.waves] == [1, 1, 0]
    assert client.calls == 2
    errors = read_jsonl(tmp_path / "run" / "normalization_errors.jsonl")
    assert [record["id"] for record in errors] == ["mbpp_601__attempt_1"]
    verifier_records = read_jsonl(tmp_path / "run" / "verifier_attempts.jsonl")
    assert [record["failure_category"] for record in verifier_records] == [
        "malformed_trace",
        "passed",
    ]
    datasets = build_rejection_sampling_datasets(
        selected_tasks=[base_task()],
        raw_records=read_jsonl(tmp_path / "run" / "raw_attempts.jsonl"),
        normalized_records=read_jsonl(tmp_path / "run" / "normalized_attempts.jsonl"),
        verifier_records=verifier_records,
        target=1,
    )
    assert datasets.accepted_unique[0]["sampling"]["attempt_number"] == 2
    assert datasets.rejected_attempts[0]["failure_category"] == "malformed_trace"


class TacoEchoClient:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def create_completion(self, messages, config):
        self.calls.append([dict(message) for message in messages])
        return raw_response("value = input()\nprint(value)\n")


def test_raw_only_collection_is_resumable_and_does_not_start_downstream_stages(
    tmp_path,
) -> None:
    task = apps_base_task()
    tasks_path = tmp_path / "selected_tasks.jsonl"
    run_dir = tmp_path / "run"
    write_jsonl(tasks_path, [task])
    client = TacoEchoClient()

    initial = collect_rejection_sampling_raw(
        selected_tasks_path=tasks_path,
        run_dir=run_dir,
        client=client,
        config=GenerationConfig(top_logprobs=1),
        max_workers=2,
        requests_per_minute=0,
    )

    assert initial.selected_tasks == 1
    assert initial.raw_attempts == 1
    assert initial.collection.succeeded == 1
    assert len(client.calls) == 1
    assert task["tests"][0]["input"] not in repr(client.calls)
    assert (run_dir / "attempt_1_tasks.jsonl").exists()
    assert (run_dir / "raw_attempts.jsonl").exists()
    assert not (run_dir / "normalized_attempts.jsonl").exists()
    assert not (run_dir / "normalization_errors.jsonl").exists()
    assert not (run_dir / "verifier_attempts.jsonl").exists()

    resumed_client = TacoEchoClient()
    resumed = collect_rejection_sampling_raw(
        selected_tasks_path=tasks_path,
        run_dir=run_dir,
        client=resumed_client,
        config=GenerationConfig(top_logprobs=1),
        max_workers=2,
        requests_per_minute=0,
    )

    assert resumed.raw_attempts == 1
    assert resumed.collection.skipped == 1
    assert resumed_client.calls == []
    assert len(read_jsonl(run_dir / "raw_attempts.jsonl")) == 1


def test_taco_rejection_sampling_fake_response_end_to_end(tmp_path) -> None:
    task = taco_base_task()
    tasks_path = tmp_path / "selected_tasks.jsonl"
    write_jsonl(tasks_path, [task])
    client = TacoEchoClient()

    summary = run_rejection_sampling(
        selected_tasks_path=tasks_path,
        run_dir=tmp_path / "run",
        client=client,
        config=GenerationConfig(top_logprobs=1),
        requests_per_minute=0,
        phase_timeout_seconds=2,
    )

    assert [wave.planned for wave in summary.waves] == [1, 0, 0]
    assert len(client.calls) == 1
    assert task["tests"][0]["input"] not in repr(client.calls)
    verifier = read_jsonl(tmp_path / "run" / "verifier_attempts.jsonl")
    assert verifier[0]["schema_version"] == "coding.verifier.taco.v1"
    assert verifier[0]["failure_category"] == "passed"


def test_campaign_validation_pins_unique_mbpp_full_train_tasks() -> None:
    tasks = []
    for offset in range(3):
        original_id = 601 + offset
        task = base_task() | {"id": f"mbpp_{original_id}"}
        task["source"] = {
            "dataset": "MBPP",
            "config": "full",
            "split": "train",
            "original_id": original_id,
            "revision": "pinned",
        }
        tasks.append(task)

    validate_campaign_tasks(tasks, expected_count=3, expected_revision="pinned")

    tasks[2]["source"]["split"] = "test"
    with pytest.raises(ValueError, match="full/train"):
        validate_campaign_tasks(tasks, expected_count=3, expected_revision="pinned")


def _attempt_artifacts(task: dict, attempt_number: int, category: str, selection_index: int):
    attempt_task = make_attempt_task(
        task,
        attempt_number=attempt_number,
        selection_index=selection_index,
    )
    raw = build_success_record(
        record_id=attempt_task["id"],
        messages=build_teacher_messages(attempt_task),
        config=GenerationConfig(top_logprobs=1),
        response=raw_response("def identity(value):\n    return value\n"),
        task=attempt_task,
    )
    normalized = normalize_raw_record(raw)
    verifier = {
        "schema_version": "coding.verifier.mbpp.v1",
        "id": attempt_task["id"],
        "status": "accepted" if category == "passed" else "rejected",
        "failure_category": category,
        "task": attempt_task,
        "trace_validation": {"valid": True, "error": None},
        "source_extraction": {"status": "passed", "removed_markdown_fence": False},
        "extracted_source": normalized["response_text"],
        "phases": [],
    }
    return raw, normalized, verifier


def test_dataset_aggregation_keeps_earliest_pass_in_seeded_order() -> None:
    task_602 = base_task() | {
        "id": "mbpp_602",
        "source": {"dataset": "MBPP", "split": "train", "original_id": 602},
    }
    task_601 = base_task()
    task_603 = base_task() | {
        "id": "mbpp_603",
        "source": {"dataset": "MBPP", "split": "train", "original_id": 603},
    }
    selected = [task_602, task_601, task_603]
    artifacts = [
        _attempt_artifacts(task_602, 1, "passed", 0),
        _attempt_artifacts(task_601, 1, "assertion_failure", 1),
        _attempt_artifacts(task_601, 2, "passed", 1),
        _attempt_artifacts(task_603, 1, "syntax_error", 2),
        _attempt_artifacts(task_603, 2, "assertion_failure", 2),
        _attempt_artifacts(task_603, 3, "timeout", 2),
    ]
    raw_records = [artifact[0] for artifact in reversed(artifacts)]
    normalized_records = [artifact[1] for artifact in artifacts]
    verifier_records = [artifact[2] for artifact in artifacts]

    datasets = build_rejection_sampling_datasets(
        selected_tasks=selected,
        raw_records=raw_records,
        normalized_records=normalized_records,
        verifier_records=verifier_records,
        target=1,
    )

    assert [record["sampling"]["problem_id"] for record in datasets.accepted_unique] == [
        "mbpp_602",
        "mbpp_601",
    ]
    assert [record["sampling"]["attempt_number"] for record in datasets.accepted_unique] == [
        1,
        2,
    ]
    assert [record["sampling"]["problem_id"] for record in datasets.accepted_first_target] == [
        "mbpp_602"
    ]
    assert OfflineTeacherTraceProvider().get_trace(datasets.accepted_unique[0]).response_text
    assert len(datasets.rejected_attempts) == 4
    assert [record["problem_id"] for record in datasets.rejected_tasks] == ["mbpp_603"]
    assert len(datasets.attempt_ledger) == 9
    ledger = {record["id"]: record for record in datasets.attempt_ledger}
    assert ledger["mbpp_602__attempt_2"]["state"] == "not_requested_after_pass"
    assert ledger["mbpp_601__attempt_3"]["state"] == "not_requested_after_pass"
    assert datasets.summary.target_met is True
    assert datasets.summary.duplicate_attempt_ids == 0


def test_dataset_aggregation_reports_target_shortfall_without_reordering() -> None:
    task = base_task()
    artifacts = [
        _attempt_artifacts(task, attempt, "assertion_failure", 0)
        for attempt in (1, 2, 3)
    ]

    datasets = build_rejection_sampling_datasets(
        selected_tasks=[task],
        raw_records=[artifact[0] for artifact in artifacts],
        normalized_records=[artifact[1] for artifact in artifacts],
        verifier_records=[artifact[2] for artifact in artifacts],
        target=2,
    )

    assert datasets.accepted_first_target == []
    assert datasets.summary.target_met is False
    assert datasets.summary.shortfall == 2


def test_dataset_outputs_are_deterministic_and_non_destructive(tmp_path) -> None:
    task = base_task()
    raw, normalized, verifier = _attempt_artifacts(task, 1, "passed", 0)
    datasets = build_rejection_sampling_datasets(
        selected_tasks=[task],
        raw_records=[raw],
        normalized_records=[normalized],
        verifier_records=[verifier],
        target=1,
    )

    first = write_rejection_sampling_outputs(tmp_path, datasets)
    repeated = write_rejection_sampling_outputs(tmp_path, datasets)

    assert set(first.values()) == {"created"}
    assert set(repeated.values()) == {"unchanged"}
    assert read_jsonl(tmp_path / "accepted_first_1.jsonl")[0]["id"].endswith("attempt_1")
    assert json.loads((tmp_path / "dataset_summary.json").read_text(encoding="utf-8"))[
        "target_met"
    ]


def test_lightweight_rejections_reference_durable_attempt_files() -> None:
    task = base_task()
    raw, normalized, verifier = _attempt_artifacts(
        task, 1, "assertion_failure", 0
    )

    datasets = build_rejection_sampling_datasets(
        selected_tasks=[task],
        raw_records=[raw],
        normalized_records=[normalized],
        verifier_records=[verifier],
        target=1,
        embed_rejected_records=False,
    )

    rejection = datasets.rejected_attempts[0]
    assert "raw_record" not in rejection
    assert rejection["artifacts"]["raw"] == {
        "path": "raw_attempts.jsonl",
        "id": "mbpp_601__attempt_1",
    }
