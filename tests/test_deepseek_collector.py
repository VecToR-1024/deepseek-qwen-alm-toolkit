import errno
import json
from pathlib import Path

import pytest

from deepseek_distill.api import GenerationConfig, build_success_record
from deepseek_distill.collector import (
    _append_jsonl,
    _bounded_parallel_results,
    _load_existing_ids,
    RateLimiter,
    collect_records,
)
from deepseek_distill.mbpp import MBPP_TASK_SCHEMA_VERSION
from deepseek_distill.multisource_tasks import make_multisource_task
from deepseek_distill.taco import TACO_TASK_SCHEMA_VERSION
from deepseek_distill.taco_retry import TACO_LENGTH_RETRY_SCHEMA_VERSION


def test_append_jsonl_retries_a_transient_windows_file_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_path = tmp_path / "raw.jsonl"
    original_open = Path.open
    attempts = 0

    def flaky_open(path: Path, *args, **kwargs):
        nonlocal attempts
        if path == output_path and args and args[0] == "a":
            attempts += 1
            if attempts < 3:
                raise PermissionError(errno.EACCES, "temporarily locked", str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)

    _append_jsonl(output_path, {"id": "attempt_1"})

    assert attempts == 3
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"id": "attempt_1"}


class FakeTeacherClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def create_completion(self, messages, config):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if prompt == "fail":
            raise RuntimeError("provider unavailable")
        return {"id": f"response-{prompt}", "model": config.model, "choices": []}


def write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_existing_id_scan_does_not_use_the_full_record_materializer(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "raw.jsonl"
    write_jsonl(
        output,
        [
            {"id": "attempt-1", "payload": "x" * 100_000},
            {"id": "attempt-2", "payload": "y" * 100_000},
        ],
    )

    def reject_materialization(path):
        raise AssertionError("existing raw records must be scanned one line at a time")

    monkeypatch.setattr(
        "deepseek_distill.collector._read_jsonl",
        reject_materialization,
    )

    assert _load_existing_ids(output) == {"attempt-1", "attempt-2"}


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_parallel_result_buffer_never_exceeds_worker_count(monkeypatch) -> None:
    class FakeFuture:
        def __init__(self, value, executor):
            self.value = value
            self.executor = executor

        def result(self):
            return self.value

    class FakeExecutor:
        instance = None

        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.live = 0
            self.max_live = 0
            self.submitted = 0
            FakeExecutor.instance = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def submit(self, function, item):
            self.submitted += 1
            self.live += 1
            self.max_live = max(self.max_live, self.live)
            return FakeFuture(function(item), self)

    def fake_wait(futures, *, return_when):
        completed = {next(iter(futures))}
        completed_future = next(iter(completed))
        completed_future.executor.live -= 1
        return completed, set(futures) - completed

    monkeypatch.setattr("deepseek_distill.collector.ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr("deepseek_distill.collector.wait", fake_wait)

    results = list(
        _bounded_parallel_results(
            lambda value: value * 2,
            list(range(5)),
            max_workers=2,
        )
    )

    assert sorted(results) == [0, 2, 4, 6, 8]
    assert FakeExecutor.instance.submitted == 5
    assert FakeExecutor.instance.max_live == 2


def test_collect_records_resumes_and_persists_successes_and_errors(tmp_path) -> None:
    input_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "raw.jsonl"
    prompts = [
        {"id": "done", "messages": [{"role": "user", "content": "done"}]},
        {"id": "good", "messages": [{"role": "user", "content": "good"}]},
        {"id": "bad", "messages": [{"role": "user", "content": "fail"}]},
    ]
    write_jsonl(input_path, prompts)
    existing = build_success_record(
        record_id="done",
        messages=prompts[0]["messages"],
        config=GenerationConfig(),
        response={"id": "existing", "choices": []},
        collected_at="2026-07-20T00:00:00Z",
    )
    write_jsonl(output_path, [existing])
    client = FakeTeacherClient()

    summary = collect_records(
        input_path=input_path,
        output_path=output_path,
        client=client,
        config=GenerationConfig(),
        max_workers=2,
        requests_per_minute=0,
    )

    records = read_jsonl(output_path)
    assert summary.total == 3
    assert summary.skipped == 1
    assert summary.succeeded == 1
    assert summary.failed == 1
    assert set(client.calls) == {"good", "fail"}
    assert {record["id"] for record in records} == {"done", "good", "bad"}
    failed = next(record for record in records if record["id"] == "bad")
    assert failed["status"] == "error"
    assert failed["error"]["type"] == "RuntimeError"
    assert "api_key" not in repr(failed).lower()


def test_collect_records_rejects_duplicate_input_ids_before_calling_provider(tmp_path) -> None:
    input_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "raw.jsonl"
    duplicate = {"id": "same", "messages": [{"role": "user", "content": "hello"}]}
    write_jsonl(input_path, [duplicate, duplicate])
    client = FakeTeacherClient()

    with pytest.raises(ValueError, match="duplicate input id"):
        collect_records(
            input_path=input_path,
            output_path=output_path,
            client=client,
            config=GenerationConfig(),
        )

    assert client.calls == []
    assert not output_path.exists()


def coding_task(*, problem_text: str = "Return the input unchanged.") -> dict:
    return {
        "schema_version": MBPP_TASK_SCHEMA_VERSION,
        "id": "mbpp_601",
        "source": {"dataset": "MBPP", "split": "train", "original_id": 601},
        "problem_text": problem_text,
        "function_name": "identity",
        "function_signature": "identity(value)",
        "tests": ["assert identity(1) == 1"],
        "metadata": {"reference_code": "def identity(value): return value"},
    }


def taco_task(*, problem_text: str = "Read one integer and print it.") -> dict:
    return {
        "schema_version": TACO_TASK_SCHEMA_VERSION,
        "id": "taco_train_000000",
        "source": {
            "dataset": "BAAI/TACO",
            "split": "train",
            "original_index": 0,
        },
        "problem_text": problem_text,
        "interface_type": "stdin_stdout",
        "tests": [{"input": "3\n", "output": "3\n"}],
        "metadata": {},
    }


def test_collect_records_builds_taco_prompt_without_test_leakage(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "raw.jsonl"
    task = taco_task()
    write_jsonl(input_path, [task])
    client = FakeTeacherClient()

    summary = collect_records(
        input_path=input_path,
        output_path=output_path,
        client=client,
        config=GenerationConfig(),
        requests_per_minute=0,
    )

    assert summary.succeeded == 1
    assert len(client.calls) == 1
    assert task["problem_text"] in client.calls[0]
    assert task["tests"][0]["input"] not in client.calls[0]
    assert read_jsonl(output_path)[0]["task"] == task


def test_collect_records_builds_blind_taco_length_retry_prompt(tmp_path) -> None:
    input_path = tmp_path / "retry_tasks.jsonl"
    output_path = tmp_path / "raw.jsonl"
    task = taco_task() | {
        "schema_version": TACO_LENGTH_RETRY_SCHEMA_VERSION,
        "id": "taco_train_000000__attempt_1__length_retry_v2",
        "problem_id": "taco_train_000000",
        "retry": {
            "source_attempt_id": "taco_train_000000__attempt_1",
            "source_finish_reason": "length",
            "max_tokens": 8192,
            "teacher_feedback": False,
        },
    }
    write_jsonl(input_path, [task])
    client = FakeTeacherClient()

    summary = collect_records(
        input_path=input_path,
        output_path=output_path,
        client=client,
        config=GenerationConfig(max_tokens=8192),
        requests_per_minute=0,
    )

    assert summary.succeeded == 1
    prompt = client.calls[0]
    assert "Task ID: taco_train_000000" in prompt
    assert "retry" not in prompt.lower()
    assert "length" not in prompt.lower()
    assert task["tests"][0]["input"] not in prompt


def test_collect_records_builds_mbpp_prompt_and_preserves_task_outside_request(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "raw.jsonl"
    write_jsonl(input_path, [coding_task()])
    client = FakeTeacherClient()

    summary = collect_records(
        input_path=input_path,
        output_path=output_path,
        client=client,
        config=GenerationConfig(),
        requests_per_minute=0,
        provider={"name": "DeepSeek", "base_url": "https://api.deepseek.com"},
    )

    record = read_jsonl(output_path)[0]
    assert summary.succeeded == 1
    assert len(client.calls) == 1
    assert "Problem:\nReturn the input unchanged." in client.calls[0]
    assert "assert identity" not in client.calls[0]
    assert record["task"]["tests"] == ["assert identity(1) == 1"]
    assert record["request"]["prompt_contract"] == {
        "id": "deepseek.python.clean.v2",
        "interface_type": "function",
    }
    assert record["provider"]["name"] == "DeepSeek"
    assert record["metrics"]["request_duration_seconds"] >= 0


def test_collect_records_dispatches_common_multisource_task_contract(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "raw.jsonl"
    task = make_multisource_task(
        task_id="apps_train_000001",
        source={
            "dataset": "codeparrot/apps",
            "split": "train",
            "original_id": 1,
            "revision": "pinned",
            "license": "MIT",
            "provenance": "https://github.com/hendrycks/apps",
            "mirror": "https://huggingface.co/datasets/codeparrot/apps",
        },
        problem_text="Read one integer and print it.",
        interface_type="stdin_stdout",
        required_interface=(
            "Complete Python program using standard input and standard output."
        ),
        tests=[{"input": "SECRET_TEST_INPUT\n", "output": "SECRET_TEST_OUTPUT\n"}],
        metadata={},
    )
    write_jsonl(input_path, [task])
    client = FakeTeacherClient()

    summary = collect_records(
        input_path=input_path,
        output_path=output_path,
        client=client,
        config=GenerationConfig(),
        requests_per_minute=0,
    )

    record = read_jsonl(output_path)[0]
    assert summary.succeeded == 1
    assert task["tests"][0]["input"] not in client.calls[0]
    assert record["request"]["prompt_contract"]["interface_type"] == "stdin_stdout"
    assert record["task"] == task


def test_missing_mbpp_problem_fails_before_any_api_call(tmp_path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "raw.jsonl"
    write_jsonl(input_path, [coding_task(problem_text="  ")])
    client = FakeTeacherClient()

    with pytest.raises(ValueError, match="actual problem statement"):
        collect_records(
            input_path=input_path,
            output_path=output_path,
            client=client,
            config=GenerationConfig(),
        )

    assert client.calls == []
    assert not output_path.exists()


def test_rate_limiter_spaces_request_starts() -> None:
    now = [10.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(
        requests_per_minute=120,
        clock=lambda: now[0],
        sleep=sleep,
    )

    limiter.wait()
    limiter.wait()

    assert sleeps == [0.5]
