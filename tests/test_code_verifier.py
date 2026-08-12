from __future__ import annotations

import json
from typing import Any

import pytest

from deepseek_distill.code_verifier import (
    SourceExtractionError,
    extract_python_source,
    verify_jsonl,
    verify_normalized_record,
)
from deepseek_distill.records import NORMALIZED_SCHEMA_VERSION


def normalized_code_record(
    source: str,
    *,
    tests: list[str] | None = None,
    function_name: str = "identity",
) -> dict:
    encoded = source.encode("utf-8")
    return {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "id": "mbpp_601",
        "request": {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "problem"},
            ]
        },
        "response_text": source,
        "content_tokens": [
            {
                "token": source,
                "bytes": list(encoded),
                "logprob": -0.1,
                "top_logprobs": [],
            }
        ],
        "validation": {"content_bytes_match": True, "warnings": []},
        "task": {
            "schema_version": "coding.task.mbpp.v1",
            "id": "mbpp_601",
            "problem_text": "Return the input unchanged.",
            "function_name": function_name,
            "function_signature": f"{function_name}(value)",
            "tests": tests or [f"assert {function_name}(3) == 3"],
            "metadata": {"test_setup_code": "", "challenge_tests": []},
        },
    }


def normalized_stdio_record(
    source: str,
    *,
    tests: list[dict[str, Any]] | None = None,
) -> dict:
    encoded = source.encode("utf-8")
    return {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "id": "taco_train_000000__attempt_1",
        "request": {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "problem"},
            ]
        },
        "response_text": source,
        "content_tokens": [
            {
                "token": source,
                "bytes": list(encoded),
                "logprob": -0.1,
                "top_logprobs": [],
            }
        ],
        "validation": {"content_bytes_match": True, "warnings": []},
        "task": {
            "schema_version": "coding.task.taco.v1",
            "id": "taco_train_000000__attempt_1",
            "problem_id": "taco_train_000000",
            "problem_text": "Read an integer and print it.",
            "interface_type": "stdin_stdout",
            "tests": tests or [{"input": "3\n", "output": "3\n"}],
            "metadata": {},
        },
    }


def test_extract_accepts_plain_python_and_one_accidental_fence() -> None:
    source = "def identity(value):\n    return value\n"

    plain = extract_python_source(source, required_function_name="identity")
    fenced = extract_python_source(
        f"```python\n{source}```", required_function_name="identity"
    )

    assert plain.source == source
    assert plain.removed_markdown_fence is False
    assert fenced.source == source
    assert fenced.removed_markdown_fence is True


def test_extract_rejects_prose_around_a_fence_instead_of_guessing() -> None:
    response = "Here is the solution:\n```python\ndef identity(value):\n    return value\n```"

    with pytest.raises(SourceExtractionError) as caught:
        extract_python_source(response, required_function_name="identity")

    assert caught.value.category == "extraction_error"


def test_extract_classifies_syntax_and_missing_function() -> None:
    with pytest.raises(SourceExtractionError) as syntax:
        extract_python_source("def identity(:\n    pass", required_function_name="identity")
    assert syntax.value.category == "syntax_error"

    with pytest.raises(SourceExtractionError) as missing:
        extract_python_source("def other(value):\n    return value\n", required_function_name="identity")
    assert missing.value.category == "missing_function"


def test_extract_rejects_module_scope_invocation_of_submitted_function() -> None:
    source = "def identity(value):\n    return value\n\nidentity(1)\n"

    with pytest.raises(SourceExtractionError) as caught:
        extract_python_source(source, required_function_name="identity")

    assert caught.value.category == "forbidden_operation"


def test_extract_allows_target_calls_inside_function_and_method_bodies() -> None:
    source = (
        "def identity(value):\n"
        "    return value\n\n"
        "def indirect(value):\n"
        "    return identity(value)\n\n"
        "class Wrapper:\n"
        "    def call(self, value):\n"
        "        return identity(value)\n"
    )

    extracted = extract_python_source(source, required_function_name="identity")

    assert extracted.source == source


def test_verifier_passes_compile_import_and_official_tests_in_children() -> None:
    record = normalized_code_record("def identity(value):\n    return value\n")

    result = verify_normalized_record(record, phase_timeout_seconds=3.0)

    assert result["failure_category"] == "passed"
    assert [phase["name"] for phase in result["phases"]] == ["compile", "import", "test"]
    assert all(phase["status"] == "passed" for phase in result["phases"])
    assert result["teacher_response"] == record["response_text"]
    assert result["extracted_source"] == record["response_text"]
    assert result["runtime"]["python_version"]
    assert result["runtime"]["executable"]


def test_verifier_classifies_assertion_failure() -> None:
    record = normalized_code_record("def identity(value):\n    return value + 1\n")

    result = verify_normalized_record(record, phase_timeout_seconds=3.0)

    assert result["failure_category"] == "assertion_failure"
    assert result["phases"][-1]["name"] == "test"
    assert result["phases"][-1]["status"] == "assertion_failure"


def test_verifier_classifies_import_error() -> None:
    source = "raise RuntimeError('module failed')\n\ndef identity(value):\n    return value\n"

    result = verify_normalized_record(
        normalized_code_record(source), phase_timeout_seconds=3.0
    )

    assert result["failure_category"] == "import_error"
    assert result["phases"][-1]["name"] == "import"


def test_verifier_classifies_timeout_without_running_in_parent() -> None:
    source = "def identity(value):\n    return value\n\nwhile True:\n    pass\n"

    result = verify_normalized_record(
        normalized_code_record(source), phase_timeout_seconds=0.75
    )

    assert result["failure_category"] == "timeout"
    assert result["phases"][-1]["name"] == "import"


@pytest.mark.parametrize(
    "source",
    [
        "def identity(value):\n    open('leak.txt', 'w').write('x')\n    return value\n",
        "import socket\ndef identity(value):\n    return value\n",
        "def identity(value):\n    return eval(str(value))\n",
    ],
)
def test_verifier_rejects_obvious_forbidden_operations_before_child_execution(
    source: str,
) -> None:
    result = verify_normalized_record(
        normalized_code_record(source), phase_timeout_seconds=3.0
    )

    assert result["failure_category"] == "forbidden_operation"
    assert result["phases"] == []


def test_verifier_rejects_malformed_trace_before_source_handling() -> None:
    record = normalized_code_record("def identity(value):\n    return value\n")
    record["content_tokens"][0]["bytes"] = list(b"wrong")

    result = verify_normalized_record(record, phase_timeout_seconds=3.0)

    assert result["failure_category"] == "malformed_trace"
    assert result["extracted_source"] is None


def test_stdio_verifier_runs_each_official_case_in_a_child_process() -> None:
    record = normalized_stdio_record(
        "value = int(input())\nprint(value)\n",
        tests=[
            {"input": "3\n", "output": "3\n"},
            {"input": "-4\n", "output": "-4\n"},
        ],
    )

    result = verify_normalized_record(record, phase_timeout_seconds=3.0)

    assert result["failure_category"] == "passed"
    assert [phase["name"] for phase in result["phases"]] == [
        "compile",
        "test_0",
        "test_1",
    ]
    assert all("3" not in (phase.get("error_message") or "") for phase in result["phases"])


def test_stdio_verifier_accepts_any_official_output_alternative() -> None:
    record = normalized_stdio_record(
        "print(7)\n",
        tests=[
            {
                "input": "",
                "output": ["seven\n", " 7 \n", "7\n"],
            }
        ],
    )

    result = verify_normalized_record(record, phase_timeout_seconds=3.0)

    assert result["failure_category"] == "passed"
    assert result["output_comparison"] == (
        "normalize_newlines_strip_outer_whitespace_any_expected_v1"
    )


def test_stdio_verifier_classifies_wrong_output_without_leaking_hidden_values() -> None:
    result = verify_normalized_record(
        normalized_stdio_record("print(999)\n"),
        phase_timeout_seconds=3.0,
    )

    assert result["failure_category"] == "assertion_failure"
    assert result["phases"][-1]["name"] == "test_0"
    assert "3" not in (result["phases"][-1].get("error_message") or "")


def test_stdio_verifier_classifies_timeout_and_forbidden_operation() -> None:
    timeout = verify_normalized_record(
        normalized_stdio_record("while True:\n    pass\n"),
        phase_timeout_seconds=0.5,
    )
    forbidden = verify_normalized_record(
        normalized_stdio_record("import socket\nprint(input())\n"),
        phase_timeout_seconds=3.0,
    )

    assert timeout["failure_category"] == "timeout"
    assert forbidden["failure_category"] == "forbidden_operation"
    assert forbidden["phases"] == []


def test_verify_jsonl_is_append_only_and_resumable(tmp_path) -> None:
    input_path = tmp_path / "normalized.jsonl"
    output_path = tmp_path / "verified.jsonl"
    records = [
        normalized_code_record("def identity(value):\n    return value\n"),
        {
            **normalized_code_record("def identity(value):\n    return value + 1\n"),
            "id": "mbpp_602",
            "task": {
                **normalized_code_record("def identity(value):\n    return value\n")["task"],
                "id": "mbpp_602",
            },
        },
    ]
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    first = verify_jsonl(
        input_path=input_path,
        output_path=output_path,
        phase_timeout_seconds=3.0,
    )
    resumed = verify_jsonl(
        input_path=input_path,
        output_path=output_path,
        phase_timeout_seconds=3.0,
    )

    output_records = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert first.total == 2
    assert first.passed == 1
    assert first.failed == 1
    assert resumed.skipped == 2
    assert len(output_records) == 2


def test_verify_jsonl_does_not_retain_all_normalized_records(
    tmp_path, monkeypatch
) -> None:
    import deepseek_distill.code_verifier as verifier_module

    input_path = tmp_path / "normalized.jsonl"
    output_path = tmp_path / "verified.jsonl"
    records = []
    for index in range(12):
        record = normalized_code_record("def identity(value):\n    return value\n")
        record["id"] = f"mbpp_{700 + index}"
        record["task"]["id"] = record["id"]
        records.append(record)
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    original_loads = verifier_module.json.loads

    class TrackedRecord(dict):
        live = 0
        peak = 0

        def __init__(self, value):
            super().__init__(value)
            type(self).live += 1
            type(self).peak = max(type(self).peak, type(self).live)

        def __del__(self):
            type(self).live -= 1

    def tracked_loads(value):
        parsed = original_loads(value)
        return TrackedRecord(parsed) if isinstance(parsed, dict) else parsed

    def fake_verify(record, **_kwargs):
        return {
            "id": record["id"],
            "status": "accepted",
            "failure_category": "passed",
        }

    monkeypatch.setattr(verifier_module.json, "loads", tracked_loads)
    monkeypatch.setattr(verifier_module, "verify_normalized_record", fake_verify)

    summary = verify_jsonl(input_path=input_path, output_path=output_path)

    assert summary.passed == 12
    assert TrackedRecord.peak <= 3


def test_verify_jsonl_can_run_bounded_parallel_work_in_input_order(
    tmp_path, monkeypatch
) -> None:
    import threading
    import time

    import deepseek_distill.code_verifier as verifier_module

    input_path = tmp_path / "normalized.jsonl"
    output_path = tmp_path / "verified.jsonl"
    records = []
    for index in range(6):
        record = normalized_code_record("def identity(value):\n    return value\n")
        record["id"] = f"mbpp_{700 + index}"
        record["task"]["id"] = record["id"]
        records.append(record)
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    lock = threading.Lock()
    active = 0
    peak = 0

    def fake_verify(record, **_kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.04 if record["id"] == "mbpp_700" else 0.01)
            return {
                "id": record["id"],
                "status": "accepted",
                "failure_category": "passed",
            }
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(verifier_module, "verify_normalized_record", fake_verify)

    summary = verify_jsonl(
        input_path=input_path,
        output_path=output_path,
        max_workers=3,
    )

    output_ids = [
        json.loads(line)["id"]
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert summary.passed == 6
    assert peak == 3
    assert output_ids == [record["id"] for record in records]
