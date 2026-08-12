import json

import pytest

from deepseek_distill.api import GenerationConfig, build_error_record
from deepseek_distill.normalize import normalize_jsonl
from deepseek_distill.records import RAW_SCHEMA_VERSION
from deepseek_distill.validate import validate_jsonl


_DEFAULT_BYTES = object()


def raw_success(
    record_id: str,
    *,
    token_bytes: list[int] | None | object = _DEFAULT_BYTES,
) -> dict:
    token_bytes = [79, 75] if token_bytes is _DEFAULT_BYTES else token_bytes
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "id": record_id,
        "status": "ok",
        "collected_at": "2026-07-20T00:00:00Z",
        "request": {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "Reply OK"}],
            "generation_config": GenerationConfig().as_metadata(),
        },
        "response": {
            "id": f"response-{record_id}",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "OK", "reasoning_content": None},
                    "logprobs": {
                        "content": [
                            {
                                "token": "OK",
                                "bytes": token_bytes,
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": "OK", "bytes": token_bytes, "logprob": -0.1},
                                ],
                            }
                        ],
                        "reasoning_content": None,
                    },
                }
            ],
            "usage": {"total_tokens": 5},
        },
    }


def write_lines(path, values: list[dict | str]) -> None:
    lines = [value if isinstance(value, str) else json.dumps(value, ensure_ascii=False) for value in values]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_normalize_jsonl_skips_api_errors_and_atomically_writes_successes(tmp_path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    normalized_path = tmp_path / "normalized.jsonl"
    api_error = build_error_record(
        record_id="bad-api",
        messages=[{"role": "user", "content": "Reply OK"}],
        config=GenerationConfig(),
        error=RuntimeError("offline"),
        collected_at="2026-07-20T00:00:00Z",
    )
    write_lines(raw_path, [raw_success("good"), api_error])

    summary = normalize_jsonl(raw_path, normalized_path)

    assert summary.total == 2
    assert summary.normalized == 1
    assert summary.api_errors == 1
    assert summary.warnings == 0
    assert read_jsonl(normalized_path)[0]["id"] == "good"


def test_normalize_jsonl_does_not_replace_existing_output_without_force(tmp_path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    normalized_path = tmp_path / "normalized.jsonl"
    write_lines(raw_path, [raw_success("good")])
    normalized_path.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        normalize_jsonl(raw_path, normalized_path)

    assert normalized_path.read_text(encoding="utf-8") == "keep me"


def test_validate_jsonl_reports_malformed_json_and_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "raw.jsonl"
    record = raw_success("duplicate")
    write_lines(path, [record, "{not-json", record])

    report = validate_jsonl(path)

    assert report.total_lines == 3
    assert report.valid_records == 1
    assert report.invalid_records == 2
    assert {issue.code for issue in report.issues} == {"invalid_json", "duplicate_id"}


def test_validate_jsonl_counts_api_errors_and_byte_warnings(tmp_path) -> None:
    path = tmp_path / "raw.jsonl"
    api_error = build_error_record(
        record_id="api-error",
        messages=[{"role": "user", "content": "Reply OK"}],
        config=GenerationConfig(),
        error=RuntimeError("offline"),
    )
    write_lines(path, [raw_success("warning", token_bytes=None), api_error])

    report = validate_jsonl(path)

    assert report.valid_records == 2
    assert report.invalid_records == 0
    assert report.api_errors == 1
    assert report.warnings == 1
