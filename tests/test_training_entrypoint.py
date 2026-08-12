from __future__ import annotations

import json
from pathlib import Path

import pytest
from transformers import TrainingArguments

from examples.train_offline_alm import (
    chat_template_kwargs_from_env,
    load_training_dataset,
    resume_checkpoint_from_env,
    warmup_from_env,
)


def write_trace(path: Path, record_id: str, challenge_tests: list[str]) -> None:
    record = {
        "schema_version": "deepseek.teacher.normalized.v1",
        "id": record_id,
        "request": {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"problem {record_id}"},
            ]
        },
        "response_text": "a",
        "content_tokens": [
            {
                "bytes": [97],
                "logprob": -0.1,
                "top_logprobs": [{"token": "a", "bytes": [97], "logprob": -0.1}],
            }
        ],
        "task": {"metadata": {"challenge_tests": challenge_tests}},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_training_loader_projects_alm_fields_before_arrow_inference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    write_trace(path, "first", [])
    write_trace(path, "second", ["assert later_string_field()"])

    dataset = load_training_dataset(path, limit=2)

    assert dataset["id"] == ["first", "second"]
    assert "task" not in dataset.column_names
    assert dataset[0]["content_tokens"] == [{"bytes": [97], "logprob": -0.1}]


def test_training_loader_applies_prefix_limit_and_rejects_invalid_limits(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    write_trace(path, "first", [])
    write_trace(path, "second", ["assert later_string_field()"])

    assert load_training_dataset(path, limit=1)["id"] == ["first"]
    with pytest.raises(ValueError, match="non-negative"):
        load_training_dataset(path, limit=-1)
    with pytest.raises(ValueError, match="only contains 2"):
        load_training_dataset(path, limit=3)


def test_training_loader_rejects_non_object_token_rows(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    record = {
        "schema_version": "deepseek.teacher.normalized.v1",
        "id": "bad",
        "request": {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "problem"},
            ]
        },
        "response_text": "a",
        "content_tokens": ["not-an-object"],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 1 content_tokens\[0\]"):
        load_training_dataset(path, limit=0)


def test_explicit_warmup_steps_take_precedence_over_deprecated_ratio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARMUP_RATIO", "0.05")
    monkeypatch.setenv("WARMUP_STEPS", "2")

    ratio, steps = warmup_from_env()
    arguments = TrainingArguments(
        output_dir=str(tmp_path / "output"),
        warmup_ratio=ratio,
        warmup_steps=steps,
    )

    assert arguments.get_warmup_steps(40) == 2


def test_warmup_ratio_remains_available_when_steps_are_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WARMUP_RATIO", "0.05")
    monkeypatch.delenv("WARMUP_STEPS", raising=False)

    assert warmup_from_env() == (0.05, 0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("false", None),
        ("0", None),
        ("true", True),
        ("1", True),
        ("/tmp/checkpoint-38", "/tmp/checkpoint-38"),
    ],
)
def test_resume_checkpoint_env_parsing(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
    expected: str | bool | None,
) -> None:
    if value is None:
        monkeypatch.delenv("RESUME_FROM_CHECKPOINT", raising=False)
    else:
        monkeypatch.setenv("RESUME_FROM_CHECKPOINT", value)

    assert resume_checkpoint_from_env() == expected


def test_chat_template_kwargs_from_env_accepts_json_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CHAT_TEMPLATE_KWARGS",
        '{"enable_thinking": false}',
    )

    assert chat_template_kwargs_from_env() == {"enable_thinking": False}


@pytest.mark.parametrize("value", ["[]", '"not-an-object"', "{bad json}"])
def test_chat_template_kwargs_from_env_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("CHAT_TEMPLATE_KWARGS", value)

    with pytest.raises(RuntimeError, match="CHAT_TEMPLATE_KWARGS"):
        chat_template_kwargs_from_env()
