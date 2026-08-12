from __future__ import annotations

import pytest

from deepseek_distill.multisource_tasks import (
    MULTISOURCE_TASK_SCHEMA_VERSION,
    build_multisource_teacher_messages,
    make_multisource_task,
    multisource_dataset_slug,
)


def test_common_task_schema_builds_versioned_stdio_prompt_without_tests() -> None:
    task = make_multisource_task(
        task_id="fixture_train_1",
        source={
            "dataset": "fixture",
            "split": "train",
            "original_id": 1,
            "revision": "abc123",
            "license": "fixture-license",
            "provenance": "https://example.test/original",
            "mirror": "https://example.test/mirror",
        },
        problem_text="Read one integer and print it.",
        interface_type="stdin_stdout",
        required_interface=(
            "Complete Python program using standard input and standard output."
        ),
        tests=[{"input": "7\n", "output": "7\n"}],
        metadata={"reference_solution_count": 1},
    )

    messages = build_multisource_teacher_messages(task)

    assert task["schema_version"] == MULTISOURCE_TASK_SCHEMA_VERSION
    assert [message["role"] for message in messages] == ["system", "user"]
    assert task["problem_text"] in messages[1]["content"]
    assert task["tests"][0]["input"] not in repr(messages)
    assert "deepseek.python.clean.v2" not in repr(messages)


def test_common_task_schema_requires_function_identity_for_function_tasks() -> None:
    with pytest.raises(ValueError, match="function_name"):
        make_multisource_task(
            task_id="fixture_function_1",
            source={
                "dataset": "fixture",
                "split": "test",
                "original_id": 1,
                "revision": "abc123",
                "license": "fixture-license",
                "provenance": "https://example.test/original",
                "mirror": "https://example.test/mirror",
            },
            problem_text="Return the input.",
            interface_type="function",
            required_interface="identity(value)",
            tests=["assert identity(1) == 1"],
            metadata={},
        )


def test_open_r1_codeforces_has_a_stable_audit_slug() -> None:
    task = make_multisource_task(
        task_id="open_r1_codeforces_train_fixture",
        source={
            "dataset": "open-r1/codeforces",
            "split": "train",
            "original_id": "1_A",
            "revision": "fixture",
            "license": "CC-BY-4.0",
            "provenance": "https://huggingface.co/datasets/open-r1/codeforces",
            "mirror": "https://huggingface.co/datasets/open-r1/codeforces",
        },
        problem_text="Read one integer and print it.",
        interface_type="stdin_stdout",
        required_interface=(
            "Complete Python program using standard input and standard output."
        ),
        tests=[{"input": "7\n", "output": "7\n"}],
        metadata={},
    )

    assert multisource_dataset_slug(task) == "open_r1_codeforces"
