from __future__ import annotations

import random

import pytest

from deepseek_distill.mbpp import (
    MBPP_REVISION,
    MBPP_TASK_SCHEMA_VERSION,
    build_teacher_messages,
    extract_function_interface,
    import_mbpp_rows,
)


def mbpp_row(task_id: int, *, text: str | None = None) -> dict:
    return {
        "task_id": task_id,
        "text": text or f"Return the input value for task {task_id}.",
        "code": "def identity(value):\n    return value",
        "test_list": ["assert identity(1) == 1", "assert identity(value=2) == 2"],
        "test_setup_code": "",
        "challenge_test_list": ["assert identity(999) == 999"],
    }


def test_import_mbpp_preserves_authoritative_fields_and_separates_tests() -> None:
    task = import_mbpp_rows([mbpp_row(601)], limit=1)[0]

    assert task["schema_version"] == MBPP_TASK_SCHEMA_VERSION
    assert task["id"] == "mbpp_601"
    assert task["source"] == {
        "dataset": "MBPP",
        "config": "full",
        "split": "train",
        "original_id": 601,
        "revision": MBPP_REVISION,
        "license": "CC-BY-4.0",
        "provenance": "https://github.com/google-research/google-research/tree/master/mbpp",
        "mirror": "https://huggingface.co/datasets/google-research-datasets/mbpp",
    }
    assert task["problem_text"] == "Return the input value for task 601."
    assert task["tests"] == [
        "assert identity(1) == 1",
        "assert identity(value=2) == 2",
    ]
    assert task["metadata"]["reference_code"].startswith("def identity")
    assert task["metadata"]["challenge_tests"] == ["assert identity(999) == 999"]


def test_first_twenty_selection_is_train_only_and_order_stable() -> None:
    rows = [mbpp_row(task_id) for task_id in range(640, 600, -1)]

    first = import_mbpp_rows(rows, limit=20, selection="first", seed=123)
    repeated = import_mbpp_rows(rows, limit=20, selection="first", seed=999)

    assert [task["source"]["original_id"] for task in first] == list(range(601, 621))
    assert [task["id"] for task in repeated] == [task["id"] for task in first]
    assert all(task["source"]["split"] == "train" for task in first)


def test_seeded_random_selection_is_deterministic() -> None:
    rows = [mbpp_row(task_id) for task_id in range(601, 651)]

    first = import_mbpp_rows(rows, limit=12, selection="random", seed=17)
    repeated = import_mbpp_rows(rows, limit=12, selection="random", seed=17)
    different = import_mbpp_rows(rows, limit=12, selection="random", seed=18)

    assert [task["id"] for task in first] == [task["id"] for task in repeated]
    assert [task["id"] for task in first] != [task["id"] for task in different]


def test_seeded_random_selection_preserves_random_sample_order() -> None:
    rows = [mbpp_row(task_id) for task_id in range(601, 651)]
    expected_ids = random.Random(20260721).sample(list(range(601, 651)), 12)

    selected = import_mbpp_rows(
        rows,
        limit=12,
        selection="random",
        seed=20260721,
    )

    assert [task["source"]["original_id"] for task in selected] == expected_ids


def test_import_rejects_non_training_ids() -> None:
    with pytest.raises(ValueError, match="not in the official MBPP training range"):
        import_mbpp_rows([mbpp_row(510)], limit=1)


def test_interface_extraction_uses_single_called_function_and_reference_signature() -> None:
    result = extract_function_interface(
        problem_text="Return the largest value.",
        tests=["assert largest([1, 2]) == 2"],
        reference_code="def largest(values):\n    return max(values)",
    )

    assert result.function_name == "largest"
    assert result.function_signature == "largest(values)"
    assert result.name_source == "tests"
    assert result.signature_source == "reference_code"


def test_interface_extraction_handles_repeated_calls() -> None:
    result = extract_function_interface(
        problem_text="Return an incremented number.",
        tests=[
            "assert increment(1) == 2 and increment(2) == 3",
            "assert increment(10) == 11",
        ],
        reference_code="def increment(number):\n    return number + 1",
    )

    assert result.function_name == "increment"
    assert result.function_signature == "increment(number)"


def test_interface_extraction_ignores_nested_helper_constructors() -> None:
    result = extract_function_interface(
        problem_text="Find the longest chain.",
        tests=[
            "assert max_chain_length([Pair(1, 2), Pair(3, 4)], 2) == 2",
        ],
        reference_code=(
            "class Pair:\n"
            "    def __init__(self, a, b):\n"
            "        self.a, self.b = a, b\n"
            "def max_chain_length(values, count):\n"
            "    return count\n"
        ),
    )

    assert result.function_name == "max_chain_length"
    assert result.function_signature == "max_chain_length(values, count)"


def test_import_records_nested_helper_interface_without_leaking_tests() -> None:
    row = {
        "task_id": 601,
        "text": "Find the longest chain of pairs.",
        "code": (
            "class Pair:\n"
            "    def __init__(self, a, b):\n"
            "        self.a, self.b = a, b\n"
            "def max_chain_length(values, count):\n"
            "    return count\n"
        ),
        "test_list": [
            "assert max_chain_length([Pair(1, 2), Pair(3, 4)], 2) == 2"
        ],
        "test_setup_code": "",
        "challenge_test_list": [],
    }

    task = import_mbpp_rows([row], limit=1)[0]
    messages = build_teacher_messages(task)

    assert task["function_signature"] == "max_chain_length(values, count)"
    assert task["supporting_interfaces"] == ["Pair(a, b)"]
    assert "Pair(a, b)" in messages[1]["content"]
    assert row["test_list"][0] not in repr(messages)


def test_interface_extraction_does_not_invent_parameters_from_calls() -> None:
    result = extract_function_interface(
        problem_text="Combine values.",
        tests=[
            "assert combine(1, 2) == 3",
            "assert combine(left=4, right=5) == 9",
        ],
        reference_code=None,
    )

    assert result.function_name == "combine"
    assert result.function_signature is None
    assert result.signature_source is None


@pytest.mark.parametrize(
    ("tests", "expected_error"),
    [
        (["assert alpha(1) == beta(1)"], "ambiguous"),
        (["assert 1 + 1 == 2"], "could not determine"),
    ],
)
def test_interface_extraction_rejects_ambiguous_or_missing_function(
    tests: list[str], expected_error: str
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        extract_function_interface(
            problem_text="Compute a value.",
            tests=tests,
            reference_code=None,
        )


def test_teacher_messages_are_distinct_and_do_not_leak_tests_or_reference_code() -> None:
    task = import_mbpp_rows([mbpp_row(601)], limit=1)[0]

    messages = build_teacher_messages(task)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Task ID: mbpp_601" in messages[1]["content"]
    assert task["problem_text"] in messages[1]["content"]
    assert "identity(value)" in messages[1]["content"]
    for hidden_text in [
        *task["tests"],
        *task["metadata"]["challenge_tests"],
        task["metadata"]["reference_code"],
    ]:
        assert hidden_text not in repr(messages)


def test_missing_problem_text_fails_before_a_request_can_be_built() -> None:
    task = import_mbpp_rows([mbpp_row(601)], limit=1)[0]
    task["problem_text"] = "   "

    with pytest.raises(ValueError, match="actual problem statement"):
        build_teacher_messages(task)
