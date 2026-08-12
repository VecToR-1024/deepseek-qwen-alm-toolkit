from __future__ import annotations

import ast

import pytest

from scripts.recover_lcb_format import recover_code, recover_document


def generation(
    *,
    question_id: str = "task-1",
    platform: str = "atcoder",
    output_text: str,
    extracted_code: str = "",
) -> dict:
    return {
        "question_id": question_id,
        "platform": platform,
        "output_text": output_text,
        "extracted_code": extracted_code,
    }


def problem(
    *,
    question_id: str = "task-1",
    platform: str = "atcoder",
    starter_code: str = "",
    output_text: str = "",
    extracted_code: str = "",
) -> dict:
    return {
        "question_id": question_id,
        "platform": platform,
        "starter_code": starter_code,
        "output_list": [output_text],
        "code_list": [extracted_code],
    }


def test_preserves_nonempty_official_extraction() -> None:
    result = recover_code(
        generation(
            output_text="not Python outside a fence",
            extracted_code="print('official')",
        ),
        problem(),
        mode="interface_wrapper",
    )

    assert result.strategy == "official"
    assert result.code == "print('official')"


def test_raw_fallback_recovers_plain_python_without_fences() -> None:
    result = recover_code(
        generation(output_text="import sys\nprint(sys.stdin.read())\n"),
        problem(),
        mode="raw_fallback",
    )

    assert result.strategy == "raw_fallback"
    assert result.code == "import sys\nprint(sys.stdin.read())"
    ast.parse(result.code)


@pytest.mark.parametrize(
    "output_text",
    [
        "```python\nprint(1)\n",
        "print(1)\n```",
    ],
)
def test_recovers_one_accidental_boundary_fence(output_text: str) -> None:
    result = recover_code(
        generation(output_text=output_text),
        problem(),
        mode="raw_fallback",
    )

    assert result.strategy == "single_fence"
    assert result.code == "print(1)"


def test_rejects_empty_or_explanatory_non_python_output() -> None:
    empty = recover_code(
        generation(output_text="   \n"),
        problem(),
        mode="raw_fallback",
    )
    prose = recover_code(
        generation(output_text="Here is the solution:\nprint(1)"),
        problem(),
        mode="raw_fallback",
    )

    assert (empty.strategy, empty.code) == ("unrecovered", "")
    assert (prose.strategy, prose.code) == ("unrecovered", "")


def test_interface_wrapper_wraps_only_matching_leetcode_methods() -> None:
    raw = (
        "from bisect import bisect_left\n"
        "LIMIT = 3\n"
        "def solve(self, nums: list[int]) -> int:\n"
        "    return bisect_left(nums, LIMIT)\n"
        "def helper(value):\n"
        "    return value\n"
    )
    row = problem(
        platform="leetcode",
        starter_code=(
            "class Solution:\n"
            "    def solve(self, nums: list[int]) -> int:\n"
            "        "
        ),
    )

    result = recover_code(
        generation(platform="leetcode", output_text=raw),
        row,
        mode="interface_wrapper",
    )
    tree = ast.parse(result.code)

    assert result.strategy == "leetcode_interface_wrapper"
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    assert [node.name for node in classes] == ["Solution"]
    assert [node.name for node in classes[0].body] == ["solve"]
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "helper"
        for node in tree.body
    )


def test_interface_wrapper_keeps_existing_solution_class_unchanged() -> None:
    raw = "class Solution:\n    def solve(self, value):\n        return value\n"
    result = recover_code(
        generation(platform="leetcode", output_text=raw),
        problem(
            platform="leetcode",
            starter_code="class Solution:\n    def solve(self, value):\n        ",
        ),
        mode="interface_wrapper",
    )

    assert result.strategy == "raw_fallback"
    assert result.code == raw.strip()


def test_interface_wrapper_does_not_guess_a_different_method_name() -> None:
    result = recover_code(
        generation(
            platform="leetcode",
            output_text="def wrong(self, value):\n    return value\n",
        ),
        problem(
            platform="leetcode",
            starter_code="class Solution:\n    def expected(self, value):\n        ",
        ),
        mode="interface_wrapper",
    )

    assert result.strategy == "raw_fallback"
    assert "class Solution" not in result.code


def test_document_recovery_preserves_raw_outputs_and_reports_strategies() -> None:
    strict = [
        problem(
            question_id="a",
            output_text="print(1)",
            extracted_code="",
        ),
        problem(
            question_id="b",
            output_text="```python\nprint(2)\n```",
            extracted_code="print(2)",
        ),
    ]
    generations = [
        generation(question_id="a", output_text="print(1)"),
        generation(
            question_id="b",
            output_text="```python\nprint(2)\n```",
            extracted_code="print(2)",
        ),
    ]

    recovered, audit = recover_document(
        strict,
        generations,
        mode="raw_fallback",
    )

    assert [row["output_list"] for row in recovered] == [
        ["print(1)"],
        ["```python\nprint(2)\n```"],
    ]
    assert [row["code_list"] for row in recovered] == [
        ["print(1)"],
        ["print(2)"],
    ]
    assert audit["records"] == 2
    assert audit["strategy_counts"] == {
        "official": 1,
        "raw_fallback": 1,
    }
    assert audit["empty_after_recovery"] == 0
    assert audit["ast_parseable_after_recovery"] == 2


def test_document_recovery_rejects_mismatched_or_duplicate_ids() -> None:
    strict = [problem(question_id="a", output_text="print(1)")]

    with pytest.raises(ValueError, match="exact task IDs"):
        recover_document(
            strict,
            [generation(question_id="b", output_text="print(1)")],
            mode="raw_fallback",
        )

    with pytest.raises(ValueError, match="duplicate generation"):
        recover_document(
            strict,
            [
                generation(question_id="a", output_text="print(1)"),
                generation(question_id="a", output_text="print(1)"),
            ],
            mode="raw_fallback",
        )
