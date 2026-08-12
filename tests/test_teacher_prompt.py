from __future__ import annotations

import pytest

from deepseek_distill.teacher_prompt import (
    CLEAN_PROMPT_CONTRACT_ID,
    FUNCTION_INTERFACE,
    STDIN_STDOUT_INTERFACE,
    build_clean_teacher_messages,
    prompt_contract_metadata,
)


@pytest.mark.parametrize(
    ("interface_type", "required_interface", "implementation_line"),
    [
        (FUNCTION_INTERFACE, "identity(value)", "Implement the required function."),
        (
            STDIN_STDOUT_INTERFACE,
            "Complete Python program using standard input and standard output.",
            "Implement the required program.",
        ),
    ],
)
def test_clean_v2_prompt_contract_is_versioned_and_format_explicit(
    interface_type: str,
    required_interface: str,
    implementation_line: str,
) -> None:
    messages = build_clean_teacher_messages(
        task_id="fixture_1",
        problem_text="Return the input unchanged.",
        required_interface=required_interface,
        interface_type=interface_type,
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "begin directly with Python source code" in messages[0]["content"]
    assert "Do not use Markdown code fences" in messages[0]["content"]
    assert "Do not include comments or docstrings unless required" in messages[0]["content"]
    assert "End immediately after the final Python statement" in messages[0]["content"]
    assert "Do not emit chat control tokens" in messages[0]["content"]
    assert "Problem:\nReturn the input unchanged." in messages[1]["content"]
    assert f"Required interface:\n{required_interface}" in messages[1]["content"]
    assert implementation_line in messages[1]["content"]
    assert prompt_contract_metadata(interface_type) == {
        "id": CLEAN_PROMPT_CONTRACT_ID,
        "interface_type": interface_type,
    }


def test_clean_v2_prompt_rejects_missing_problem_before_request_construction() -> None:
    with pytest.raises(ValueError, match="actual problem statement"):
        build_clean_teacher_messages(
            task_id="fixture_1",
            problem_text=" ",
            required_interface="solve(value)",
            interface_type=FUNCTION_INTERFACE,
        )
