"""Versioned clean-source prompt contract shared by coding datasets."""

from __future__ import annotations

from typing import Literal


CLEAN_PROMPT_CONTRACT_ID = "deepseek.python.clean.v2"
FUNCTION_INTERFACE = "function"
STDIN_STDOUT_INTERFACE = "stdin_stdout"
InterfaceType = Literal["function", "stdin_stdout"]

_COMMON_RULES = """- Your response must begin directly with Python source code.
- Return only valid Python source code.
- Do not use Markdown code fences.
- Do not include explanations, commentary, tests, or example usage.
- Do not include comments or docstrings unless required for correctness.
- End immediately after the final Python statement.
- Do not emit chat control tokens or literal end-of-response markers.
- You may use the Python standard library.
- Do not use external packages, network access, files, subprocesses, eval, or
  exec.
- The code must run under Python 3.10 or newer."""

FUNCTION_SYSTEM_MESSAGE = f"""You are generating a reference solution for a Python programming benchmark.

Write a correct, deterministic, and reasonably efficient solution that follows
the required function name and interface exactly.

Rules:
{_COMMON_RULES}
- Do not print anything unless the task explicitly requires printing.
- Do not read from stdin unless the task explicitly specifies stdin/stdout.
- Do not call the implemented function at module scope.
- Preserve the exact required function name, parameters, and return behavior.
- Hidden unit tests will import the generated module and call the required
  function directly."""

STDIN_STDOUT_SYSTEM_MESSAGE = f"""You are generating a reference solution for a Python programming benchmark.

Write a correct, deterministic, and reasonably efficient solution for the
given standard-input/standard-output problem.

Rules:
{_COMMON_RULES}
- Read the required input from standard input and write only the required
  answer to standard output.
- Do not hard-code sample inputs or outputs.
- Hidden benchmark tests will execute the complete program independently."""


def build_clean_teacher_messages(
    *,
    task_id: str,
    problem_text: str,
    required_interface: str,
    interface_type: InterfaceType,
) -> list[dict[str, str]]:
    """Build the exact clean-v2 system/user request without benchmark tests."""

    prompt_task_id = _required_text(task_id, "task_id")
    problem = _required_text(
        problem_text,
        "teacher request requires an actual problem statement",
        direct_error=True,
    )
    interface = _required_text(required_interface, "required_interface")
    system_message, implementation_noun = _interface_contract(interface_type)
    user_message = f"""Task ID: {prompt_task_id}

Problem:
{problem}

Required interface:
{interface}

Implement the required {implementation_noun}.

Return only the complete Python source code."""
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def prompt_contract_metadata(interface_type: InterfaceType) -> dict[str, str]:
    """Return the persisted identifier for an exact prompt variant."""

    _interface_contract(interface_type)
    return {
        "id": CLEAN_PROMPT_CONTRACT_ID,
        "interface_type": interface_type,
    }


def _interface_contract(interface_type: str) -> tuple[str, str]:
    if interface_type == FUNCTION_INTERFACE:
        return FUNCTION_SYSTEM_MESSAGE, "function"
    if interface_type == STDIN_STDOUT_INTERFACE:
        return STDIN_STDOUT_SYSTEM_MESSAGE, "program"
    raise ValueError(
        "interface_type must be 'function' or 'stdin_stdout'"
    )


def _required_text(
    value: str,
    label: str,
    *,
    direct_error: bool = False,
) -> str:
    if not isinstance(value, str) or not value.strip():
        if direct_error:
            raise ValueError(label)
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()
