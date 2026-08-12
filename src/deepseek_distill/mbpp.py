"""Authoritative MBPP task ingestion and DeepSeek prompt construction.

The importer pins the official Google Research dataset mirror and reads only
the original MBPP training split (task IDs 601--974).  Benchmark tests and
reference code remain in the task record, but prompt construction has no code
path that reads either field.
"""

from __future__ import annotations

import ast
import builtins
import copy
import os
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .teacher_prompt import (
    FUNCTION_INTERFACE,
    FUNCTION_SYSTEM_MESSAGE as SYSTEM_MESSAGE,
    build_clean_teacher_messages,
)


MBPP_TASK_SCHEMA_VERSION = "coding.task.mbpp.v1"
MBPP_DATASET_ID = "google-research-datasets/mbpp"
MBPP_CONFIG = "full"
MBPP_SPLIT = "train"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MBPP_LICENSE = "CC-BY-4.0"
MBPP_PROVENANCE = "https://github.com/google-research/google-research/tree/master/mbpp"
MBPP_MIRROR = "https://huggingface.co/datasets/google-research-datasets/mbpp"
MBPP_TRAIN_ID_MIN = 601
MBPP_TRAIN_ID_MAX = 974

_DEF_PATTERN = re.compile(
    r"\bdef\s+([A-Za-z_]\w*)\s*\(([^\n()]*(?:\([^\n()]*\)[^\n()]*)*)\)\s*(?:->\s*[^:]+)?\s*:",
    re.MULTILINE,
)
_BUILTIN_NAMES = frozenset(dir(builtins))


@dataclass(frozen=True, slots=True)
class FunctionInterface:
    """A deterministic benchmark interface with provenance for each field."""

    function_name: str
    function_signature: str | None
    name_source: str
    signature_source: str | None


def extract_function_interface(
    *,
    problem_text: str,
    tests: Sequence[str],
    reference_code: str | None,
    allow_reference_code: bool = True,
) -> FunctionInterface:
    """Extract a required function without inventing parameter names.

    The precedence is an explicit ``def`` in the problem, calls in benchmark
    tests, and finally licensed reference code.  Test calls establish a name,
    but positional call values are not enough to infer parameter names; the
    signature is therefore omitted unless an explicit or reference definition
    supplies it (or every observed call has no arguments).
    """
    if not isinstance(problem_text, str):
        raise ValueError("problem_text must be a string")
    _validate_string_sequence(tests, "tests")
    if reference_code is not None and not isinstance(reference_code, str):
        raise ValueError("reference_code must be a string or null")

    explicit = _definitions_from_text(problem_text)
    if len(explicit) > 1:
        raise ValueError("ambiguous explicit function definitions in problem statement")
    if explicit:
        name, signature = explicit[0]
        return FunctionInterface(name, signature, "problem_text", "problem_text")

    calls = _called_names(tests)
    reference_definitions = (
        _definitions_from_python(reference_code or "") if allow_reference_code else []
    )
    if calls:
        counts = Counter(call.name for call in calls)
        highest = max(counts.values())
        candidates = sorted(name for name, count in counts.items() if count == highest)
        if len(candidates) != 1:
            raise ValueError(
                "ambiguous function calls in benchmark tests: " + ", ".join(candidates)
            )
        name = candidates[0]
        matching_reference = [
            signature for candidate, signature in reference_definitions if candidate == name
        ]
        if len(matching_reference) > 1:
            raise ValueError(f"ambiguous reference definitions for {name}")
        if matching_reference:
            return FunctionInterface(name, matching_reference[0], "tests", "reference_code")
        name_calls = [call for call in calls if call.name == name]
        if name_calls and all(not call.has_arguments for call in name_calls):
            return FunctionInterface(name, f"{name}()", "tests", "tests")
        return FunctionInterface(name, None, "tests", None)

    if allow_reference_code:
        if len(reference_definitions) > 1:
            raise ValueError("ambiguous function definitions in reference code")
        if reference_definitions:
            name, signature = reference_definitions[0]
            return FunctionInterface(name, signature, "reference_code", "reference_code")
    raise ValueError("could not determine a required function name")


def import_mbpp_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int = 20,
    selection: Literal["first", "random"] = "first",
    seed: int = 0,
    revision: str = MBPP_REVISION,
) -> list[dict[str, Any]]:
    """Convert official full/train rows into versioned internal task records."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if selection not in {"first", "random"}:
        raise ValueError("selection must be 'first' or 'random'")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")

    indexed: list[tuple[int, Mapping[str, Any]]] = []
    seen_ids: set[int] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"MBPP row {position} must be an object")
        task_id = _task_id(row.get("task_id"), position)
        if not MBPP_TRAIN_ID_MIN <= task_id <= MBPP_TRAIN_ID_MAX:
            raise ValueError(
                f"MBPP task {task_id} is not in the official MBPP training range "
                f"{MBPP_TRAIN_ID_MIN}--{MBPP_TRAIN_ID_MAX}"
            )
        if task_id in seen_ids:
            raise ValueError(f"duplicate MBPP task_id {task_id}")
        seen_ids.add(task_id)
        indexed.append((task_id, row))
    indexed.sort(key=lambda item: item[0])
    if limit > len(indexed):
        raise ValueError(f"requested {limit} MBPP tasks but only {len(indexed)} rows are available")

    if selection == "first":
        selected = indexed[:limit]
    else:
        selected = random.Random(seed).sample(indexed, limit)
    return [_build_task(task_id, row, revision=revision) for task_id, row in selected]


def load_mbpp_tasks(
    *,
    limit: int = 20,
    selection: Literal["first", "random"] = "first",
    seed: int = 0,
    revision: str = MBPP_REVISION,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Download the pinned official mirror and import only ``full/train``."""
    if cache_dir is not None:
        # huggingface_hub resolves HF_HOME at import time, before datasets uses
        # its own cache_dir.  Set it here so an explicit cache path controls
        # both layers in a fresh CLI process.
        os.environ["HF_HOME"] = str(Path(cache_dir).resolve())
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("MBPP import requires the optional 'datasets' dependency") from error

    dataset = load_dataset(
        MBPP_DATASET_ID,
        MBPP_CONFIG,
        split=MBPP_SPLIT,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return import_mbpp_rows(
        dataset,
        limit=limit,
        selection=selection,
        seed=seed,
        revision=revision,
    )


def build_teacher_messages(task: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build exactly one system and one user message without reading tests."""
    if not isinstance(task, Mapping):
        raise ValueError("task must be an object")
    task_id = task.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task.id must be a non-empty string")
    prompt_task_id = task.get("problem_id", task_id)
    if not isinstance(prompt_task_id, str) or not prompt_task_id.strip():
        raise ValueError("task.problem_id must be a non-empty string when provided")
    problem_text = task.get("problem_text")
    if not isinstance(problem_text, str) or not problem_text.strip():
        raise ValueError("teacher request requires an actual problem statement")
    function_name = task.get("function_name")
    if not isinstance(function_name, str) or not function_name.strip():
        raise ValueError("teacher request requires a deterministically extracted function name")
    signature = task.get("function_signature")
    if signature is not None and (not isinstance(signature, str) or not signature.strip()):
        raise ValueError("function_signature must be a non-empty string or null")
    function_interface = signature.strip() if isinstance(signature, str) else function_name.strip()
    supporting_interfaces = task.get("supporting_interfaces") or []
    if not isinstance(supporting_interfaces, list) or any(
        not isinstance(item, str) or not item.strip() for item in supporting_interfaces
    ):
        raise ValueError("supporting_interfaces must contain non-empty strings")
    if supporting_interfaces:
        function_interface += "\nSupporting interfaces:\n" + "\n".join(
            supporting_interfaces
        )
    return build_clean_teacher_messages(
        task_id=prompt_task_id,
        problem_text=problem_text,
        required_interface=function_interface,
        interface_type=FUNCTION_INTERFACE,
    )


def _build_task(
    task_id: int,
    row: Mapping[str, Any],
    *,
    revision: str,
) -> dict[str, Any]:
    problem_text = _required_text(row.get("text"), f"MBPP task {task_id}.text")
    reference_code = _required_text(row.get("code"), f"MBPP task {task_id}.code")
    tests = _string_list(row.get("test_list"), f"MBPP task {task_id}.test_list")
    test_setup_code = row.get("test_setup_code")
    if test_setup_code is None:
        test_setup_code = ""
    if not isinstance(test_setup_code, str):
        raise ValueError(f"MBPP task {task_id}.test_setup_code must be a string")
    challenge_tests = _string_list(
        row.get("challenge_test_list") or [],
        f"MBPP task {task_id}.challenge_test_list",
    )
    interface = extract_function_interface(
        problem_text=problem_text,
        tests=tests,
        reference_code=reference_code,
        allow_reference_code=True,
    )
    return {
        "schema_version": MBPP_TASK_SCHEMA_VERSION,
        "id": f"mbpp_{task_id}",
        "source": {
            "dataset": "MBPP",
            "config": MBPP_CONFIG,
            "split": MBPP_SPLIT,
            "original_id": task_id,
            "revision": revision,
            "license": MBPP_LICENSE,
            "provenance": MBPP_PROVENANCE,
            "mirror": MBPP_MIRROR,
        },
        "problem_text": problem_text,
        "function_name": interface.function_name,
        "function_signature": interface.function_signature,
        "supporting_interfaces": _supporting_interfaces(
            tests,
            reference_code,
            required_function_name=interface.function_name,
        ),
        "tests": tests,
        "metadata": {
            "reference_code": reference_code,
            "test_setup_code": test_setup_code,
            "challenge_tests": challenge_tests,
            "interface_extraction": {
                "name_source": interface.name_source,
                "signature_source": interface.signature_source,
            },
        },
    }


@dataclass(frozen=True, slots=True)
class _ObservedCall:
    name: str
    has_arguments: bool


def _called_names(tests: Sequence[str]) -> list[_ObservedCall]:
    calls: list[_ObservedCall] = []
    for position, test in enumerate(tests):
        try:
            tree = ast.parse(test, filename=f"<mbpp-test-{position}>")
        except SyntaxError as error:
            raise ValueError(f"benchmark test {position} is not valid Python") from error
        for statement in tree.body:
            root = statement.test if isinstance(statement, ast.Assert) else statement
            calls.extend(_outer_candidate_calls(root))
    return calls


def _supporting_interfaces(
    tests: Sequence[str],
    reference_code: str,
    *,
    required_function_name: str,
) -> list[str]:
    helper_names: set[str] = set()
    for position, test in enumerate(tests):
        try:
            tree = ast.parse(test, filename=f"<mbpp-test-{position}>")
        except SyntaxError as error:
            raise ValueError(f"benchmark test {position} is not valid Python") from error
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == required_function_name
            ):
                continue
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
                for nested in ast.walk(argument):
                    if (
                        isinstance(nested, ast.Call)
                        and isinstance(nested.func, ast.Name)
                        and nested.func.id not in _BUILTIN_NAMES
                        and nested.func.id != required_function_name
                    ):
                        helper_names.add(nested.func.id)

    try:
        reference_tree = ast.parse(reference_code)
    except SyntaxError:
        reference_tree = ast.Module(body=[], type_ignores=[])
    interfaces: list[str] = []
    for helper_name in sorted(helper_names):
        matching = [
            node
            for node in reference_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == helper_name
        ]
        if len(matching) != 1:
            interfaces.append(helper_name)
            continue
        definition = matching[0]
        if isinstance(definition, ast.ClassDef):
            initializers = [
                node
                for node in definition.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__init__"
            ]
            if len(initializers) == 1:
                arguments = copy.deepcopy(initializers[0].args)
                if arguments.posonlyargs:
                    arguments.posonlyargs.pop(0)
                elif arguments.args:
                    arguments.args.pop(0)
                interfaces.append(f"{helper_name}({ast.unparse(arguments)})")
                continue
        elif isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
            interfaces.append(f"{helper_name}({ast.unparse(definition.args)})")
            continue
        interfaces.append(helper_name)
    return interfaces


def _outer_candidate_calls(node: ast.AST) -> list[_ObservedCall]:
    """Find assertion-subject calls without counting calls inside arguments."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id not in _BUILTIN_NAMES:
            return [_ObservedCall(node.func.id, bool(node.args or node.keywords))]
        nested: list[_ObservedCall] = []
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            nested.extend(_outer_candidate_calls(argument))
        return nested
    calls: list[_ObservedCall] = []
    for child in ast.iter_child_nodes(node):
        calls.extend(_outer_candidate_calls(child))
    return calls


def _definitions_from_text(text: str) -> list[tuple[str, str]]:
    definitions = _definitions_from_python(text)
    if definitions:
        return definitions
    parsed: list[tuple[str, str]] = []
    for match in _DEF_PATTERN.finditer(text):
        name, arguments = match.groups()
        snippet = f"def {name}({arguments}):\n    pass\n"
        definitions = _definitions_from_python(snippet)
        if definitions:
            parsed.append(definitions[0])
    return parsed


def _definitions_from_python(source: str) -> list[tuple[str, str]]:
    if not source.strip():
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        (node.name, f"{node.name}({ast.unparse(node.args)})")
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _task_id(value: Any, position: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"MBPP row {position}.task_id must be an integer")
    return value


def _required_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    _validate_string_sequence(value, context)
    return list(value)


def _validate_string_sequence(value: Sequence[str], context: str) -> None:
    if isinstance(value, (str, bytes)) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{context} must contain non-empty strings")
