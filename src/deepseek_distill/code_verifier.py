"""Conservative extraction and isolated execution of benchmark solutions."""

from __future__ import annotations

import ast
import copy
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .durable_io import open_text_append
from .offline_teacher import OfflineTeacherTraceProvider, TeacherTraceError


VERIFIER_SCHEMA_VERSION = "coding.verifier.mbpp.v1"
TACO_VERIFIER_SCHEMA_VERSION = "coding.verifier.taco.v1"
_FENCE_PREFIXES = ("```python", "```py", "```")
_FORBIDDEN_CALLS = frozenset(
    {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}
)
_FORBIDDEN_MODULES = frozenset(
    {
        "asyncio",
        "ftplib",
        "http",
        "multiprocessing",
        "os",
        "pathlib",
        "shutil",
        "smtplib",
        "socket",
        "subprocess",
        "tempfile",
        "urllib",
        "webbrowser",
    }
)


class SourceExtractionError(ValueError):
    """A conservative extraction rejection with a durable category."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True, slots=True)
class ExtractedSource:
    source: str
    removed_markdown_fence: bool


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    total: int
    skipped: int
    passed: int
    failed: int
    failure_counts: dict[str, int]


def extract_python_source(
    response_text: str,
    *,
    required_function_name: str,
) -> ExtractedSource:
    """Accept plain Python or exactly one all-enclosing Markdown fence."""
    if not isinstance(response_text, str) or not response_text.strip():
        raise SourceExtractionError("extraction_error", "teacher response is empty")
    if not isinstance(required_function_name, str) or not required_function_name:
        raise SourceExtractionError("missing_function", "required function name is missing")

    candidate = response_text
    removed_fence = False
    if "```" in response_text:
        candidate = _unwrap_single_fence(response_text)
        removed_fence = True
    try:
        tree = ast.parse(candidate, filename="<teacher-response>")
    except SyntaxError as error:
        location = f"line {error.lineno}" if error.lineno is not None else "unknown line"
        raise SourceExtractionError(
            "syntax_error", f"teacher source is not valid Python at {location}: {error.msg}"
        ) from error

    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == required_function_name
    ]
    if not definitions:
        raise SourceExtractionError(
            "missing_function",
            f"source does not define required module-level function {required_function_name!r}",
        )
    if len(definitions) > 1:
        raise SourceExtractionError(
            "extraction_error",
            f"source defines required function {required_function_name!r} more than once",
        )
    _reject_module_scope_call(tree, required_function_name)
    _reject_forbidden_operations(tree)
    return ExtractedSource(candidate, removed_fence)


def extract_python_program_source(response_text: str) -> ExtractedSource:
    """Extract a complete stdin/stdout program without requiring a function."""
    if not isinstance(response_text, str) or not response_text.strip():
        raise SourceExtractionError("extraction_error", "teacher response is empty")
    candidate = response_text
    removed_fence = False
    if "```" in response_text:
        candidate = _unwrap_single_fence(response_text)
        removed_fence = True
    try:
        tree = ast.parse(candidate, filename="<teacher-response>")
    except SyntaxError as error:
        location = f"line {error.lineno}" if error.lineno is not None else "unknown line"
        raise SourceExtractionError(
            "syntax_error", f"teacher source is not valid Python at {location}: {error.msg}"
        ) from error
    _reject_forbidden_operations(tree, allow_input=True)
    return ExtractedSource(candidate, removed_fence)


def verify_normalized_record(
    record: Mapping[str, Any],
    *,
    phase_timeout_seconds: float = 5.0,
    max_output_characters: int = 65_536,
) -> dict[str, Any]:
    """Verify one normalized trace without executing its source in this process."""
    if (
        isinstance(phase_timeout_seconds, bool)
        or not isinstance(phase_timeout_seconds, (int, float))
        or not math.isfinite(float(phase_timeout_seconds))
        or phase_timeout_seconds <= 0
    ):
        raise ValueError("phase_timeout_seconds must be finite and positive")
    if max_output_characters <= 0:
        raise ValueError("max_output_characters must be positive")
    if not isinstance(record, Mapping):
        raise ValueError("record must be an object")

    record_id = record.get("id")
    task = record.get("task")
    task_copy = copy.deepcopy(dict(task)) if isinstance(task, Mapping) else None
    teacher_response = record.get("response_text")
    interface_type = task.get("interface_type") if isinstance(task, Mapping) else None
    result: dict[str, Any] = {
        "schema_version": (
            TACO_VERIFIER_SCHEMA_VERSION
            if interface_type == "stdin_stdout"
            else VERIFIER_SCHEMA_VERSION
        ),
        "id": record_id,
        "status": "rejected",
        "failure_category": None,
        "task": task_copy,
        "teacher_response": teacher_response if isinstance(teacher_response, str) else None,
        "extracted_source": None,
        "source_extraction": None,
        "trace_validation": {"valid": False, "error": None},
        "phases": [],
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "phase_timeout_seconds": float(phase_timeout_seconds),
        },
    }

    if not isinstance(record_id, str) or not record_id:
        return _reject(result, "malformed_trace", "normalized record id is missing")
    if not isinstance(task, Mapping) or task.get("id") != record_id:
        return _reject(result, "malformed_trace", "normalized record task metadata is missing")
    if interface_type == "stdin_stdout":
        return _verify_stdio_record(
            record,
            task=task,
            teacher_response=teacher_response,
            result=result,
            phase_timeout_seconds=float(phase_timeout_seconds),
            max_output_characters=max_output_characters,
        )
    function_name = task.get("function_name")
    tests = task.get("tests")
    metadata = task.get("metadata")
    if not isinstance(function_name, str) or not function_name:
        return _reject(result, "missing_function", "task function_name is missing")
    if not isinstance(tests, list) or any(not isinstance(test, str) for test in tests):
        return _reject(result, "malformed_trace", "task tests must be a list of strings")
    if not isinstance(metadata, Mapping):
        return _reject(result, "malformed_trace", "task metadata is missing")
    challenge_tests = metadata.get("challenge_tests") or []
    if not isinstance(challenge_tests, list) or any(
        not isinstance(test, str) for test in challenge_tests
    ):
        return _reject(result, "malformed_trace", "task challenge_tests must be strings")
    setup_code = metadata.get("test_setup_code") or ""
    if not isinstance(setup_code, str):
        return _reject(result, "malformed_trace", "task test_setup_code must be a string")

    try:
        OfflineTeacherTraceProvider().get_trace(record)
    except (TeacherTraceError, ValueError) as error:
        result["trace_validation"]["error"] = str(error)
        return _reject(result, "malformed_trace", str(error))
    result["trace_validation"] = {"valid": True, "error": None}

    try:
        extracted = extract_python_source(
            teacher_response,
            required_function_name=function_name,
        )
    except SourceExtractionError as error:
        result["source_extraction"] = {
            "status": "rejected",
            "removed_markdown_fence": False,
            "error": str(error),
        }
        return _reject(result, error.category, str(error))
    result["extracted_source"] = extracted.source
    result["source_extraction"] = {
        "status": "passed",
        "removed_markdown_fence": extracted.removed_markdown_fence,
        "error": None,
    }

    payload = {
        "required_function_name": function_name,
        "test_setup_code": setup_code,
        "tests": [*tests, *challenge_tests],
        "primary_test_count": len(tests),
        "challenge_test_count": len(challenge_tests),
    }
    with tempfile.TemporaryDirectory(prefix="deepseek-mbpp-verify-") as directory:
        working_directory = Path(directory)
        source_path = working_directory / "submission.py"
        payload_path = working_directory / "task.json"
        source_path.write_text(extracted.source, encoding="utf-8", newline="\n")
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
        for phase in ("compile", "import", "test"):
            phase_result = _run_phase(
                phase=phase,
                source_path=source_path,
                payload_path=payload_path,
                working_directory=working_directory,
                timeout_seconds=float(phase_timeout_seconds),
                max_output_characters=max_output_characters,
            )
            result["phases"].append(phase_result)
            if phase_result["status"] != "passed":
                return _reject(result, phase_result["status"], phase_result.get("error_message"))

    result["status"] = "accepted"
    result["failure_category"] = "passed"
    return result


def verify_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    phase_timeout_seconds: float = 5.0,
    max_output_characters: int = 65_536,
    max_workers: int = 1,
) -> VerificationSummary:
    """Append verification results for normalized IDs not already completed."""
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")
    seen_inputs: set[str] = set()
    for line_number, record in _read_jsonl_objects(input_path):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{input_path}:{line_number}: id must be a non-empty string")
        if record_id in seen_inputs:
            raise ValueError(f"{input_path}:{line_number}: duplicate input id {record_id!r}")
        seen_inputs.add(record_id)
    existing = _existing_verifier_ids(output_path)

    passed = 0
    failed = 0
    skipped = 0
    categories: Counter[str] = Counter()
    processed_inputs: set[str] = set()

    def pending_records() -> Iterator[dict[str, Any]]:
        nonlocal skipped
        for line_number, record in _read_jsonl_objects(input_path):
            record_id = record.get("id")
            if (
                not isinstance(record_id, str)
                or record_id not in seen_inputs
                or record_id in processed_inputs
            ):
                raise ValueError(
                    f"{input_path}:{line_number}: input changed during verification"
                )
            processed_inputs.add(record_id)
            if record_id in existing:
                skipped += 1
                continue
            yield record

    def verify_one(record: Mapping[str, Any]) -> dict[str, Any]:
        return verify_normalized_record(
            record,
            phase_timeout_seconds=phase_timeout_seconds,
            max_output_characters=max_output_characters,
        )

    for result in _bounded_ordered_map(
        verify_one,
        pending_records(),
        max_workers=max_workers,
    ):
        _append_jsonl(output_path, result)
        if result["failure_category"] == "passed":
            passed += 1
        else:
            failed += 1
            categories[result["failure_category"]] += 1
    if processed_inputs != seen_inputs:
        raise ValueError(f"{input_path}: input changed during verification")
    return VerificationSummary(
        total=len(seen_inputs),
        skipped=skipped,
        passed=passed,
        failed=failed,
        failure_counts=dict(sorted(categories.items())),
    )


def _bounded_ordered_map(
    function: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    max_workers: int,
) -> Iterator[Any]:
    """Run bounded parallel work while yielding results in input order."""
    if max_workers == 1:
        for item in items:
            yield function(item)
        return

    item_iterator = iter(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        active = deque()
        for _ in range(max_workers):
            try:
                item = next(item_iterator)
            except StopIteration:
                break
            active.append(executor.submit(function, item))
        while active:
            yield active.popleft().result()
            try:
                item = next(item_iterator)
            except StopIteration:
                continue
            active.append(executor.submit(function, item))


def _unwrap_single_fence(response_text: str) -> str:
    stripped = response_text.strip()
    matching_prefix = next(
        (prefix for prefix in _FENCE_PREFIXES if stripped.startswith(prefix + "\n")),
        None,
    )
    if matching_prefix is None or not stripped.endswith("```"):
        raise SourceExtractionError(
            "extraction_error", "Markdown fence must enclose the complete response"
        )
    if stripped.count("```") != 2:
        raise SourceExtractionError(
            "extraction_error", "response must contain exactly one Markdown code fence"
        )
    source = stripped[len(matching_prefix) + 1 : -3]
    if not source.strip():
        raise SourceExtractionError("extraction_error", "Markdown code fence is empty")
    return source


def _reject_module_scope_call(tree: ast.Module, required_function_name: str) -> None:
    visitor = _ModuleExecutionCallVisitor(required_function_name)
    for statement in tree.body:
        visitor.visit(statement)
    if visitor.found:
        raise SourceExtractionError(
            "forbidden_operation",
            f"required function {required_function_name!r} is called at module scope",
        )


class _ModuleExecutionCallVisitor(ast.NodeVisitor):
    """Visit expressions executed while defining a module, not function bodies."""

    def __init__(self, required_function_name: str) -> None:
        self.required_function_name = required_function_name
        self.found = False

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == self.required_function_name:
            self.found = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition_expressions(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_definition_expressions(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)

    def _visit_definition_expressions(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)


def _reject_forbidden_operations(tree: ast.Module, *, allow_input: bool = False) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _validate_import(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise SourceExtractionError(
                    "forbidden_operation", "relative imports are not allowed"
                )
            _validate_import(node.module or "")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FORBIDDEN_CALLS
            and not (allow_input and node.func.id == "input")
        ):
            raise SourceExtractionError(
                "forbidden_operation", f"call to {node.func.id!r} is not allowed"
            )


def _verify_stdio_record(
    record: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    teacher_response: Any,
    result: dict[str, Any],
    phase_timeout_seconds: float,
    max_output_characters: int,
) -> dict[str, Any]:
    tests = task.get("tests")
    metadata = task.get("metadata")
    if not isinstance(metadata, Mapping):
        return _reject(result, "malformed_trace", "task metadata is missing")
    if not isinstance(tests, list) or not tests:
        return _reject(result, "malformed_trace", "task tests must be a non-empty list")
    normalized_tests: list[tuple[str, tuple[str, ...]]] = []
    for test in tests:
        if not isinstance(test, Mapping):
            return _reject(result, "malformed_trace", "stdin/stdout test must be an object")
        test_input = test.get("input")
        expected_value = test.get("output")
        if isinstance(expected_value, str):
            expected_outputs = (expected_value,)
        elif (
            isinstance(expected_value, list)
            and expected_value
            and all(isinstance(item, str) for item in expected_value)
        ):
            expected_outputs = tuple(expected_value)
        else:
            expected_outputs = ()
        if not isinstance(test_input, str) or not expected_outputs:
            return _reject(
                result,
                "malformed_trace",
                "stdin/stdout test input must be a string and output must be "
                "a string or non-empty string list",
            )
        normalized_tests.append((test_input, expected_outputs))

    try:
        OfflineTeacherTraceProvider().get_trace(record)
    except (TeacherTraceError, ValueError) as error:
        result["trace_validation"]["error"] = str(error)
        return _reject(result, "malformed_trace", str(error))
    result["trace_validation"] = {"valid": True, "error": None}

    try:
        extracted = extract_python_program_source(teacher_response)
    except SourceExtractionError as error:
        result["source_extraction"] = {
            "status": "rejected",
            "removed_markdown_fence": False,
            "error": str(error),
        }
        return _reject(result, error.category, str(error))
    result["extracted_source"] = extracted.source
    result["source_extraction"] = {
        "status": "passed",
        "removed_markdown_fence": extracted.removed_markdown_fence,
        "error": None,
    }
    result["output_comparison"] = (
        "normalize_newlines_strip_outer_whitespace_any_expected_v1"
    )

    with tempfile.TemporaryDirectory(prefix="deepseek-taco-verify-") as directory:
        working_directory = Path(directory)
        source_path = working_directory / "submission.py"
        payload_path = working_directory / "task.json"
        source_path.write_text(extracted.source, encoding="utf-8", newline="\n")
        payload_path.write_text("{}", encoding="utf-8", newline="\n")
        compile_result = _run_phase(
            phase="compile",
            source_path=source_path,
            payload_path=payload_path,
            working_directory=working_directory,
            timeout_seconds=phase_timeout_seconds,
            max_output_characters=max_output_characters,
        )
        result["phases"].append(compile_result)
        if compile_result["status"] != "passed":
            return _reject(
                result,
                compile_result["status"],
                compile_result.get("error_message"),
            )
        for test_index, (test_input, expected_outputs) in enumerate(normalized_tests):
            phase_result = _run_stdio_case(
                test_index=test_index,
                source_path=source_path,
                test_input=test_input,
                expected_outputs=expected_outputs,
                working_directory=working_directory,
                timeout_seconds=phase_timeout_seconds,
                max_output_characters=max_output_characters,
            )
            result["phases"].append(phase_result)
            if phase_result["status"] != "passed":
                return _reject(
                    result,
                    phase_result["status"],
                    phase_result.get("error_message"),
                )

    result["status"] = "accepted"
    result["failure_category"] = "passed"
    return result


def _run_stdio_case(
    *,
    test_index: int,
    source_path: Path,
    test_input: str,
    expected_outputs: tuple[str, ...],
    working_directory: Path,
    timeout_seconds: float,
    max_output_characters: int,
) -> dict[str, Any]:
    phase = f"test_{test_index}"
    command = [sys.executable, "-I", str(source_path)]
    kwargs: dict[str, Any] = {
        "cwd": working_directory,
        "env": _sanitized_environment(working_directory),
        "input": test_input,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout_seconds,
        "check": False,
    }
    if os.name == "posix":
        kwargs["preexec_fn"] = _linux_resource_limiter(timeout_seconds)
    elif os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(command, **kwargs)
    except subprocess.TimeoutExpired as error:
        return {
            "name": phase,
            "status": "timeout",
            "returncode": None,
            "stdout": _bounded_text(error.stdout, max_output_characters),
            "stderr": _bounded_text(error.stderr, max_output_characters),
            "error_type": "TimeoutExpired",
            "error_message": f"{phase} exceeded {timeout_seconds:.3f} seconds",
        }
    stdout = _bounded_text(completed.stdout, max_output_characters)
    stderr = _bounded_text(completed.stderr, max_output_characters)
    if completed.returncode != 0:
        return {
            "name": phase,
            "status": "runtime_error",
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "error_type": "ChildProcessError",
            "error_message": "submitted program exited with a non-zero status",
        }
    actual_output = _normalize_stdio_output(completed.stdout)
    normalized_expected = {
        _normalize_stdio_output(expected_output)
        for expected_output in expected_outputs
    }
    if actual_output not in normalized_expected:
        return {
            "name": phase,
            "status": "assertion_failure",
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "error_type": "OutputMismatch",
            "error_message": "program output did not match expected benchmark output",
        }
    return {
        "name": phase,
        "status": "passed",
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error_type": None,
        "error_message": None,
    }


def _normalize_stdio_output(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _validate_import(module_name: str) -> None:
    root = module_name.partition(".")[0]
    if root in _FORBIDDEN_MODULES:
        raise SourceExtractionError(
            "forbidden_operation", f"import of {root!r} is not allowed"
        )
    if root not in sys.stdlib_module_names:
        raise SourceExtractionError(
            "forbidden_operation", f"external package import {root!r} is not allowed"
        )


def _run_phase(
    *,
    phase: str,
    source_path: Path,
    payload_path: Path,
    working_directory: Path,
    timeout_seconds: float,
    max_output_characters: int,
) -> dict[str, Any]:
    result_path = working_directory / f"{phase}.result.json"
    runner_path = Path(__file__).with_name("_verifier_child.py").resolve()
    command = [
        sys.executable,
        "-I",
        str(runner_path),
        "--phase",
        phase,
        "--source",
        str(source_path),
        "--task",
        str(payload_path),
        "--result",
        str(result_path),
    ]
    kwargs: dict[str, Any] = {
        "cwd": working_directory,
        "env": _sanitized_environment(working_directory),
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout_seconds,
        "check": False,
    }
    if os.name == "posix":
        kwargs["preexec_fn"] = _linux_resource_limiter(timeout_seconds)
    elif os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(command, **kwargs)
    except subprocess.TimeoutExpired as error:
        return {
            "name": phase,
            "status": "timeout",
            "returncode": None,
            "stdout": _bounded_text(error.stdout, max_output_characters),
            "stderr": _bounded_text(error.stderr, max_output_characters),
            "error_type": "TimeoutExpired",
            "error_message": f"{phase} phase exceeded {timeout_seconds:.3f} seconds",
        }

    process_stdout = _bounded_text(completed.stdout, max_output_characters)
    process_stderr = _bounded_text(completed.stderr, max_output_characters)
    if not result_path.exists():
        fallback = "syntax_error" if phase == "compile" else (
            "import_error" if phase == "import" else "runtime_error"
        )
        return {
            "name": phase,
            "status": fallback,
            "returncode": completed.returncode,
            "stdout": process_stdout,
            "stderr": process_stderr,
            "error_type": "ChildProcessError",
            "error_message": "verifier child exited without a result record",
        }
    try:
        child = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return {
            "name": phase,
            "status": "runtime_error",
            "returncode": completed.returncode,
            "stdout": process_stdout,
            "stderr": process_stderr,
            "error_type": type(error).__name__,
            "error_message": "verifier child wrote an invalid result record",
        }
    return {
        "name": phase,
        "status": child.get("status", "runtime_error"),
        "returncode": completed.returncode,
        "stdout": _bounded_text(child.get("stdout", ""), max_output_characters),
        "stderr": _bounded_text(child.get("stderr", ""), max_output_characters),
        "process_stdout": process_stdout,
        "process_stderr": process_stderr,
        "error_type": child.get("error_type"),
        "error_message": child.get("error_message"),
        "tests_run": child.get("tests_run", 0),
    }


def _sanitized_environment(working_directory: Path) -> dict[str, str]:
    environment = {
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "TEMP": str(working_directory),
        "TMP": str(working_directory),
    }
    for name in ("SYSTEMROOT", "WINDIR", "COMSPEC"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _linux_resource_limiter(timeout_seconds: float):
    def apply_limits() -> None:
        import resource

        limits = [
            (resource.RLIMIT_CPU, max(1, math.ceil(timeout_seconds) + 1)),
            (resource.RLIMIT_AS, 1_073_741_824),
            (resource.RLIMIT_FSIZE, 10_485_760),
            (resource.RLIMIT_NOFILE, 64),
        ]
        if hasattr(resource, "RLIMIT_NPROC"):
            limits.append((resource.RLIMIT_NPROC, 16))
        for resource_name, limit in limits:
            try:
                resource.setrlimit(resource_name, (limit, limit))
            except (OSError, ValueError):
                pass

    return apply_limits


def _bounded_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _reject(result: dict[str, Any], category: str, message: str | None) -> dict[str, Any]:
    result["status"] = "rejected"
    result["failure_category"] = category
    result["error"] = message
    return result


def _read_jsonl_objects(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL value must be an object")
            yield line_number, value


def _existing_verifier_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    identifiers: set[str] = set()
    for line_number, record in _read_jsonl_objects(path):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
        if record_id in identifiers:
            raise ValueError(f"{path}:{line_number}: duplicate verifier id {record_id!r}")
        identifiers.add(record_id)
    return identifiers


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with open_text_append(path) as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
