"""Standalone child-process runner for untrusted benchmark submissions."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import traceback
from pathlib import Path
from types import ModuleType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("compile", "import", "test"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = io.StringIO()
    errors = io.StringIO()
    tests_run = 0
    try:
        source = args.source.read_text(encoding="utf-8")
        task = json.loads(args.task.read_text(encoding="utf-8"))
        if args.phase == "compile":
            compile(source, str(args.source), "exec")
        elif args.phase == "import":
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                module = _import_submission(args.source)
            required = getattr(module, task["required_function_name"], None)
            if not callable(required):
                raise MissingFunctionError(
                    f"required function {task['required_function_name']!r} is not callable"
                )
        else:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                module = _import_submission(args.source)
                namespace = dict(vars(module))
                setup_code = task.get("test_setup_code") or ""
                if setup_code:
                    exec(compile(setup_code, "<mbpp-test-setup>", "exec"), namespace)
                for test_number, test in enumerate(task["tests"], start=1):
                    exec(compile(test, f"<mbpp-test-{test_number}>", "exec"), namespace)
                    tests_run = test_number
    except BaseException as error:  # child must durably classify generated-code failures
        status = _failure_status(args.phase, error)
        result = {
            "status": status,
            "stdout": output.getvalue(),
            "stderr": errors.getvalue(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(limit=20),
            "tests_run": tests_run,
        }
    else:
        result = {
            "status": "passed",
            "stdout": output.getvalue(),
            "stderr": errors.getvalue(),
            "error_type": None,
            "error_message": None,
            "traceback": None,
            "tests_run": tests_run,
        }
    args.result.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


class MissingFunctionError(RuntimeError):
    pass


def _import_submission(path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location("submission", path)
    if specification is None or specification.loader is None:
        raise ImportError("could not create a module specification")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _failure_status(phase: str, error: BaseException) -> str:
    if isinstance(error, MissingFunctionError):
        return "missing_function"
    if isinstance(error, SyntaxError):
        return "syntax_error"
    if phase == "import":
        return "import_error"
    if phase == "test" and isinstance(error, AssertionError):
        return "assertion_failure"
    if phase == "test":
        return "runtime_error"
    return "runtime_error"


if __name__ == "__main__":
    raise SystemExit(main())
