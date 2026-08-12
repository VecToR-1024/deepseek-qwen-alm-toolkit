"""Safe English ODEX ingestion with an explicit test-split training exception."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .multisource_tasks import (
    build_multisource_teacher_messages,
    make_multisource_task,
    select_unique_tasks,
)


ODEX_DATASET_ID = "neulab/odex"
ODEX_CONFIG = "en"
ODEX_SPLIT = "test"
ODEX_REVISION = "c3741a66c486b1a23beefdf6c75b06dba288d4f9"
ODEX_LICENSE = "CC-BY-SA-4.0"
ODEX_PROVENANCE = "https://github.com/zorazrw/odex"
ODEX_MIRROR = "https://huggingface.co/datasets/neulab/odex"
ODEX_RAW_FILE = "en_test.jsonl"
ODEX_DEFAULT_SEED = 20260731
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

build_teacher_messages = build_multisource_teacher_messages


def import_odex_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    selection: str = "random",
    seed: int = ODEX_DEFAULT_SEED,
    revision: str = ODEX_REVISION,
) -> list[dict[str, Any]]:
    """Select standard-library-only English ODEX tasks with executable tests."""

    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")
    eligible: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"ODEX row {position} must be an object")
        original_id = _task_id(row.get("task_id"), position)
        task = _build_task(
            row,
            original_id=original_id,
            position=position,
            revision=revision,
        )
        if task is None or original_id in seen_ids:
            continue
        seen_ids.add(original_id)
        eligible.append(task)
    return select_unique_tasks(
        eligible,
        limit=limit,
        selection=selection,
        seed=seed,
    )


def load_odex_tasks(
    *,
    limit: int,
    selection: str = "random",
    seed: int = ODEX_DEFAULT_SEED,
    revision: str = ODEX_REVISION,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Download the pinned raw English JSONL without executing remote code."""

    if cache_dir is not None:
        os.environ["HF_HOME"] = str(Path(cache_dir).resolve())
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "ODEX import requires the optional 'data' dependencies"
        ) from error
    local_path = hf_hub_download(
        repo_id=ODEX_DATASET_ID,
        repo_type="dataset",
        filename=ODEX_RAW_FILE,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return import_odex_rows(
        _read_jsonl(Path(local_path)),
        limit=limit,
        selection=selection,
        seed=seed,
        revision=revision,
    )


def _build_task(
    row: Mapping[str, Any],
    *,
    original_id: int,
    position: int,
    revision: str,
) -> dict[str, Any] | None:
    entry_point = _required_text(
        row.get("entry_point"),
        f"ODEX row {position}.entry_point",
    )
    prompt = _required_text(row.get("prompt"), f"ODEX row {position}.prompt")
    canonical_solution = _required_text(
        row.get("canonical_solution"),
        f"ODEX row {position}.canonical_solution",
    )
    suffix = row.get("suffix") or ""
    if not isinstance(suffix, str):
        raise ValueError(f"ODEX row {position}.suffix must be a string")
    signature = _extract_signature(prompt, entry_point=entry_point, position=position)
    libraries = _string_list(
        row.get("library") or [],
        context=f"ODEX row {position}.library",
    )
    test_start = _required_text(
        row.get("test_start"),
        f"ODEX row {position}.test_start",
    )
    raw_tests = _string_list(
        row.get("test"),
        context=f"ODEX row {position}.test",
        require_nonempty=True,
    )
    harness = test_start.rstrip() + "".join(raw_tests)
    harness += f"\n\ncheck({entry_point})\n"
    reference_source = f"{prompt}{canonical_solution}{suffix}"
    if not (
        _libraries_are_permitted(libraries)
        and _source_is_permitted(reference_source)
        and _source_is_permitted(harness)
    ):
        return None

    return make_multisource_task(
        task_id=f"odex_en_test_{original_id}",
        source={
            "dataset": ODEX_DATASET_ID,
            "config": ODEX_CONFIG,
            "split": ODEX_SPLIT,
            "original_id": original_id,
            "revision": revision,
            "license": ODEX_LICENSE,
            "provenance": ODEX_PROVENANCE,
            "mirror": ODEX_MIRROR,
            "raw_file": ODEX_RAW_FILE,
        },
        problem_text=_required_text(
            row.get("intent"),
            f"ODEX row {position}.intent",
        ),
        interface_type="function",
        required_interface=signature,
        function_name=entry_point,
        function_signature=signature,
        tests=[harness],
        metadata={
            "libraries": libraries,
            "reference_solution_count": 1,
            "reference_solution_sha256": hashlib.sha256(
                reference_source.encode("utf-8")
            ).hexdigest(),
            "original_test_count": len(raw_tests),
            "test_setup_code": "",
            "challenge_tests": [],
            "benchmark_split_role": (
                "test_used_as_training_source_by_explicit_project_decision"
            ),
            "eligibility": {
                "function_interface_extracted_from_prompt": True,
                "standard_library_only": True,
                "forbidden_operations_excluded": True,
            },
        },
    )


def _extract_signature(prompt: str, *, entry_point: str, position: int) -> str:
    try:
        tree = ast.parse(prompt, filename=f"<odex-prompt-{position}>")
    except SyntaxError as original_error:
        if not prompt.rstrip().endswith(":"):
            raise ValueError(
                f"ODEX row {position}.prompt is not a parseable function signature"
            ) from original_error
        try:
            tree = ast.parse(
                prompt.rstrip() + "\n    pass\n",
                filename=f"<odex-prompt-{position}>",
            )
        except SyntaxError as error:
            raise ValueError(
                f"ODEX row {position}.prompt is not a parseable function signature"
            ) from error
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
    ]
    if len(definitions) != 1:
        raise ValueError(
            f"ODEX row {position}.prompt must define entry_point exactly once"
        )
    return f"{entry_point}({ast.unparse(definitions[0].args)})"


def _libraries_are_permitted(libraries: list[str]) -> bool:
    for library in libraries:
        root = library.partition(".")[0]
        if root in _FORBIDDEN_MODULES or root not in sys.stdlib_module_names:
            return False
    return True


def _source_is_permitted(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(not _import_is_permitted(alias.name) for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if node.level or not _import_is_permitted(node.module or ""):
                return False
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FORBIDDEN_CALLS
        ):
            return False
    return True


def _import_is_permitted(module: str) -> bool:
    root = module.partition(".")[0]
    return root in sys.stdlib_module_names and root not in _FORBIDDEN_MODULES


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield value


def _task_id(value: Any, position: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"ODEX row {position}.task_id must be a non-negative integer")
    return value


def _string_list(
    value: Any,
    *,
    context: str,
    require_nonempty: bool = False,
) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{context} must be a list of strings")
    if require_nonempty and (not value or any(not item.strip() for item in value)):
        raise ValueError(f"{context} must contain non-empty strings")
    return list(value)


def _required_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value
