"""Safe xCodeEval compact ingestion with hidden tests kept outside prompts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .multisource_tasks import (
    build_multisource_teacher_messages,
    make_multisource_task,
    select_unique_tasks,
)


XCODEEVAL_DATASET_ID = "NTU-NLP-sg/xCodeEval"
XCODEEVAL_CONFIG = "program_synthesis"
XCODEEVAL_SPLIT = "compact"
XCODEEVAL_RAW_SPLIT = "validation"
XCODEEVAL_REVISION = "aa09386604075a3e80bcdb315dd68bcc117dedc0"
XCODEEVAL_LICENSE = (
    "CC-BY-NC-4.0 in pinned loader; Hugging Face dataset card says CC-BY-4.0"
)
XCODEEVAL_PROVENANCE = "https://github.com/ntunlp/xCodeEval"
XCODEEVAL_MIRROR = "https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval"
XCODEEVAL_COMPACT_FILE = (
    "program_synthesis/validation/prog_syn_val.jsonl"
)
XCODEEVAL_UNIT_TEST_FILE = "unittest_db.json"
XCODEEVAL_DEFAULT_SEED = 20260731
_STDIO_INTERFACE = "Complete Python program using standard input and standard output."

build_teacher_messages = build_multisource_teacher_messages


def import_xcodeeval_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    unit_test_db: Mapping[str, Any],
    limit: int,
    selection: str = "random",
    seed: int = XCODEEVAL_DEFAULT_SEED,
    revision: str = XCODEEVAL_REVISION,
) -> list[dict[str, Any]]:
    """Build unique compact problems with official execution tests."""

    if not isinstance(unit_test_db, Mapping):
        raise ValueError("unit_test_db must be an object")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")
    eligible: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"xCodeEval row {position} must be an object")
        source_id = _required_text(
            row.get("src_uid"),
            f"xCodeEval row {position}.src_uid",
        )
        raw_tests = unit_test_db.get(source_id)
        if raw_tests is None or source_id in seen_ids:
            continue
        tests = _normalize_tests(
            raw_tests,
            context=f"xCodeEval unit tests {source_id}",
        )
        if not tests:
            continue
        seen_ids.add(source_id)
        problem_text = _build_problem_text(row, position=position)
        eligible.append(
            make_multisource_task(
                task_id=f"xcodeeval_compact_{source_id}",
                source={
                    "dataset": XCODEEVAL_DATASET_ID,
                    "config": XCODEEVAL_CONFIG,
                    "split": XCODEEVAL_SPLIT,
                    "raw_split": XCODEEVAL_RAW_SPLIT,
                    "original_id": source_id,
                    "revision": revision,
                    "license": XCODEEVAL_LICENSE,
                    "license_dataset_card": "CC-BY-4.0",
                    "license_pinned_loader": "CC-BY-NC-4.0",
                    "provenance": XCODEEVAL_PROVENANCE,
                    "mirror": XCODEEVAL_MIRROR,
                    "raw_file": XCODEEVAL_COMPACT_FILE,
                    "unit_test_file": XCODEEVAL_UNIT_TEST_FILE,
                },
                problem_text=problem_text,
                interface_type="stdin_stdout",
                required_interface=_STDIO_INTERFACE,
                tests=tests,
                metadata={
                    "difficulty": row.get("difficulty"),
                    "tags": _optional_string_list(
                        row.get("tags"),
                        context=f"xCodeEval row {position}.tags",
                    ),
                    "input_from": row.get("input_from"),
                    "output_to": row.get("output_to"),
                    "time_limit": row.get("time_limit"),
                    "memory_limit": row.get("memory_limit"),
                    "created_at": row.get("created_at"),
                    "official_test_count": len(tests),
                    "official_tests_sha256": _json_digest(tests),
                    "benchmark_split_role": (
                        "compact_used_as_training_source_by_explicit_project_decision"
                    ),
                    "eligibility": {
                        "interface_type": "stdin_stdout",
                        "official_hidden_tests_retained_outside_prompt": True,
                    },
                },
            )
        )
    eligible.sort(key=lambda task: task["id"])
    return select_unique_tasks(
        eligible,
        limit=limit,
        selection=selection,
        seed=seed,
    )


def load_xcodeeval_tasks(
    *,
    limit: int,
    selection: str = "random",
    seed: int = XCODEEVAL_DEFAULT_SEED,
    revision: str = XCODEEVAL_REVISION,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Download two pinned JSON files without executing xCodeEval's loader."""

    if cache_dir is not None:
        os.environ["HF_HOME"] = str(Path(cache_dir).resolve())
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "xCodeEval import requires the optional 'data' dependencies"
        ) from error

    compact_path = hf_hub_download(
        repo_id=XCODEEVAL_DATASET_ID,
        repo_type="dataset",
        filename=XCODEEVAL_COMPACT_FILE,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    unit_test_path = hf_hub_download(
        repo_id=XCODEEVAL_DATASET_ID,
        repo_type="dataset",
        filename=XCODEEVAL_UNIT_TEST_FILE,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    unit_test_db = json.loads(Path(unit_test_path).read_text(encoding="utf-8"))
    if not isinstance(unit_test_db, dict):
        raise ValueError("xCodeEval unittest_db.json must contain an object")
    return import_xcodeeval_rows(
        _read_jsonl(Path(compact_path)),
        unit_test_db=unit_test_db,
        limit=limit,
        selection=selection,
        seed=seed,
        revision=revision,
    )


def _build_problem_text(row: Mapping[str, Any], *, position: int) -> str:
    description = _required_text(
        row.get("description"),
        f"xCodeEval row {position}.description",
    )
    input_spec = _required_text(
        row.get("input_spec"),
        f"xCodeEval row {position}.input_spec",
    )
    output_spec = _required_text(
        row.get("output_spec"),
        f"xCodeEval row {position}.output_spec",
    )
    parts = [
        description,
        f"Input:\n{input_spec}",
        f"Output:\n{output_spec}",
    ]
    notes = row.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"xCodeEval row {position}.notes must be a string or null")
    if isinstance(notes, str) and notes.strip():
        parts.append(f"Notes:\n{notes.strip()}")
    sample_inputs = _optional_string_list(
        row.get("sample_inputs"),
        context=f"xCodeEval row {position}.sample_inputs",
    )
    sample_outputs = _optional_string_list(
        row.get("sample_outputs"),
        context=f"xCodeEval row {position}.sample_outputs",
    )
    if len(sample_inputs) != len(sample_outputs):
        raise ValueError(
            f"xCodeEval row {position} sample input/output counts differ"
        )
    for index, (sample_input, sample_output) in enumerate(
        zip(sample_inputs, sample_outputs),
        start=1,
    ):
        parts.append(
            f"Example {index} input:\n{sample_input}\n"
            f"Example {index} output:\n{sample_output}"
        )
    return "\n\n".join(parts)


def _normalize_tests(value: Any, *, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    normalized: list[dict[str, Any]] = []
    for index, test in enumerate(value):
        if not isinstance(test, Mapping):
            raise ValueError(f"{context}[{index}] must be an object")
        test_input = test.get("input")
        output_value = test.get("output")
        if not isinstance(test_input, str):
            raise ValueError(f"{context}[{index}].input must be a string")
        if isinstance(output_value, str):
            outputs: str | list[str] = output_value
        elif (
            isinstance(output_value, list)
            and output_value
            and all(isinstance(item, str) for item in output_value)
        ):
            outputs = list(output_value)
        else:
            raise ValueError(
                f"{context}[{index}].output must be a string or "
                "non-empty string list"
            )
        normalized.append({"input": test_input, "output": outputs})
    return normalized


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


def _optional_string_list(value: Any, *, context: str) -> list[str]:
    if value is None or value == "":
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{context} must be a list of strings")
    return list(value)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _required_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value.strip()
