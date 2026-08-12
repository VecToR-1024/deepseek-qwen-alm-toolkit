from __future__ import annotations

import copy

import pytest

from deepseek_distill.code_verifier import verify_normalized_record
from deepseek_distill.open_r1_codeforces import (
    OPEN_R1_CODEFORCES_REVISION,
    build_teacher_messages,
    import_open_r1_codeforces_rows,
)
from deepseek_distill.records import NORMALIZED_SCHEMA_VERSION
from deepseek_distill.source_catalog import SOURCE_SPECS


def codeforces_row(index: int) -> dict:
    return {
        "id": f"{1000 + index}_{chr(65 + index)}",
        "aliases": [f"alias-{index}"],
        "contest_id": str(1000 + index),
        "contest_name": f"Contest {index}",
        "contest_type": "CF",
        "contest_start": 1_700_000_000 + index,
        "contest_start_year": 2023,
        "index": chr(65 + index),
        "time_limit": 2.0,
        "memory_limit": 256.0,
        "title": f"Problem {index}",
        "description": f"Compute the answer for problem {index}.",
        "input_format": "The input contains one integer n.",
        "output_format": "Print n.",
        "interaction_format": "",
        "note": "The sample prints its input.",
        "examples": [{"input": f"{index}\n", "output": f"{index}\n"}],
        "editorial": f"SECRET-EDITORIAL-{index}",
        "rating": 800 + 100 * index,
        "tags": ["implementation"],
        "testset_size": 1,
        "official_tests": [
            {"input": f"SECRET-TEST-{index}\n", "output": f"{index}\n"}
        ],
        "official_tests_complete": True,
        "input_mode": "stdio",
        "generated_checker": "",
        "executable": True,
        "generated_tests": 0,
    }


def test_open_r1_codeforces_selection_is_deterministic_and_test_complete() -> None:
    rows = [codeforces_row(index) for index in range(8)]

    tasks = import_open_r1_codeforces_rows(
        rows,
        limit=3,
        selection="random",
        seed=20260803,
    )
    repeated = import_open_r1_codeforces_rows(
        reversed(rows),
        limit=3,
        selection="random",
        seed=20260803,
    )

    assert [task["id"] for task in tasks] == [task["id"] for task in repeated]
    assert all(task["source"]["split"] == "train" for task in tasks)
    assert all(
        task["source"]["revision"] == OPEN_R1_CODEFORCES_REVISION
        for task in tasks
    )
    assert all(task["interface_type"] == "stdin_stdout" for task in tasks)
    assert all(len(task["tests"]) == 1 for task in tasks)


def test_open_r1_codeforces_hard_profile_filters_before_selection() -> None:
    rows = [codeforces_row(index) for index in range(8)]

    tasks = import_open_r1_codeforces_rows(
        rows,
        limit=2,
        selection="first",
        difficulty_profile="hard-v1",
    )

    assert [task["metadata"]["rating"] for task in tasks] == [1400, 1500]


def test_open_r1_codeforces_prompt_keeps_statement_but_hides_verifier_data() -> None:
    row = codeforces_row(2)
    task = import_open_r1_codeforces_rows([row], limit=1, selection="first")[0]

    messages = build_teacher_messages(task)
    rendered = repr(messages)

    assert row["description"] in messages[1]["content"]
    assert row["input_format"] in messages[1]["content"]
    assert row["output_format"] in messages[1]["content"]
    assert row["examples"][0]["input"].strip() in messages[1]["content"]
    assert "SECRET-TEST-2" not in rendered
    assert "SECRET-EDITORIAL-2" not in rendered
    assert len(task["metadata"]["editorial_sha256"]) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("official_tests_complete", False),
        ("input_mode", "file"),
        ("generated_checker", "print(1)"),
        ("interaction_format", "interactive protocol"),
        ("executable", False),
        ("official_tests", []),
    ],
)
def test_open_r1_codeforces_rejects_contracts_current_verifier_cannot_prove(
    field: str,
    value: object,
) -> None:
    row = codeforces_row(1)
    row[field] = value

    with pytest.raises(ValueError, match="only 0 eligible"):
        import_open_r1_codeforces_rows([row], limit=1, selection="first")


def test_open_r1_codeforces_alias_order_does_not_change_identity() -> None:
    row = codeforces_row(3)
    row["aliases"] = ["shared-copy", "1003_D"]
    task = import_open_r1_codeforces_rows([row], limit=1, selection="first")[0]
    row["aliases"].reverse()
    repeated = import_open_r1_codeforces_rows([row], limit=1, selection="first")[0]

    assert task["id"] == repeated["id"]
    assert task["source"]["original_id"] == "1003_D"
    assert task["metadata"]["aliases"] == ["1003_D", "shared-copy"]


def test_open_r1_codeforces_rejects_malformed_official_tests() -> None:
    row = codeforces_row(4)
    row["official_tests"] = [{"input": "1\n", "output": 1}]

    with pytest.raises(ValueError, match="official_tests"):
        import_open_r1_codeforces_rows([row], limit=1, selection="first")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", ""),
        ("description", ""),
        ("description", "   "),
    ],
)
def test_open_r1_codeforces_skips_rows_without_an_actual_problem_statement(
    field: str,
    value: str,
) -> None:
    incomplete = codeforces_row(6)
    incomplete[field] = value
    valid = codeforces_row(7)

    tasks = import_open_r1_codeforces_rows(
        [incomplete, valid],
        limit=1,
        selection="first",
    )

    assert tasks[0]["source"]["original_id"] == valid["id"]


def test_open_r1_codeforces_reuses_existing_normalized_stdio_verifier() -> None:
    row = codeforces_row(5)
    row["official_tests"] = [{"input": "42\n", "output": "42\n"}]
    task = import_open_r1_codeforces_rows([row], limit=1, selection="first")[0]
    source = "import sys\nprint(sys.stdin.read().strip())\n"
    attempt_id = f"{task['id']}__attempt_1"
    attempt_task = copy.deepcopy(task)
    attempt_task["problem_id"] = task["id"]
    attempt_task["id"] = attempt_id
    record = {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "id": attempt_id,
        "request": {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "problem"},
            ]
        },
        "response_text": source,
        "content_tokens": [
            {
                "token": source,
                "bytes": list(source.encode("utf-8")),
                "logprob": -0.1,
                "top_logprobs": [],
            }
        ],
        "validation": {"content_bytes_match": True, "warnings": []},
        "task": attempt_task,
    }

    result = verify_normalized_record(record, phase_timeout_seconds=3.0)

    assert result["failure_category"] == "passed"
    assert result["trace_validation"]["valid"] is True
    assert [phase["name"] for phase in result["phases"]] == ["compile", "test_0"]


def test_open_r1_codeforces_is_registered_for_existing_generic_clis() -> None:
    spec = SOURCE_SPECS["open-r1-codeforces"]

    assert spec.dataset_id == "open-r1/codeforces"
    assert spec.config == "verifiable"
    assert spec.split == "train"
    assert spec.revision == OPEN_R1_CODEFORCES_REVISION
    assert "stdio_exact_output_without_custom_checker_only" in spec.notes
