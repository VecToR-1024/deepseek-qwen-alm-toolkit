from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from deepseek_distill.cross_tokenizer_aligner import ByteOffsetEncoding
from deepseek_distill.training_contract_audit import (
    build_training_contract_report,
)
from deepseek_distill.training_data_audit import (
    assertion_structure_fingerprints,
    audit_labeled_sequence,
    build_mbpp_overlap_report,
    ending_features,
    extract_mbpp_problem,
    nearest_text_matches,
    response_style_features,
)


@dataclass
class AuditTokenizer:
    response: str
    eos_token_id: int = 99
    eos_token: str = "<|im_end|>"
    pad_token_id: int = 98
    pad_token: str = "<|endoftext|>"
    bos_token_id: int | None = None
    bos_token: str | None = None
    chat_template: str = "fake qwen template"
    chat_template_calls: list[dict] | None = None

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs,
    ) -> str:
        if self.chat_template_calls is not None:
            self.chat_template_calls.append(kwargs)
        assert tokenize is False
        if add_generation_prompt:
            return "<ctx>"
        return "<ctx>" + self.response + "<|im_end|>\n"

    def encode_with_byte_offsets(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> ByteOffsetEncoding:
        assert add_special_tokens is False
        response_start = len(b"<ctx>")
        response_end = response_start + len(self.response.encode("utf-8"))
        suffix_end = response_end + len(b"<|im_end|>")
        return ByteOffsetEncoding(
            (10, 11, self.eos_token_id, 12),
            (
                (0, response_start),
                (response_start, response_end),
                (response_end, suffix_end),
                (suffix_end, suffix_end + 1),
            ),
        )

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return {
            10: "<ctx>",
            11: self.response,
            99: "<|im_end|>",
            12: "\n",
        }[token_id]


def normalized_record(
    record_id: str,
    response: str,
    *,
    dataset: str = "MBPP",
    problem_text: str = "Write a function.",
    tests: list[str] | None = None,
) -> dict:
    return {
        "schema_version": "deepseek.teacher.normalized.v1",
        "id": record_id,
        "request": {"messages": [{"role": "user", "content": problem_text}]},
        "response_text": response,
        "content_tokens": [
            {
                "token": response,
                "bytes": list(response.encode("utf-8")),
                "logprob": -0.1,
                "top_logprobs": [],
            }
        ],
        "task": {
            "source": {"dataset": dataset},
            "problem_text": problem_text,
            "function_name": "solve",
            "tests": tests or ["assert solve(1) == 2"],
        },
    }


def test_audit_labeled_sequence_distinguishes_present_from_supervised_eos() -> None:
    ignored = audit_labeled_sequence(
        input_ids=[10, 11, 99, 12],
        labels=[-100, 11, -100, -100],
        eos_token_ids=[99],
        id_to_token=lambda token_id: {
            10: "prompt",
            11: "answer",
            99: "<|im_end|>",
            12: "\n",
        }[token_id],
    )
    supervised = audit_labeled_sequence(
        input_ids=[10, 11, 99],
        labels=[-100, 11, 99],
        eos_token_ids=[99],
        id_to_token=str,
    )

    assert ignored["eos_present"] is True
    assert ignored["eos_positions"] == [2]
    assert ignored["eos_labels"] == [-100]
    assert ignored["eos_supervised"] is False
    assert ignored["last_supervised_token"] == {
        "position": 1,
        "id": 11,
        "token": "answer",
    }
    assert supervised["eos_supervised"] is True


def test_response_style_features_count_comments_fences_and_docstrings() -> None:
    source = (
        '"""module docs"""\n'
        "def solve(x):\n"
        "    # explain the branch\n"
        "    return x + 1  # inline\n"
    )
    features = response_style_features(source)

    assert features["line_count"] == 4
    assert features["nonempty_line_count"] == 4
    assert features["comment_line_count"] == 2
    assert features["comment_token_count"] == 2
    assert features["docstring_line_count"] == 1
    assert features["code_fence_count"] == 0
    assert math.isclose(features["comment_line_ratio"], 0.5)


def test_ending_features_preserve_exact_final_bytes() -> None:
    features = ending_features(
        "def f():\n    return 1\n",
        [{"bytes": [32, 49]}, {"bytes": [10]}],
    )

    assert features["ends_with_newline"] is True
    assert features["last_non_whitespace_character"] == "1"
    assert features["last_teacher_token"]["bytes"] == [10]
    assert features["last_teacher_token"]["utf8"] == "\n"


def test_extract_mbpp_problem_removes_embedded_public_assertions() -> None:
    prompt = (
        '"""\nWrite a function to add two integers.\n'
        "assert add(1, 2) == 3\n"
        '"""\n'
    )

    assert extract_mbpp_problem(prompt) == "Write a function to add two integers."


def test_assertion_structure_fingerprints_ignore_literals_but_keep_call_shape() -> None:
    first = assertion_structure_fingerprints(
        ["assert add(1, 2) == 3", "assert add(a=1, b=2) == 3"],
        function_name="add",
    )
    second = assertion_structure_fingerprints(
        ["assert total(4, 5) == 9", "assert total(a=4, b=5) == 9"],
        function_name="total",
    )

    assert first == second
    assert first[0] != first[1]


def test_nearest_text_matches_finds_exact_and_near_duplicates() -> None:
    train = [
        {"id": "train_1", "text": "Write a function to add two integers."},
        {"id": "train_2", "text": "Return the longest string in a list."},
    ]
    heldout = [
        {"id": "held_1", "text": "Write a function to add two integers."},
        {"id": "held_2", "text": "Find the longest string from a list."},
    ]

    report = nearest_text_matches(train, heldout, top_k=2)

    assert report["exact_normalized_match_count"] == 1
    assert report["pairs"][0]["train_id"] == "train_1"
    assert report["pairs"][0]["heldout_id"] == "held_1"
    assert report["pairs"][0]["tfidf_cosine"] == pytest.approx(1.0)
    assert report["heldout_with_similarity_at_least"]["0.7"] == 2
    assert report["heldout_with_tfidf_at_least"]["0.9"] == 1
    assert report["nearest_tfidf_distribution"]["count"] == 2


def test_build_training_contract_report_counts_supervised_eos_and_alm_chunks() -> None:
    response = "def solve(x):\n    # increment\n    return x + 1"
    report = build_training_contract_report(
        [
            normalized_record("mbpp_1", response),
            normalized_record("taco_1", response, dataset="TACO"),
        ],
        AuditTokenizer(response),
    )

    assert report["records"] == 2
    assert report["end_token_supervision"]["eos_present_records"] == 2
    assert report["end_token_supervision"]["eos_supervised_records"] == 2
    assert report["end_token_supervision"]["assistant_suffixes"] == {
        "<|im_end|>\n": 2
    }
    assert report["teacher_response"]["records_with_comments"] == 2
    assert report["teacher_response"]["records_with_code_fences"] == 0
    assert report["alm_preprocessing"]["total_chunks"] == 2
    assert report["alm_preprocessing"]["zero_chunk_records"] == 0
    assert report["alm_preprocessing"]["boundary_drops"] == 0
    assert report["sources"] == {"MBPP": 1, "TACO": 1}
    assert report["teacher_response"]["by_source"]["MBPP"]["records"] == 1
    assert report["teacher_response"]["by_source"]["TACO"]["records"] == 1


def test_training_contract_report_uses_the_configured_chat_template_kwargs() -> None:
    response = "def solve(x): return x"
    calls: list[dict] = []

    report = build_training_contract_report(
        [normalized_record("qwen3", response)],
        AuditTokenizer(response, chat_template_calls=calls),
        chat_template_kwargs={"enable_thinking": False},
    )

    assert report["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls == [{"enable_thinking": False}] * 4


def test_build_mbpp_overlap_report_separates_text_and_test_structure() -> None:
    training = [
        normalized_record(
            "mbpp_1",
            "def solve(x): return x + 1",
            problem_text="Write a function to add one to an integer.",
            tests=["assert solve(1) == 2"],
        )
    ]
    heldout = [
        {
            "task_id": "Mbpp/2",
            "prompt": '"""Write a function to add one to an integer."""',
            "entry_point": "increment",
            "assertion": (
                "assert increment(5) == 6\n"
                "assert increment(1) == 2"
            ),
        }
    ]

    report = build_mbpp_overlap_report(training, heldout)

    assert report["train_tasks"] == 1
    assert report["heldout_tasks"] == 1
    assert report["text"]["exact_normalized_match_count"] == 1
    assert report["tests"]["exact_normalized_test_match_count"] == 0
    assert report["tests"]["heldout_with_any_exact_assertion_match"] == 1
    assert report["tests"]["exact_assertion_match_count"] == 1
    assert report["tests"]["heldout_with_any_exact_named_assertion_match"] == 0
    assert report["tests"]["exact_named_assertion_match_count"] == 0
    assert report["tests"]["exact_structure_match_heldout_count"] == 1
