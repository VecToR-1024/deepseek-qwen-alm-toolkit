from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from deepseek_distill.alignment import (
    AlignmentError,
    align_teacher_content,
    map_candidate_to_single_token,
)


@dataclass
class TableTokenizer:
    encodings: dict[str, list[int]]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(self.encodings[text])


def test_maps_candidate_when_it_appends_exactly_one_student_token() -> None:
    tokenizer = TableTokenizer({"<ctx>": [10], "<ctx>    ": [10, 42]})

    token_id = map_candidate_to_single_token(
        tokenizer,
        prefix_text="<ctx>",
        candidate_bytes=b"    ",
    )

    assert token_id == 42


def test_rejects_candidate_when_appending_it_retokenizes_the_prefix() -> None:
    tokenizer = TableTokenizer({"<ctx>a": [10, 11], "<ctx>ab": [10, 99]})

    token_id = map_candidate_to_single_token(
        tokenizer,
        prefix_text="<ctx>a",
        candidate_bytes=b"b",
    )

    assert token_id is None


def test_rejects_candidate_that_requires_multiple_student_tokens() -> None:
    tokenizer = TableTokenizer({"<ctx>": [10], "<ctx>return": [10, 20, 21]})

    token_id = map_candidate_to_single_token(
        tokenizer,
        prefix_text="<ctx>",
        candidate_bytes=b"return",
    )

    assert token_id is None


def test_rejects_candidate_bytes_that_are_not_complete_utf8() -> None:
    tokenizer = TableTokenizer({"<ctx>": [10]})

    token_id = map_candidate_to_single_token(
        tokenizer,
        prefix_text="<ctx>",
        candidate_bytes=b"\xe4",
    )

    assert token_id is None


def test_maps_multibyte_unicode_from_bytes() -> None:
    tokenizer = TableTokenizer({"<ctx>": [10], "<ctx>你": [10, 88]})

    token_id = map_candidate_to_single_token(
        tokenizer,
        prefix_text="<ctx>",
        candidate_bytes="你".encode("utf-8"),
    )

    assert token_id == 88


def candidate(text: str, probability: float, *, token_id_bytes: bytes | None = None) -> dict:
    raw_bytes = text.encode("utf-8") if token_id_bytes is None else token_id_bytes
    return {
        "token": text,
        "bytes": list(raw_bytes),
        "logprob": math.log(probability),
    }


def teacher_row(actual_text: str, candidates: list[dict], *, actual_bytes: bytes | None = None) -> dict:
    raw_bytes = actual_text.encode("utf-8") if actual_bytes is None else actual_bytes
    return {
        "token": actual_text,
        "bytes": list(raw_bytes),
        "logprob": candidates[0]["logprob"],
        "top_logprobs": candidates,
    }


def test_aligns_teacher_positions_and_puts_all_unmapped_mass_in_tail() -> None:
    tokenizer = TableTokenizer(
        {
            "<ctx>": [100],
            "<ctx>a": [100, 11],
            "<ctx>x": [100, 21],
            "<ctx>yz": [100, 30, 31],
            "<ctx>ab": [100, 11, 12],
            "<ctx>ac": [100, 11, 13],
        }
    )
    rows = [
        teacher_row(
            "a",
            [candidate("a", 0.6), candidate("x", 0.2), candidate("yz", 0.1)],
        ),
        teacher_row("b", [candidate("b", 0.7), candidate("c", 0.1)]),
    ]

    result = align_teacher_content(
        tokenizer,
        context_text="<ctx>",
        response_text="ab",
        content_tokens=rows,
    )

    assert result.student_input_ids == (100, 11, 12)
    assert [item.student_logit_position for item in result.soft_positions] == [0, 1]
    assert result.soft_positions[0].mapped_student_token_ids == (11, 21)
    assert result.soft_positions[0].teacher_probs == pytest.approx((0.6, 0.2))
    assert result.soft_positions[0].teacher_tail_prob == pytest.approx(0.2)
    assert result.soft_positions[1].mapped_student_token_ids == (12, 13)
    assert result.soft_positions[1].teacher_tail_prob == pytest.approx(0.2)
    assert result.stats.aligned_position_ratio == 1.0
    assert result.stats.mean_mapped_candidates_per_aligned_position == 2.0


def test_skips_position_whose_prefix_is_not_stable_in_final_student_sequence() -> None:
    tokenizer = TableTokenizer(
        {
            "<ctx>": [100],
            "<ctx>a": [100, 11],
            "<ctx>ab": [100, 99],
            "<ctx>ac": [100, 11, 13],
        }
    )
    rows = [
        teacher_row("a", [candidate("ab", 0.8)]),
        teacher_row("b", [candidate("b", 0.7), candidate("c", 0.2)]),
    ]

    result = align_teacher_content(
        tokenizer,
        context_text="<ctx>",
        response_text="ab",
        content_tokens=rows,
    )

    assert [item.teacher_position for item in result.soft_positions] == [0]
    assert result.soft_positions[0].mapped_student_token_ids == (99,)
    assert result.stats.skipped_unstable_prefix == 1


def test_drops_every_candidate_in_a_student_token_collision() -> None:
    tokenizer = TableTokenizer(
        {
            "<ctx>": [100],
            "<ctx>A": [100, 11],
            "<ctx>a": [100, 11],
            "<ctx>b": [100, 12],
        }
    )
    rows = [
        teacher_row(
            "b",
            [candidate("A", 0.3), candidate("a", 0.2), candidate("b", 0.4)],
        )
    ]

    result = align_teacher_content(
        tokenizer,
        context_text="<ctx>",
        response_text="b",
        content_tokens=rows,
    )

    assert result.soft_positions[0].mapped_student_token_ids == (12,)
    assert result.soft_positions[0].teacher_probs == pytest.approx((0.4,))
    assert result.soft_positions[0].teacher_tail_prob == pytest.approx(0.6)
    assert result.stats.candidate_collisions == 2


def test_skips_teacher_position_inside_an_incomplete_utf8_prefix() -> None:
    tokenizer = TableTokenizer({"<ctx>": [100], "<ctx>你": [100, 88]})
    rows = [
        teacher_row(
            "byte-1",
            [candidate("你", 0.8)],
            actual_bytes=b"\xe4",
        ),
        teacher_row(
            "byte-2",
            [candidate("unused", 0.7, token_id_bytes=b"\xbd\xa0")],
            actual_bytes=b"\xbd\xa0",
        ),
    ]

    result = align_teacher_content(
        tokenizer,
        context_text="<ctx>",
        response_text="你",
        content_tokens=rows,
    )

    assert [item.teacher_position for item in result.soft_positions] == [0]
    assert result.stats.skipped_invalid_prefix_bytes == 1


def test_drops_all_teacher_positions_that_target_the_same_student_logit() -> None:
    tokenizer = TableTokenizer(
        {
            "<ctx>": [100],
            "<ctx>a": [100],
            "<ctx>ab": [100, 12],
        }
    )
    rows = [
        teacher_row("a", [candidate("ab", 0.8)]),
        teacher_row("b", [candidate("b", 0.7)]),
    ]

    result = align_teacher_content(
        tokenizer,
        context_text="<ctx>",
        response_text="ab",
        content_tokens=rows,
    )

    assert result.soft_positions == ()
    assert result.stats.skipped_duplicate_logit_positions == 2


def test_rejects_teacher_bytes_that_do_not_reconstruct_response_text() -> None:
    tokenizer = TableTokenizer({"<ctx>x": [100, 11]})
    rows = [teacher_row("x", [candidate("x", 0.8)], actual_bytes=b"y")]

    with pytest.raises(AlignmentError, match="do not reconstruct response_text"):
        align_teacher_content(
            tokenizer,
            context_text="<ctx>",
            response_text="x",
            content_tokens=rows,
        )


def test_positions_are_indexed_against_the_full_training_chat_template() -> None:
    tokenizer = TableTokenizer(
        {
            "<ctx>": [100],
            "<ctx>x": [100, 11],
            "<ctx>x<end>": [100, 11, 999],
        }
    )
    rows = [teacher_row("x", [candidate("x", 0.8)])]

    result = align_teacher_content(
        tokenizer,
        context_text="<ctx>",
        response_text="x",
        content_tokens=rows,
        student_full_text="<ctx>x<end>",
    )

    assert result.student_input_ids == (100, 11, 999)
    assert result.soft_positions[0].student_logit_position == 0
