from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from deepseek_distill.alignment import AlignmentResult, AlignmentStats, SoftPosition
from deepseek_distill.cross_tokenizer_aligner import (
    ByteOffsetEncoding,
    CrossTokenizerAligner,
    HuggingFaceByteOffsetTokenizer,
    compare_strict_and_span,
    diagnose_with_strict_fallback,
)


@dataclass
class DirectOffsetTokenizer:
    encodings: dict[str, ByteOffsetEncoding]
    decode_was_called: bool = False

    def encode_with_byte_offsets(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> ByteOffsetEncoding:
        return self.encodings[text]

    def decode(self, token_ids: list[int]) -> str:
        self.decode_was_called = True
        return "different text"


def encoding(ids: list[int], offsets: list[tuple[int, int]]) -> ByteOffsetEncoding:
    return ByteOffsetEncoding(token_ids=tuple(ids), byte_offsets=tuple(offsets))


def align(
    response: str,
    teacher_pieces: list[bytes],
    student_ids: list[int],
    student_offsets: list[tuple[int, int]],
):
    tokenizer = DirectOffsetTokenizer({response: encoding(student_ids, student_offsets)})
    result = CrossTokenizerAligner(tokenizer).align(
        response_text=response,
        teacher_token_bytes=teacher_pieces,
    )
    return tokenizer, result


def test_zero_width_bos_does_not_shift_response_token_positions() -> None:
    tokenizer, result = align("x", [b"x"], [1, 10], [(0, 0), (0, 1)])

    assert result.groups[0].teacher_positions == (0,)
    assert result.groups[0].student_positions == (1,)
    assert result.stats.zero_width_student_positions == 1
    assert tokenizer.decode_was_called is False


def test_leading_spaces_are_aligned_by_bytes_not_trimmed_text() -> None:
    _, result = align("  x", [b"  ", b"x"], [20, 21], [(0, 1), (1, 3)])

    assert [(group.start_byte, group.end_byte) for group in result.groups] == [(0, 3)]
    assert result.groups[0].teacher_positions == (0, 1)
    assert result.groups[0].student_positions == (0, 1)


def test_newlines_and_python_indentation_keep_exact_byte_boundaries() -> None:
    response = "\n    return x\n"
    teacher = [b"\n", b"    ", b"return", b" x", b"\n"]
    student_offsets = [(0, 1), (1, 3), (3, 5), (5, 11), (11, 13), (13, 14)]

    _, result = align(response, teacher, list(range(30, 36)), student_offsets)

    assert result.stats.teacher_position_coverage == 1.0
    assert result.stats.student_position_coverage == 1.0
    assert b"".join(teacher) == response.encode("utf-8")
    assert [(g.teacher_positions, g.student_positions) for g in result.groups] == [
        ((0,), (0,)),
        ((1,), (1, 2)),
        ((2,), (3,)),
        ((3,), (4,)),
        ((4,), (5,)),
    ]


def test_chinese_tokens_can_split_inside_multibyte_utf8_characters() -> None:
    response = "你好"
    teacher = [b"\xe4", b"\xbd", b"\xa0", "好".encode("utf-8")]

    _, result = align(response, teacher, [40, 41], [(0, 3), (3, 6)])

    assert result.groups[0].teacher_positions == (0, 1, 2)
    assert result.groups[0].student_positions == (0,)
    assert result.groups[1].teacher_positions == (3,)
    assert result.groups[1].student_positions == (1,)
    assert result.stats.many_teacher_to_one_student_groups == 1


def test_one_teacher_token_can_group_to_multiple_student_tokens() -> None:
    _, result = align("return", [b"return"], [50, 51], [(0, 2), (2, 6)])

    assert result.groups[0].teacher_positions == (0,)
    assert result.groups[0].student_positions == (0, 1)
    assert result.stats.one_teacher_to_many_student_groups == 1


def test_multiple_teacher_tokens_can_group_to_one_student_token() -> None:
    _, result = align("return", [b"re", b"turn"], [60], [(0, 6)])

    assert result.groups[0].teacher_positions == (0, 1)
    assert result.groups[0].student_positions == (0,)
    assert result.stats.many_teacher_to_one_student_groups == 1


def test_alignment_never_relies_on_decode_then_encode_round_trip() -> None:
    tokenizer = DirectOffsetTokenizer({"abc": encoding([99], [(0, 3)])})

    result = CrossTokenizerAligner(tokenizer).align(
        response_text="abc",
        teacher_token_bytes=[b"abc"],
    )

    assert result.student_tokens[0].token_id == 99
    assert tokenizer.decode_was_called is False


class ByteLevelMarker:
    def __repr__(self) -> str:
        return "ByteLevel"


@dataclass
class BackendEncoding:
    ids: list[int]
    tokens: list[str]
    offsets: list[tuple[int, int]]


class ByteFallbackBackend:
    pre_tokenizer = ByteLevelMarker()
    decoder = ByteLevelMarker()

    def encode(self, text: str, *, add_special_tokens: bool) -> BackendEncoding:
        assert text == "你"
        assert add_special_tokens is False
        return BackendEncoding(
            ids=[101, 102, 103],
            tokens=["ä", "½", "ł"],
            offsets=[(0, 1), (0, 1), (0, 1)],
        )


@dataclass
class FastTokenizerStub:
    backend_tokenizer: ByteFallbackBackend


def test_huggingface_adapter_splits_repeated_offsets_for_utf8_byte_fallback() -> None:
    adapter = HuggingFaceByteOffsetTokenizer(FastTokenizerStub(ByteFallbackBackend()))

    encoded = adapter.encode_with_byte_offsets("你")

    assert encoded.token_ids == (101, 102, 103)
    assert encoded.byte_offsets == ((0, 1), (1, 2), (2, 3))


def _strict_result() -> AlignmentResult:
    return AlignmentResult(
        student_input_ids=(100, 11, 12),
        soft_positions=(
            SoftPosition(
                teacher_position=0,
                student_logit_position=0,
                mapped_student_token_ids=(11, 21),
                teacher_probs=(0.4, 0.3),
                teacher_tail_prob=0.3,
                mapped_teacher_tokens=("a", "x"),
            ),
        ),
        stats=AlignmentStats(
            teacher_positions=2,
            aligned_positions=1,
            total_candidates=4,
            mapped_candidates=2,
            candidate_collisions=0,
            skipped_invalid_prefix_bytes=0,
            skipped_unknown_prefix_bytes=0,
            skipped_unstable_prefix=1,
            skipped_no_mapped_candidates=0,
            skipped_duplicate_logit_positions=0,
        ),
    )


def _row(probabilities: list[float]) -> dict:
    return {
        "top_logprobs": [
            {"token": str(index), "bytes": [index], "logprob": math.log(probability)}
            for index, probability in enumerate(probabilities, start=1)
        ]
    }


def test_comparison_separates_span_coverage_from_loss_ready_mass() -> None:
    _, span = align("ab", [b"a", b"b"], [70], [(0, 2)])
    strict = _strict_result()

    comparison = compare_strict_and_span(
        strict_result=strict,
        span_result=span,
        content_tokens=[_row([0.5, 0.4]), _row([0.6, 0.2])],
    )

    assert comparison.strict_position_coverage == 0.5
    assert comparison.span_position_coverage == 1.0
    assert comparison.total_teacher_topk_mass == pytest.approx(1.7)
    assert comparison.strict_retained_topk_mass == pytest.approx(0.7)
    assert comparison.span_covered_topk_mass == pytest.approx(1.7)
    assert comparison.loss_ready_topk_mass == pytest.approx(0.7)
    assert comparison.strict_retained_topk_mass_ratio == pytest.approx(0.7 / 1.7)
    assert comparison.span_covered_topk_mass_ratio == 1.0
    assert strict.soft_positions[0].teacher_tail_prob == pytest.approx(0.3)


def test_response_is_sliced_from_one_full_text_encoding_with_bos_and_suffix() -> None:
    full_text = "<ctx> 你<end>"
    tokenizer = DirectOffsetTokenizer(
        {
            full_text: encoding(
                [1, 80, 81, 82],
                [(0, 0), (0, 5), (5, 9), (9, 14)],
            )
        }
    )

    result = CrossTokenizerAligner(tokenizer).align(
        context_text="<ctx>",
        response_text=" 你",
        student_full_text=full_text,
        teacher_token_bytes=[b" ", "你".encode("utf-8")],
    )

    assert [token.position for token in result.student_tokens] == [2]
    assert result.student_tokens[0].start_byte == 0
    assert result.student_tokens[0].end_byte == 4


def test_token_crossing_prompt_boundary_is_clipped_for_span_diagnostics() -> None:
    full_text = "<ctx>\n\nx<end>"
    tokenizer = DirectOffsetTokenizer(
        {
            full_text: encoding(
                [90, 91, 92],
                [(0, 7), (7, 8), (8, 13)],
            )
        }
    )

    result = CrossTokenizerAligner(tokenizer).align(
        context_text="<ctx>\n",
        response_text="\nx",
        student_full_text=full_text,
        teacher_token_bytes=[b"\n", b"x"],
    )

    assert result.student_tokens[0].position == 0
    assert result.student_tokens[0].is_boundary_clipped is True
    assert result.groups[0].teacher_positions == (0,)
    assert result.groups[0].student_positions == (0,)
    assert result.stats.boundary_clipped_student_positions == 1


@dataclass
class StrictTableTokenizer:
    encodings: dict[str, list[int]]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(self.encodings[text])


def test_unsafe_span_offsets_fall_back_to_the_existing_strict_result() -> None:
    strict_tokenizer = StrictTableTokenizer(
        {"<ctx>": [100], "<ctx>x": [100, 11]}
    )
    span_tokenizer = DirectOffsetTokenizer(
        {"<ctx>x": encoding([100], [(0, 5)])}
    )
    content_tokens = [
        {
            "bytes": [120],
            "top_logprobs": [
                {"token": "x", "bytes": [120], "logprob": math.log(0.8)}
            ],
        }
    ]

    diagnostic = diagnose_with_strict_fallback(
        strict_tokenizer=strict_tokenizer,
        span_aligner=CrossTokenizerAligner(span_tokenizer),
        context_text="<ctx>",
        response_text="x",
        content_tokens=content_tokens,
    )

    assert diagnostic.used_strict_fallback is True
    assert diagnostic.span_result is None
    assert diagnostic.comparison is None
    assert diagnostic.span_error == (
        "student token offsets do not reconstruct the response byte length"
    )
    assert diagnostic.training_result.soft_positions[0].mapped_student_token_ids == (11,)
    assert diagnostic.training_result.soft_positions[0].teacher_tail_prob == pytest.approx(0.2)
