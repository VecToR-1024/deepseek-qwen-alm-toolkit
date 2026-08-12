from __future__ import annotations

from dataclasses import dataclass

import pytest

from deepseek_distill.alm_preprocessing import ALMExampleBuilder
from deepseek_distill.cross_tokenizer_aligner import ByteOffsetEncoding
from deepseek_distill.offline_teacher import (
    OfflineTeacherTraceProvider,
    TeacherTraceError,
)
from deepseek_distill.records import NORMALIZED_SCHEMA_VERSION


def normalized_record(
    response: str,
    pieces: list[bytes],
    logprobs: list[float],
) -> dict:
    return {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "id": "sample",
        "request": {"messages": [{"role": "user", "content": "write code"}]},
        "response_text": response,
        "content_tokens": [
            {
                "token": f"t{index}",
                "bytes": list(piece),
                "logprob": logprob,
                "top_logprobs": [],
            }
            for index, (piece, logprob) in enumerate(
                zip(pieces, logprobs, strict=True)
            )
        ],
    }


@dataclass
class DirectChatOffsetTokenizer:
    response: str
    token_ids: list[int]
    offsets: list[tuple[int, int]]
    context: str = "<ctx>"
    suffix: str = "<end>"
    eos_token_id: int = 101
    decode_was_called: bool = False
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
            self.chat_template_calls.append(
                {
                    "add_generation_prompt": add_generation_prompt,
                    "kwargs": kwargs,
                }
            )
        assert tokenize is False
        if add_generation_prompt:
            return self.context
        assert messages[-1] == {"role": "assistant", "content": self.response}
        return self.context + self.response + self.suffix

    def encode_with_byte_offsets(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> ByteOffsetEncoding:
        assert text == self.context + self.response + self.suffix
        assert add_special_tokens is False
        return ByteOffsetEncoding(tuple(self.token_ids), tuple(self.offsets))

    def decode(self, token_ids: list[int]) -> str:
        self.decode_was_called = True
        raise AssertionError("ALM alignment must not use decoded token equality")


def test_offline_provider_extracts_authoritative_bytes_logprobs_and_full_text() -> None:
    response = "你\n  x"
    pieces = [b"\xe4", b"\xbd\xa0", b"\n  ", b"x"]
    record = normalized_record(response, pieces, [-0.1, -0.2, -0.3, -0.4])

    trace = OfflineTeacherTraceProvider().get_trace(record)

    assert trace.record_id == "sample"
    assert trace.response_text == response
    assert trace.token_bytes == tuple(pieces)
    assert trace.token_logprobs == pytest.approx((-0.1, -0.2, -0.3, -0.4))


def test_offline_provider_rejects_bytes_that_do_not_reconstruct_response() -> None:
    record = normalized_record("x", [b"y"], [-0.2])

    with pytest.raises(TeacherTraceError, match="reconstruct response_text"):
        OfflineTeacherTraceProvider().get_trace(record)


def test_one_teacher_token_to_multiple_student_tokens_has_hand_computed_chunk_sum() -> None:
    response = "return"
    context_len = len(b"<ctx>")
    tokenizer = DirectChatOffsetTokenizer(
        response=response,
        token_ids=[100, 11, 12, 101],
        offsets=[
            (0, context_len),
            (context_len, context_len + 2),
            (context_len + 2, context_len + 6),
            (context_len + 6, context_len + 11),
        ],
    )

    example = ALMExampleBuilder(tokenizer).build(
        normalized_record(response, [b"return"], [-0.7])
    )

    assert example["input_ids"] == [100, 11, 12, 101]
    assert example["labels"] == [-100, 11, 12, 101]
    assert example["alm_student_chunk_ids"] == [-1, 0, 0, -1]
    assert example["alm_teacher_chunk_logprobs"] == pytest.approx([-0.7])
    assert example["alm_chunk_count"] == 1
    assert tokenizer.decode_was_called is False


def test_multiple_teacher_tokens_to_one_student_token_sums_log_likelihoods() -> None:
    response = "return"
    context_len = len(b"<ctx>")
    tokenizer = DirectChatOffsetTokenizer(
        response=response,
        token_ids=[100, 20, 101],
        offsets=[
            (0, context_len),
            (context_len, context_len + 6),
            (context_len + 6, context_len + 11),
        ],
    )

    example = ALMExampleBuilder(tokenizer).build(
        normalized_record(response, [b"re", b"turn"], [-0.2, -0.3])
    )

    assert example["labels"] == [-100, 20, 101]
    assert example["alm_student_chunk_ids"] == [-1, 0, -1]
    assert example["alm_teacher_chunk_logprobs"] == pytest.approx([-0.5])


@pytest.mark.parametrize(
    (
        "response",
        "teacher_pieces",
        "teacher_logprobs",
        "student_lengths",
        "expected_chunks",
    ),
    [
        (
            "你好",
            [b"\xe4", b"\xbd", b"\xa0", "好".encode("utf-8")],
            [-0.1, -0.2, -0.3, -0.4],
            [3, 3],
            [-0.6, -0.4],
        ),
        (
            "\n    return x\n",
            [b"\n", b"    ", b"return", b" x", b"\n"],
            [-0.1, -0.2, -0.3, -0.4, -0.5],
            [1, 2, 2, 6, 2, 1],
            [-0.1, -0.2, -0.3, -0.4, -0.5],
        ),
    ],
)
def test_utf8_and_python_whitespace_preserve_exact_byte_chunks(
    response: str,
    teacher_pieces: list[bytes],
    teacher_logprobs: list[float],
    student_lengths: list[int],
    expected_chunks: list[float],
) -> None:
    context_len = len(b"<ctx>")
    cursor = context_len
    response_offsets = []
    for length in student_lengths:
        response_offsets.append((cursor, cursor + length))
        cursor += length
    tokenizer = DirectChatOffsetTokenizer(
        response=response,
        token_ids=[100, *range(200, 200 + len(student_lengths)), 101],
        offsets=[
            (0, context_len),
            *response_offsets,
            (cursor, cursor + len(b"<end>")),
        ],
    )

    example = ALMExampleBuilder(tokenizer).build(
        normalized_record(response, teacher_pieces, teacher_logprobs)
    )

    assert example["alm_teacher_chunk_logprobs"] == pytest.approx(expected_chunks)
    assert example["alm_chunk_count"] == len(expected_chunks)


def test_boundary_clipped_student_chunk_is_not_used_for_alm() -> None:
    response = "x"
    context_len = len(b"<ctx>")
    tokenizer = DirectChatOffsetTokenizer(
        response=response,
        token_ids=[10, 11, 101],
        offsets=[
            (0, context_len - 1),
            (context_len - 1, context_len + 1),
            (context_len + 1, context_len + 6),
        ],
    )

    example = ALMExampleBuilder(tokenizer).build(
        normalized_record(response, [b"x"], [-0.25])
    )

    assert example["labels"] == [-100, 11, 101]
    assert example["alm_student_chunk_ids"] == [-1, -1, -1]
    assert example["alm_teacher_chunk_logprobs"] == []
    assert example["alm_dropped_boundary_chunks"] == 1


def test_only_assistant_suffix_eos_is_supervised_and_trailing_newline_is_masked() -> None:
    response = "x"
    context = "<ctx><end>"
    response_start = len(context.encode("utf-8"))
    response_end = response_start + len(response.encode("utf-8"))
    eos_end = response_end + len(b"<end>")
    tokenizer = DirectChatOffsetTokenizer(
        response=response,
        context=context,
        suffix="<end>\n",
        token_ids=[100, 101, 11, 101, 12],
        offsets=[
            (0, len(b"<ctx>")),
            (len(b"<ctx>"), response_start),
            (response_start, response_end),
            (response_end, eos_end),
            (eos_end, eos_end + 1),
        ],
    )

    example = ALMExampleBuilder(tokenizer).build(
        normalized_record(response, [b"x"], [-0.25])
    )

    assert example["labels"] == [-100, -100, 11, 101, -100]
    assert example["alm_student_chunk_ids"] == [-1, -1, 0, -1, -1]


def test_chat_template_kwargs_are_applied_to_prompt_and_completion() -> None:
    response = "x"
    context_len = len(b"<ctx>")
    calls: list[dict] = []
    tokenizer = DirectChatOffsetTokenizer(
        response=response,
        token_ids=[100, 11, 101],
        offsets=[
            (0, context_len),
            (context_len, context_len + 1),
            (context_len + 1, context_len + 6),
        ],
        chat_template_calls=calls,
    )

    ALMExampleBuilder(
        tokenizer,
        chat_template_kwargs={"enable_thinking": False},
    ).build(normalized_record(response, [b"x"], [-0.25]))

    assert calls == [
        {
            "add_generation_prompt": True,
            "kwargs": {"enable_thinking": False},
        },
        {
            "add_generation_prompt": False,
            "kwargs": {"enable_thinking": False},
        },
    ]
