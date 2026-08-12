"""Build loss-ready ALM chunks from offline DeepSeek traces.

The cumulative-byte endpoint grouping follows the Apache-2.0 tokenkit ALM
reference implementation, adapted to provider-supplied bytes instead of a
locally loaded teacher tokenizer/model:
https://github.com/bminixhofer/tokenkit/blob/main/tokenkit/align.py
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any, Protocol

from .alignment import AlignmentError
from .cross_tokenizer_aligner import (
    ByteOffsetTokenizer,
    CrossTokenizerAligner,
    HuggingFaceByteOffsetTokenizer,
)
from .offline_teacher import OfflineTeacherTraceProvider


class ALMChatTokenizer(Protocol):
    """Chat-template surface needed to build a teacher-forced student input."""

    eos_token_id: int

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> str: ...


class ALMExampleBuilder:
    """Convert one normalized record to causal-SFT and ALM tensor fields."""

    def __init__(
        self,
        tokenizer: ALMChatTokenizer,
        *,
        trace_provider: OfflineTeacherTraceProvider | None = None,
        byte_offset_tokenizer: ByteOffsetTokenizer | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._trace_provider = trace_provider or OfflineTeacherTraceProvider()
        self._chat_template_kwargs = dict(chat_template_kwargs or {})
        if any(not isinstance(key, str) for key in self._chat_template_kwargs):
            raise ValueError("chat_template_kwargs keys must be strings")
        if byte_offset_tokenizer is None:
            if hasattr(tokenizer, "encode_with_byte_offsets"):
                byte_offset_tokenizer = tokenizer  # type: ignore[assignment]
            else:
                byte_offset_tokenizer = HuggingFaceByteOffsetTokenizer(tokenizer)
        self._aligner = CrossTokenizerAligner(byte_offset_tokenizer)

    def build(self, record: Mapping[str, Any]) -> dict[str, Any]:
        trace = self._trace_provider.get_trace(record)
        request = record.get("request")
        if not isinstance(request, Mapping):
            raise AlignmentError("teacher record request must be an object")
        source_messages = request.get("messages")
        if not isinstance(source_messages, list) or not source_messages:
            raise AlignmentError("teacher request.messages must be a non-empty list")
        if any(not isinstance(message, Mapping) for message in source_messages):
            raise AlignmentError("teacher request.messages must contain objects")
        messages = [dict(message) for message in source_messages]

        generation_context = self._tokenizer.apply_chat_template(
            copy.deepcopy(messages),
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs,
        )
        full_messages = copy.deepcopy(messages)
        full_messages.append({"role": "assistant", "content": trace.response_text})
        full_training_text = self._tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            **self._chat_template_kwargs,
        )
        if not isinstance(generation_context, str) or not isinstance(
            full_training_text, str
        ):
            raise AlignmentError("chat template must return text when tokenize=False")

        alignment = self._aligner.align(
            context_text=generation_context,
            response_text=trace.response_text,
            student_full_text=full_training_text,
            teacher_token_bytes=trace.token_bytes,
        )

        labels = [-100] * len(alignment.student_input_ids)
        student_tokens = {token.position: token for token in alignment.student_tokens}
        for token in alignment.student_tokens:
            if token.position > 0:
                labels[token.position] = alignment.student_input_ids[token.position]
        assistant_eos_position = _assistant_eos_position(
            tokenizer=self._tokenizer,
            student_input_ids=alignment.student_input_ids,
            student_byte_offsets=alignment.student_byte_offsets,
            response_end_byte=len(
                (generation_context + trace.response_text).encode("utf-8")
            ),
        )
        labels[assistant_eos_position] = alignment.student_input_ids[
            assistant_eos_position
        ]

        student_chunk_ids = [-1] * len(alignment.student_input_ids)
        teacher_chunk_logprobs: list[float] = []
        dropped_boundary_chunks = 0
        for group in alignment.groups:
            if any(
                student_tokens[position].is_boundary_clipped
                for position in group.student_positions
            ):
                dropped_boundary_chunks += 1
                continue
            chunk_id = len(teacher_chunk_logprobs)
            teacher_chunk_logprobs.append(
                math.fsum(trace.token_logprobs[position] for position in group.teacher_positions)
            )
            for position in group.student_positions:
                if position <= 0:
                    raise AlignmentError("a student ALM token has no preceding causal logit")
                if student_chunk_ids[position] != -1:
                    raise AlignmentError("a student token was assigned to more than one ALM chunk")
                student_chunk_ids[position] = chunk_id

        return {
            "input_ids": list(alignment.student_input_ids),
            "attention_mask": [1] * len(alignment.student_input_ids),
            "labels": labels,
            "alm_student_chunk_ids": student_chunk_ids,
            "alm_teacher_chunk_logprobs": teacher_chunk_logprobs,
            "alm_chunk_count": len(teacher_chunk_logprobs),
            "alm_dropped_boundary_chunks": dropped_boundary_chunks,
        }


def _assistant_eos_position(
    *,
    tokenizer: ALMChatTokenizer,
    student_input_ids: tuple[int, ...],
    student_byte_offsets: tuple[tuple[int, int], ...],
    response_end_byte: int,
) -> int:
    """Locate the template EOS at the exact assistant-response byte boundary."""

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if (
        isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or eos_token_id < 0
    ):
        raise AlignmentError("tokenizer.eos_token_id must be a non-negative integer")
    candidates = [
        position
        for position, (token_id, (start_byte, end_byte)) in enumerate(
            zip(student_input_ids, student_byte_offsets, strict=True)
        )
        if token_id == eos_token_id
        and start_byte == response_end_byte
        and end_byte > start_byte
    ]
    if len(candidates) != 1:
        raise AlignmentError(
            "chat template must place exactly one EOS token at the assistant "
            "response byte boundary"
        )
    position = candidates[0]
    if position == 0:
        raise AlignmentError("assistant EOS has no preceding causal logit")
    return position
