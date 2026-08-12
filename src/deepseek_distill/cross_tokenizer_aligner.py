"""Cross-tokenizer alignment on shared UTF-8 byte spans.

The linear common-endpoint walk is also used by tokenkit's ALM alignment.  This
module consumes actual token bytes/offsets only; it never decodes token IDs and
then re-encodes the resulting text.  GOLD's probability merging and ULD
objective are intentionally not included.

Reference implementations (Apache-2.0):
https://github.com/bminixhofer/tokenkit/blob/main/tokenkit/align.py
https://github.com/huggingface/trl/blob/main/trl/experimental/gold/gold_trainer.py
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .alignment import (
    AlignmentError,
    AlignmentResult,
    TokenizerEncoder,
    align_teacher_content,
)
from .records import SENTINEL_LOGPROB


@dataclass(frozen=True, slots=True)
class ByteOffsetEncoding:
    """Token IDs and UTF-8 byte offsets returned by the same encoding call."""

    token_ids: tuple[int, ...]
    byte_offsets: tuple[tuple[int, int], ...]


class ByteOffsetTokenizer(Protocol):
    """Tokenizer boundary required by :class:`CrossTokenizerAligner`."""

    def encode_with_byte_offsets(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> ByteOffsetEncoding: ...


@dataclass(frozen=True, slots=True)
class ByteToken:
    """One actual token placed on the response-relative UTF-8 byte axis."""

    position: int
    token_id: int | None
    start_byte: int
    end_byte: int
    is_boundary_clipped: bool = False


@dataclass(frozen=True, slots=True)
class SpanGroup:
    """Tokens from both vocabularies bounded by the same byte boundaries."""

    teacher_positions: tuple[int, ...]
    student_positions: tuple[int, ...]
    start_byte: int
    end_byte: int


@dataclass(frozen=True, slots=True)
class SpanAlignmentStats:
    teacher_positions: int
    student_positions: int
    aligned_teacher_positions: int
    aligned_student_positions: int
    groups: int
    one_to_one_groups: int
    one_teacher_to_many_student_groups: int
    many_teacher_to_one_student_groups: int
    many_to_many_groups: int
    zero_width_teacher_positions: int
    zero_width_student_positions: int
    boundary_clipped_student_positions: int

    @property
    def teacher_position_coverage(self) -> float:
        if self.teacher_positions == 0:
            return 0.0
        return self.aligned_teacher_positions / self.teacher_positions

    @property
    def student_position_coverage(self) -> float:
        if self.student_positions == 0:
            return 0.0
        return self.aligned_student_positions / self.student_positions


@dataclass(frozen=True, slots=True)
class SpanAlignmentResult:
    """Actual-trajectory span groups; not a loss-ready soft-target mapping."""

    student_input_ids: tuple[int, ...]
    student_byte_offsets: tuple[tuple[int, int], ...]
    teacher_tokens: tuple[ByteToken, ...]
    student_tokens: tuple[ByteToken, ...]
    groups: tuple[SpanGroup, ...]
    stats: SpanAlignmentStats


@dataclass(frozen=True, slots=True)
class AlignmentComparison:
    """Strict-vs-span diagnostics with loss-ready mass kept explicit."""

    strict_position_coverage: float
    span_position_coverage: float
    total_teacher_topk_mass: float
    strict_retained_topk_mass: float
    span_covered_topk_mass: float
    loss_ready_topk_mass: float
    strict_retained_topk_mass_ratio: float
    span_covered_topk_mass_ratio: float


@dataclass(frozen=True, slots=True)
class AlignmentDiagnosticResult:
    """Strict training result plus an optional span-only diagnostic."""

    strict_result: AlignmentResult
    span_result: SpanAlignmentResult | None
    comparison: AlignmentComparison | None
    span_error: str | None

    @property
    def used_strict_fallback(self) -> bool:
        return self.span_result is None

    @property
    def training_result(self) -> AlignmentResult:
        """The strict top-20 result used by the legacy alignment pipeline."""
        return self.strict_result


class CrossTokenizerAligner:
    """Group teacher and student tokens using common UTF-8 byte boundaries.

    The student offsets must be produced together with the student IDs.  The
    teacher side uses DeepSeek's authoritative ``bytes`` arrays, so a teacher
    token may contain only part of a multibyte UTF-8 character.
    """

    def __init__(self, student_tokenizer: ByteOffsetTokenizer) -> None:
        self._student_tokenizer = student_tokenizer

    def align(
        self,
        *,
        response_text: str,
        teacher_token_bytes: Sequence[bytes],
        context_text: str = "",
        student_full_text: str | None = None,
        add_special_tokens: bool = False,
    ) -> SpanAlignmentResult:
        """Align one response while encoding the student full text once."""
        if not isinstance(response_text, str) or not isinstance(context_text, str):
            raise AlignmentError("context_text and response_text must be strings")
        full_text = context_text + response_text if student_full_text is None else student_full_text
        if not isinstance(full_text, str):
            raise AlignmentError("student_full_text must be a string")
        response_start_char = len(context_text)
        response_end_char = response_start_char + len(response_text)
        if full_text[:response_start_char] != context_text:
            raise AlignmentError("student_full_text does not start with context_text")
        if full_text[response_start_char:response_end_char] != response_text:
            raise AlignmentError("response_text does not immediately follow context_text")

        encoded = self._student_tokenizer.encode_with_byte_offsets(
            full_text,
            add_special_tokens=add_special_tokens,
        )
        _validate_encoding(encoded, len(full_text.encode("utf-8")))

        response_start_byte = len(context_text.encode("utf-8"))
        response_end_byte = response_start_byte + len(response_text.encode("utf-8"))
        student_tokens, zero_width_student, boundary_clipped_student = _response_tokens(
            encoded,
            response_start_byte=response_start_byte,
            response_end_byte=response_end_byte,
        )
        teacher_tokens, zero_width_teacher = _teacher_tokens(teacher_token_bytes)
        response_byte_length = response_end_byte - response_start_byte
        _require_exact_byte_coverage(
            teacher_tokens,
            response_byte_length=response_byte_length,
            side="teacher",
        )
        _require_exact_byte_coverage(
            student_tokens,
            response_byte_length=response_byte_length,
            side="student",
        )

        groups = _group_at_shared_end_boundaries(student_tokens, teacher_tokens)
        aligned_teacher = sum(len(group.teacher_positions) for group in groups)
        aligned_student = sum(len(group.student_positions) for group in groups)
        one_to_one = sum(
            len(group.teacher_positions) == 1 and len(group.student_positions) == 1
            for group in groups
        )
        one_to_many = sum(
            len(group.teacher_positions) == 1 and len(group.student_positions) > 1
            for group in groups
        )
        many_to_one = sum(
            len(group.teacher_positions) > 1 and len(group.student_positions) == 1
            for group in groups
        )
        many_to_many = sum(
            len(group.teacher_positions) > 1 and len(group.student_positions) > 1
            for group in groups
        )
        stats = SpanAlignmentStats(
            teacher_positions=len(teacher_token_bytes),
            student_positions=len(student_tokens),
            aligned_teacher_positions=aligned_teacher,
            aligned_student_positions=aligned_student,
            groups=len(groups),
            one_to_one_groups=one_to_one,
            one_teacher_to_many_student_groups=one_to_many,
            many_teacher_to_one_student_groups=many_to_one,
            many_to_many_groups=many_to_many,
            zero_width_teacher_positions=zero_width_teacher,
            zero_width_student_positions=zero_width_student,
            boundary_clipped_student_positions=boundary_clipped_student,
        )
        return SpanAlignmentResult(
            student_input_ids=encoded.token_ids,
            student_byte_offsets=encoded.byte_offsets,
            teacher_tokens=teacher_tokens,
            student_tokens=student_tokens,
            groups=groups,
            stats=stats,
        )


class HuggingFaceByteOffsetTokenizer:
    """Adapter for a Hugging Face fast ByteLevel tokenizer.

    IDs, token pieces, and character offsets all come from the same backend
    encoding.  Character offsets are converted to UTF-8 byte offsets without
    decoding token IDs.
    """

    def __init__(self, tokenizer: Any) -> None:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise AlignmentError("a Hugging Face fast tokenizer with backend_tokenizer is required")
        is_byte_level = "ByteLevel" in repr(backend.pre_tokenizer) or "ByteLevel" in repr(
            backend.decoder
        )
        if not is_byte_level:
            raise AlignmentError(
                "span alignment currently supports ByteLevel tokenizers such as Qwen"
            )
        self._backend = backend

    def encode_with_byte_offsets(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> ByteOffsetEncoding:
        encoding = self._backend.encode(text, add_special_tokens=add_special_tokens)
        char_to_byte = [0]
        for character in text:
            char_to_byte.append(char_to_byte[-1] + len(character.encode("utf-8")))

        byte_offsets: list[tuple[int, int]] = []
        for start, end in encoding.offsets:
            if not 0 <= start <= end <= len(text):
                raise AlignmentError("tokenizer returned an invalid character offset")
            byte_offsets.append((char_to_byte[start], char_to_byte[end]))
        normalized = _normalize_byte_level_offsets(
            byte_offsets,
            list(encoding.tokens),
            text.encode("utf-8"),
        )
        return ByteOffsetEncoding(
            token_ids=tuple(encoding.ids),
            byte_offsets=tuple(normalized),
        )


def compare_strict_and_span(
    *,
    strict_result: AlignmentResult,
    span_result: SpanAlignmentResult,
    content_tokens: Sequence[Mapping[str, Any]],
) -> AlignmentComparison:
    """Compare coverage and top-k mass without changing the training target.

    ``span_covered_topk_mass`` is geometric coverage of teacher positions.  It
    is not treated as loss-ready because a multi-token counterfactual sequence
    needs conditional student logits that the current one-position loss does
    not have.  Consequently ``loss_ready_topk_mass`` remains the strict mapped
    mass and the existing tail bucket remains untouched.
    """
    if len(content_tokens) != strict_result.stats.teacher_positions:
        raise AlignmentError("content_tokens length does not match strict alignment stats")
    if len(content_tokens) != span_result.stats.teacher_positions:
        raise AlignmentError("content_tokens length does not match span alignment stats")

    position_masses = tuple(
        _topk_mass(row, position) for position, row in enumerate(content_tokens)
    )
    total_mass = math.fsum(position_masses)
    strict_mass = math.fsum(
        probability
        for soft_position in strict_result.soft_positions
        for probability in soft_position.teacher_probs
    )
    covered_positions = {
        teacher_position
        for group in span_result.groups
        for teacher_position in group.teacher_positions
    }
    span_mass = math.fsum(position_masses[position] for position in covered_positions)
    return AlignmentComparison(
        strict_position_coverage=strict_result.stats.aligned_position_ratio,
        span_position_coverage=span_result.stats.teacher_position_coverage,
        total_teacher_topk_mass=total_mass,
        strict_retained_topk_mass=strict_mass,
        span_covered_topk_mass=span_mass,
        loss_ready_topk_mass=strict_mass,
        strict_retained_topk_mass_ratio=_ratio(strict_mass, total_mass),
        span_covered_topk_mass_ratio=_ratio(span_mass, total_mass),
    )


def diagnose_with_strict_fallback(
    *,
    strict_tokenizer: TokenizerEncoder,
    span_aligner: CrossTokenizerAligner,
    context_text: str,
    response_text: str,
    content_tokens: Sequence[Mapping[str, Any]],
    student_full_text: str | None = None,
) -> AlignmentDiagnosticResult:
    """Run span diagnostics while preserving strict alignment for training.

    Missing provider bytes or unsafe student offsets do not affect the current
    pipeline: the strict result is returned as ``training_result`` and the span
    failure is recorded for inspection.
    """
    strict_result = align_teacher_content(
        strict_tokenizer,
        context_text=context_text,
        response_text=response_text,
        content_tokens=content_tokens,
        student_full_text=student_full_text,
    )
    try:
        teacher_bytes = tuple(
            _provider_bytes(row, position) for position, row in enumerate(content_tokens)
        )
        span_result = span_aligner.align(
            context_text=context_text,
            response_text=response_text,
            student_full_text=student_full_text,
            teacher_token_bytes=teacher_bytes,
        )
        comparison = compare_strict_and_span(
            strict_result=strict_result,
            span_result=span_result,
            content_tokens=content_tokens,
        )
    except AlignmentError as error:
        return AlignmentDiagnosticResult(
            strict_result=strict_result,
            span_result=None,
            comparison=None,
            span_error=str(error),
        )
    return AlignmentDiagnosticResult(
        strict_result=strict_result,
        span_result=span_result,
        comparison=comparison,
        span_error=None,
    )


def _validate_encoding(encoded: ByteOffsetEncoding, text_byte_length: int) -> None:
    if not isinstance(encoded, ByteOffsetEncoding):
        raise AlignmentError("encode_with_byte_offsets must return ByteOffsetEncoding")
    if len(encoded.token_ids) != len(encoded.byte_offsets):
        raise AlignmentError("student token IDs and byte offsets must have equal length")
    for position, token_id in enumerate(encoded.token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise AlignmentError(f"student token ID {position} must be a non-negative integer")
    cursor = 0
    for position, offset in enumerate(encoded.byte_offsets):
        if (
            not isinstance(offset, tuple)
            or len(offset) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offset)
        ):
            raise AlignmentError(f"student byte offset {position} must be an integer pair")
        start, end = offset
        if not 0 <= start <= end <= text_byte_length:
            raise AlignmentError(f"student byte offset {position} is outside the encoded text")
        if start != end and start < cursor:
            raise AlignmentError("student byte offsets must be monotonic and non-overlapping")
        if start != end:
            cursor = end


def _response_tokens(
    encoded: ByteOffsetEncoding,
    *,
    response_start_byte: int,
    response_end_byte: int,
) -> tuple[tuple[ByteToken, ...], int, int]:
    tokens: list[ByteToken] = []
    zero_width = 0
    boundary_clipped = 0
    for position, (token_id, (start, end)) in enumerate(
        zip(encoded.token_ids, encoded.byte_offsets, strict=True)
    ):
        if start == end:
            zero_width += 1
            continue
        if end <= response_start_byte or start >= response_end_byte:
            continue
        is_boundary_clipped = start < response_start_byte or end > response_end_byte
        boundary_clipped += is_boundary_clipped
        tokens.append(
            ByteToken(
                position=position,
                token_id=token_id,
                start_byte=max(start, response_start_byte) - response_start_byte,
                end_byte=min(end, response_end_byte) - response_start_byte,
                is_boundary_clipped=is_boundary_clipped,
            )
        )
    return tuple(tokens), zero_width, boundary_clipped


def _teacher_tokens(
    teacher_token_bytes: Sequence[bytes],
) -> tuple[tuple[ByteToken, ...], int]:
    if isinstance(teacher_token_bytes, (str, bytes)):
        raise AlignmentError("teacher_token_bytes must be a sequence of bytes values")
    tokens: list[ByteToken] = []
    cursor = 0
    zero_width = 0
    for position, piece in enumerate(teacher_token_bytes):
        if not isinstance(piece, bytes):
            raise AlignmentError(f"teacher token bytes {position} must be bytes")
        end = cursor + len(piece)
        if end == cursor:
            zero_width += 1
        else:
            tokens.append(
                ByteToken(
                    position=position,
                    token_id=None,
                    start_byte=cursor,
                    end_byte=end,
                )
            )
        cursor = end
    return tuple(tokens), zero_width


def _require_exact_byte_coverage(
    tokens: Sequence[ByteToken],
    *,
    response_byte_length: int,
    side: str,
) -> None:
    cursor = 0
    for token in tokens:
        if token.start_byte != cursor or token.end_byte <= token.start_byte:
            raise AlignmentError(f"{side} token offsets do not cover a contiguous response")
        cursor = token.end_byte
    if cursor != response_byte_length:
        raise AlignmentError(f"{side} token offsets do not reconstruct the response byte length")


def _group_at_shared_end_boundaries(
    student_tokens: Sequence[ByteToken],
    teacher_tokens: Sequence[ByteToken],
) -> tuple[SpanGroup, ...]:
    """Walk both streams and close a group at every common byte end."""
    student_index = teacher_index = 0
    student_group_start = teacher_group_start = 0
    groups: list[SpanGroup] = []
    while student_index < len(student_tokens) and teacher_index < len(teacher_tokens):
        student_end = student_tokens[student_index].end_byte
        teacher_end = teacher_tokens[teacher_index].end_byte
        if student_end < teacher_end:
            student_index += 1
            continue
        if student_end > teacher_end:
            teacher_index += 1
            continue

        student_index += 1
        teacher_index += 1
        student_group = student_tokens[student_group_start:student_index]
        teacher_group = teacher_tokens[teacher_group_start:teacher_index]
        if not student_group or not teacher_group:
            raise AlignmentError("shared byte boundary produced an empty alignment group")
        if student_group[0].start_byte != teacher_group[0].start_byte:
            raise AlignmentError("teacher and student group starts do not share a byte boundary")
        groups.append(
            SpanGroup(
                teacher_positions=tuple(token.position for token in teacher_group),
                student_positions=tuple(token.position for token in student_group),
                start_byte=teacher_group[0].start_byte,
                end_byte=teacher_end,
            )
        )
        student_group_start = student_index
        teacher_group_start = teacher_index

    if student_index != len(student_tokens) or teacher_index != len(teacher_tokens):
        raise AlignmentError("teacher and student token streams end at different byte boundaries")
    return tuple(groups)


def _topk_mass(row: Mapping[str, Any], position: int) -> float:
    if not isinstance(row, Mapping):
        raise AlignmentError(f"content_tokens[{position}] must be an object")
    candidates = row.get("top_logprobs")
    if not isinstance(candidates, list):
        raise AlignmentError(f"content_tokens[{position}].top_logprobs must be a list")
    probabilities: list[float] = []
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise AlignmentError(
                f"content_tokens[{position}].top_logprobs[{candidate_index}] must be an object"
            )
        value = candidate.get("logprob")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AlignmentError("teacher candidate logprob must be a number")
        logprob = float(value)
        if not math.isfinite(logprob) or logprob > 1e-7:
            raise AlignmentError("teacher candidate logprob must be finite and non-positive")
        probabilities.append(0.0 if logprob == SENTINEL_LOGPROB else math.exp(logprob))
    mass = math.fsum(probabilities)
    if mass > 1.0 + 1e-7:
        raise AlignmentError(f"teacher top-k mass at position {position} exceeds one")
    return min(mass, 1.0)


def _provider_bytes(row: Mapping[str, Any], position: int) -> bytes:
    if not isinstance(row, Mapping):
        raise AlignmentError(f"content_tokens[{position}] must be an object")
    value = row.get("bytes")
    if value is None:
        raise AlignmentError(f"content_tokens[{position}].bytes is required for span alignment")
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255
        for item in value
    ):
        raise AlignmentError(
            f"content_tokens[{position}].bytes must contain integers between 0 and 255"
        )
    return bytes(value)


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else numerator / denominator


def _byte_to_unicode_decoder() -> dict[str, int]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = list(byte_values)
    extra = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            unicode_values.append(256 + extra)
            extra += 1
    return {
        chr(unicode_value): byte_value
        for byte_value, unicode_value in zip(byte_values, unicode_values, strict=True)
    }


_BYTE_LEVEL_DECODER = _byte_to_unicode_decoder()


def _piece_byte_length(piece: str, text_bytes: bytes, start: int) -> int | None:
    source_bytes: list[int] = []
    for character in piece:
        byte_value = _BYTE_LEVEL_DECODER.get(character)
        if byte_value is None:
            return None
        source_bytes.append(byte_value)
    if source_bytes and source_bytes[0] == ord(" ") and text_bytes[start : start + 1] != b" ":
        source_bytes.pop(0)
    return len(source_bytes)


def _normalize_byte_level_offsets(
    offsets: Sequence[tuple[int, int]],
    pieces: Sequence[str],
    text_bytes: bytes,
) -> list[tuple[int, int]]:
    """Normalize byte-fallback overlaps from fast-tokenizer char offsets."""
    if len(offsets) != len(pieces):
        raise AlignmentError("token pieces and offsets must have equal length")
    normalized = list(offsets)
    index = 0
    while index < len(offsets):
        run_end = index + 1
        while run_end < len(offsets) and offsets[run_end] == offsets[index]:
            run_end += 1
        if run_end - index > 1:
            start, end = offsets[index]
            lengths = [len(piece) for piece in pieces[index:run_end]]
            if math.fsum(lengths) == end - start:
                cursor = start
                for offset_index, length in zip(
                    range(index, run_end),
                    lengths,
                    strict=True,
                ):
                    normalized[offset_index] = (cursor, cursor + length)
                    cursor += length
        index = run_end

    result: list[tuple[int, int]] = []
    cursor = 0
    for index, (start, end) in enumerate(normalized):
        if start == end:
            result.append((cursor, cursor))
            continue
        piece_length = _piece_byte_length(pieces[index], text_bytes, start)
        next_start = normalized[index + 1][0] if index + 1 < len(normalized) else None
        overlaps = start < cursor or (next_start is not None and next_start < end)
        if piece_length is not None and (overlaps or piece_length == end - start):
            candidate_start = max(start, cursor)
            candidate_end = candidate_start + piece_length
            if candidate_end <= end:
                start, end = candidate_start, candidate_end
        if start < cursor or end < start:
            raise AlignmentError(
                "ByteLevel tokenizer returned offsets that could not be normalized"
            )
        result.append((start, end))
        cursor = end
    return result
