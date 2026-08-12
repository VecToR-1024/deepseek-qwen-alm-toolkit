"""Strict cross-tokenizer alignment on a shared UTF-8 text prefix."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .records import SENTINEL_LOGPROB


class AlignmentError(ValueError):
    """Raised when normalized teacher data cannot be aligned safely."""


class TokenizerEncoder(Protocol):
    """Small tokenizer surface needed by the alignment core."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class SoftPosition:
    teacher_position: int
    student_logit_position: int
    mapped_student_token_ids: tuple[int, ...]
    teacher_probs: tuple[float, ...]
    teacher_tail_prob: float
    mapped_teacher_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AlignmentStats:
    teacher_positions: int
    aligned_positions: int
    total_candidates: int
    mapped_candidates: int
    candidate_collisions: int
    skipped_invalid_prefix_bytes: int
    skipped_unknown_prefix_bytes: int
    skipped_unstable_prefix: int
    skipped_no_mapped_candidates: int
    skipped_duplicate_logit_positions: int

    @property
    def aligned_position_ratio(self) -> float:
        if self.teacher_positions == 0:
            return 0.0
        return self.aligned_positions / self.teacher_positions

    @property
    def mean_mapped_candidates_per_aligned_position(self) -> float:
        if self.aligned_positions == 0:
            return 0.0
        return self.mapped_candidates / self.aligned_positions

    @property
    def candidate_mapping_ratio(self) -> float:
        if self.total_candidates == 0:
            return 0.0
        return self.mapped_candidates / self.total_candidates


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    student_input_ids: tuple[int, ...]
    soft_positions: tuple[SoftPosition, ...]
    stats: AlignmentStats


def map_candidate_to_single_token(
    tokenizer: TokenizerEncoder,
    *,
    prefix_text: str,
    candidate_bytes: bytes,
) -> int | None:
    """Return the appended student token ID when the mapping is exact.

    Bytes are authoritative because provider token strings may use display
    placeholders for byte-level tokens. Incomplete UTF-8 candidates cannot be
    represented safely by a text tokenizer and are therefore not aligned.
    """
    try:
        candidate_text = candidate_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None

    prefix_ids = _encode(tokenizer, prefix_text)
    candidate_ids = _encode(
        tokenizer,
        prefix_text + candidate_text,
    )
    if len(candidate_ids) != len(prefix_ids) + 1:
        return None
    if candidate_ids[:-1] != prefix_ids:
        return None
    return candidate_ids[-1]


def align_teacher_content(
    tokenizer: TokenizerEncoder,
    *,
    context_text: str,
    response_text: str,
    content_tokens: Sequence[Mapping[str, Any]],
    student_full_text: str | None = None,
) -> AlignmentResult:
    """Align normalized teacher top-k rows to strict student token positions."""
    rows = _validated_rows(content_tokens, response_text)
    full_text = context_text + response_text if student_full_text is None else student_full_text
    student_input_ids = tuple(_encode(tokenizer, full_text))
    prefix_bytes = bytearray()
    prefix_is_known = True
    tentative: list[SoftPosition] = []
    total_candidates = 0
    candidate_collisions = 0
    skipped_invalid_prefix_bytes = 0
    skipped_unknown_prefix_bytes = 0
    skipped_unstable_prefix = 0
    skipped_no_mapped_candidates = 0

    for teacher_position, row in enumerate(rows):
        candidates = row["top_logprobs"]
        total_candidates += len(candidates)
        if not prefix_is_known:
            skipped_unknown_prefix_bytes += 1
        else:
            try:
                response_prefix = bytes(prefix_bytes).decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                skipped_invalid_prefix_bytes += 1
            else:
                prefix_text = context_text + response_prefix
                prefix_ids = _encode(tokenizer, prefix_text)
                if (
                    not prefix_ids
                    or len(prefix_ids) >= len(student_input_ids)
                    or tuple(prefix_ids) != student_input_ids[: len(prefix_ids)]
                ):
                    skipped_unstable_prefix += 1
                else:
                    mapped, collisions = _map_position_candidates(
                        tokenizer,
                        prefix_text=prefix_text,
                        prefix_ids=prefix_ids,
                        candidates=candidates,
                    )
                    candidate_collisions += collisions
                    if not mapped:
                        skipped_no_mapped_candidates += 1
                    else:
                        mapped_mass = math.fsum(item[2] for item in mapped)
                        if mapped_mass > 1.0 + 1e-7:
                            raise AlignmentError(
                                f"teacher position {teacher_position} mapped probability mass exceeds one"
                            )
                        tentative.append(
                            SoftPosition(
                                teacher_position=teacher_position,
                                student_logit_position=len(prefix_ids) - 1,
                                mapped_student_token_ids=tuple(item[0] for item in mapped),
                                mapped_teacher_tokens=tuple(item[1] for item in mapped),
                                teacher_probs=tuple(item[2] for item in mapped),
                                teacher_tail_prob=max(0.0, 1.0 - min(mapped_mass, 1.0)),
                            )
                        )

        actual_bytes = row["bytes"]
        if actual_bytes is None:
            prefix_is_known = False
        elif prefix_is_known:
            prefix_bytes.extend(actual_bytes)

    logit_counts = Counter(item.student_logit_position for item in tentative)
    soft_positions = tuple(
        item for item in tentative if logit_counts[item.student_logit_position] == 1
    )
    skipped_duplicate_logit_positions = len(tentative) - len(soft_positions)
    mapped_candidates = sum(len(item.mapped_student_token_ids) for item in soft_positions)
    stats = AlignmentStats(
        teacher_positions=len(rows),
        aligned_positions=len(soft_positions),
        total_candidates=total_candidates,
        mapped_candidates=mapped_candidates,
        candidate_collisions=candidate_collisions,
        skipped_invalid_prefix_bytes=skipped_invalid_prefix_bytes,
        skipped_unknown_prefix_bytes=skipped_unknown_prefix_bytes,
        skipped_unstable_prefix=skipped_unstable_prefix,
        skipped_no_mapped_candidates=skipped_no_mapped_candidates,
        skipped_duplicate_logit_positions=skipped_duplicate_logit_positions,
    )
    return AlignmentResult(
        student_input_ids=student_input_ids,
        soft_positions=soft_positions,
        stats=stats,
    )


def _map_position_candidates(
    tokenizer: TokenizerEncoder,
    *,
    prefix_text: str,
    prefix_ids: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, str, float]], int]:
    mapped: list[tuple[int, str, float]] = []
    for candidate_index, candidate in enumerate(candidates):
        token = candidate.get("token")
        if not isinstance(token, str):
            raise AlignmentError(f"candidate {candidate_index}.token must be a string")
        candidate_bytes = _optional_bytes(candidate.get("bytes"), f"candidate {candidate_index}.bytes")
        probability = _probability(candidate.get("logprob"), f"candidate {candidate_index}.logprob")
        if candidate_bytes is None or probability == 0.0:
            continue
        try:
            candidate_text = candidate_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        candidate_ids = _encode(tokenizer, prefix_text + candidate_text)
        if len(candidate_ids) != len(prefix_ids) + 1 or candidate_ids[:-1] != list(prefix_ids):
            continue
        mapped.append((candidate_ids[-1], token, probability))

    counts = Counter(item[0] for item in mapped)
    collisions = sum(count for count in counts.values() if count > 1)
    return [item for item in mapped if counts[item[0]] == 1], collisions


def _validated_rows(
    content_tokens: Sequence[Mapping[str, Any]],
    response_text: str,
) -> list[dict[str, Any]]:
    if not isinstance(response_text, str):
        raise AlignmentError("response_text must be a string")
    if not isinstance(content_tokens, Sequence) or isinstance(content_tokens, (str, bytes)):
        raise AlignmentError("content_tokens must be a sequence")

    rows: list[dict[str, Any]] = []
    reconstructed_parts: list[bytes] = []
    reconstruction_is_complete = True
    for position, value in enumerate(content_tokens):
        if not isinstance(value, Mapping):
            raise AlignmentError(f"content_tokens[{position}] must be an object")
        candidates = value.get("top_logprobs")
        if not isinstance(candidates, list):
            raise AlignmentError(f"content_tokens[{position}].top_logprobs must be a list")
        if any(not isinstance(candidate, Mapping) for candidate in candidates):
            raise AlignmentError(f"content_tokens[{position}].top_logprobs must contain objects")
        actual_bytes = _optional_bytes(value.get("bytes"), f"content_tokens[{position}].bytes")
        if actual_bytes is None:
            reconstruction_is_complete = False
        else:
            reconstructed_parts.append(actual_bytes)
        rows.append({"bytes": actual_bytes, "top_logprobs": candidates})

    if reconstruction_is_complete:
        try:
            reconstructed = b"".join(reconstructed_parts).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise AlignmentError("teacher token bytes are not valid UTF-8") from error
        if reconstructed != response_text:
            raise AlignmentError("teacher token bytes do not reconstruct response_text")
    return rows


def _optional_bytes(value: Any, context: str) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise AlignmentError(f"{context} must be a list or null")
    if any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255 for item in value):
        raise AlignmentError(f"{context} must contain integers between 0 and 255")
    return bytes(value)


def _probability(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlignmentError(f"{context} must be a number")
    logprob = float(value)
    if not math.isfinite(logprob) or logprob > 1e-7:
        raise AlignmentError(f"{context} must be a finite non-positive number")
    return 0.0 if logprob == SENTINEL_LOGPROB else math.exp(logprob)


def _encode(tokenizer: TokenizerEncoder, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in token_ids):
        raise AlignmentError("tokenizer.encode must return non-negative integer token IDs")
    return token_ids
