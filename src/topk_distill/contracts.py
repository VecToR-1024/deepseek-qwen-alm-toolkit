"""Validation and normalization for teacher sequence-logprob responses."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def normalize_sequence_logprobs(
    response: Mapping[str, Any],
    *,
    completion_lengths: Sequence[int],
    top_k: int,
) -> dict[str, list]:
    """Return the fixed-width response shape expected by TRL.

    Providers may return fewer than ``top_k`` entries near constrained tokens.
    Missing entries are padded with ``-inf`` and token id ``0``. When the API
    cannot score the actually observed token, ``actual_logprobs`` is filled
    with ``-inf``; this is valid for forward-KL (``beta=0``) only.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    logprobs = response.get("logprobs")
    token_ids = response.get("logprob_token_ids")
    if not isinstance(logprobs, list) or not isinstance(token_ids, list):
        raise ValueError("response must contain list-valued logprobs and logprob_token_ids")
    if len(logprobs) != len(completion_lengths) or len(token_ids) != len(completion_lengths):
        raise ValueError("response batch size does not match completion_lengths")

    normalized_lps: list[list[list[float]]] = []
    normalized_ids: list[list[list[int]]] = []
    for batch_index, expected_length in enumerate(completion_lengths):
        sequence_lps = logprobs[batch_index]
        sequence_ids = token_ids[batch_index]
        if len(sequence_lps) != expected_length or len(sequence_ids) != expected_length:
            raise ValueError(
                f"batch {batch_index} has the wrong number of completion positions: "
                f"expected {expected_length}"
            )

        fixed_sequence_lps: list[list[float]] = []
        fixed_sequence_ids: list[list[int]] = []
        for position, (position_lps, position_ids) in enumerate(zip(sequence_lps, sequence_ids, strict=True)):
            if len(position_lps) != len(position_ids):
                raise ValueError(f"batch {batch_index} position {position} has mismatched token/logprob counts")
            if len(position_lps) > top_k:
                raise ValueError(f"batch {batch_index} position {position} contains more than top_k entries")
            if len(set(position_ids)) != len(position_ids):
                raise ValueError(f"batch {batch_index} position {position} contains duplicate token ids")

            pairs = []
            for token_id, logprob in zip(position_ids, position_lps, strict=True):
                value = float(logprob)
                if not math.isfinite(value) or value > 1e-7:
                    raise ValueError(f"batch {batch_index} position {position} contains an invalid logprob")
                pairs.append((int(token_id), value))
            pairs.sort(key=lambda item: item[1], reverse=True)
            padding = top_k - len(pairs)
            fixed_sequence_ids.append([item[0] for item in pairs] + [0] * padding)
            fixed_sequence_lps.append([item[1] for item in pairs] + [-math.inf] * padding)
        normalized_ids.append(fixed_sequence_ids)
        normalized_lps.append(fixed_sequence_lps)

    actual = response.get("actual_logprobs")
    if actual is None:
        normalized_actual = [[[-math.inf] for _ in range(length)] for length in completion_lengths]
    else:
        if len(actual) != len(completion_lengths):
            raise ValueError("actual_logprobs batch size does not match completion_lengths")
        normalized_actual = []
        for batch_index, expected_length in enumerate(completion_lengths):
            if len(actual[batch_index]) != expected_length:
                raise ValueError(f"batch {batch_index} actual_logprobs has the wrong number of positions")
            normalized_actual.append([[float(row[0])] for row in actual[batch_index]])

    return {
        "logprobs": normalized_lps,
        "logprob_token_ids": normalized_ids,
        "actual_logprobs": normalized_actual,
    }
