"""Pluggable HTTP client implementing TRL's sequence-logprob contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .contracts import normalize_sequence_logprobs


class SequenceLogprobClient(Protocol):
    """The minimal client surface consumed by TRL DistillationTrainer."""

    supports_actual_logprobs: bool

    def get_sequence_logprobs(
        self,
        sequences: list[list[int]],
        prompt_lengths: list[int],
        top_logprobs: int = 100,
        temperature: float = 1.0,
    ) -> dict[str, list]: ...


class HttpSequenceLogprobClient:
    """POST token sequences to a provider-specific scoring endpoint.

    The endpoint receives token IDs rather than text, so the provider and the
    student must use the same token-id mapping. Its JSON response may contain
    fewer than ``top_logprobs`` entries; the client pads and validates it.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
        supports_actual_logprobs: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.headers = {"Content-Type": "application/json", **dict(headers or {})}
        self.timeout = timeout
        self.supports_actual_logprobs = supports_actual_logprobs

    def get_sequence_logprobs(
        self,
        sequences: list[list[int]],
        prompt_lengths: list[int],
        top_logprobs: int = 100,
        temperature: float = 1.0,
    ) -> dict[str, list]:
        completion_lengths = self._completion_lengths(sequences, prompt_lengths)
        payload = {
            "sequences": sequences,
            "prompt_lengths": prompt_lengths,
            "top_logprobs": top_logprobs,
            "temperature": temperature,
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self.headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - endpoint is user-configured
                response_json: Any = json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"teacher API returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"teacher API request failed: {error.reason}") from error

        if not isinstance(response_json, Mapping):
            raise ValueError("teacher API must return a JSON object")
        return normalize_sequence_logprobs(
            response_json,
            completion_lengths=completion_lengths,
            top_k=top_logprobs,
        )

    @staticmethod
    def _completion_lengths(
        sequences: Sequence[Sequence[int]],
        prompt_lengths: Sequence[int],
    ) -> list[int]:
        if len(sequences) != len(prompt_lengths):
            raise ValueError("sequences and prompt_lengths must have the same batch size")
        lengths = []
        for index, (sequence, prompt_length) in enumerate(zip(sequences, prompt_lengths, strict=True)):
            if prompt_length < 0 or prompt_length > len(sequence):
                raise ValueError(f"invalid prompt length for sequence {index}")
            lengths.append(len(sequence) - prompt_length)
        return lengths
