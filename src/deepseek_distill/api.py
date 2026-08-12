"""OpenAI-compatible DeepSeek V4 Pro request adapter.

The request shape follows DeepSeek's official Chat Completion and thinking-mode
examples:
https://api-docs.deepseek.com/api/create-chat-completion/
https://api-docs.deepseek.com/guides/thinking_mode
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .records import ACTUAL_ONLY_TRACE_PROFILE, RAW_SCHEMA_VERSION

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Reproducible non-thinking configuration for phase-one collection."""

    model: str = DEFAULT_MODEL
    temperature: float = 1.0
    top_p: float = 1.0
    top_logprobs: int | None = 20
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if self.top_logprobs is not None and not 0 <= self.top_logprobs <= 20:
            raise ValueError("top_logprobs must be between 0 and 20")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than zero and at most one")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer or null")

    def as_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "thinking": {"type": "disabled"},
            "temperature": self.temperature,
            "top_p": self.top_p,
            "logprobs": True,
        }
        if self.top_logprobs is None:
            metadata["trace_profile"] = ACTUAL_ONLY_TRACE_PROFILE
        else:
            metadata["top_logprobs"] = self.top_logprobs
        if self.max_tokens is not None:
            metadata["max_tokens"] = self.max_tokens
        return metadata

    def as_api_kwargs(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": copy.deepcopy(list(messages)),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "logprobs": True,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if self.top_logprobs is not None:
            kwargs["top_logprobs"] = self.top_logprobs
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs


class DeepSeekClient:
    """Small injectable wrapper around the OpenAI Python SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 2,
        sdk_client: Any | None = None,
    ) -> None:
        if sdk_client is not None:
            self._sdk_client = sdk_client
            return
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError('install the collection dependency with: pip install -e ".[collect]"') from error
        self._sdk_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def create_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        config: GenerationConfig,
    ) -> dict[str, Any]:
        _validate_messages(messages)
        response = self._sdk_client.chat.completions.create(**config.as_api_kwargs(messages))
        if isinstance(response, Mapping):
            return copy.deepcopy(dict(response))
        model_dump = getattr(response, "model_dump", None)
        if not callable(model_dump):
            raise TypeError("OpenAI SDK response must provide model_dump(mode='json')")
        dumped = model_dump(mode="json")
        if not isinstance(dumped, dict):
            raise TypeError("OpenAI SDK response model_dump must return a JSON object")
        return dumped


def build_success_record(
    *,
    record_id: str,
    messages: Sequence[Mapping[str, Any]],
    config: GenerationConfig,
    response: Mapping[str, Any],
    collected_at: str | None = None,
    task: Mapping[str, Any] | None = None,
    provider: Mapping[str, Any] | None = None,
    request_duration_seconds: float | None = None,
    prompt_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the append-only raw record written by probe and batch collection."""
    if not record_id:
        raise ValueError("record_id must be non-empty")
    _validate_messages(messages)
    record = {
        "schema_version": RAW_SCHEMA_VERSION,
        "id": record_id,
        "status": "ok",
        "collected_at": collected_at or utc_now_iso(),
        "request": _build_request_metadata(
            messages,
            config,
            prompt_contract=prompt_contract,
        ),
        "response": copy.deepcopy(dict(response)),
    }
    _add_collection_metadata(
        record,
        record_id=record_id,
        task=task,
        provider=provider,
        request_duration_seconds=request_duration_seconds,
    )
    return record


def build_error_record(
    *,
    record_id: str,
    messages: Sequence[Mapping[str, Any]],
    config: GenerationConfig,
    error: Exception,
    collected_at: str | None = None,
    task: Mapping[str, Any] | None = None,
    provider: Mapping[str, Any] | None = None,
    request_duration_seconds: float | None = None,
    prompt_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a terminal provider failure without leaking client credentials."""
    if not record_id:
        raise ValueError("record_id must be non-empty")
    _validate_messages(messages)
    record = {
        "schema_version": RAW_SCHEMA_VERSION,
        "id": record_id,
        "status": "error",
        "collected_at": collected_at or utc_now_iso(),
        "request": _build_request_metadata(
            messages,
            config,
            prompt_contract=prompt_contract,
        ),
        "response": None,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "details": _safe_error_details(error),
        },
    }
    _add_collection_metadata(
        record,
        record_id=record_id,
        task=task,
        provider=provider,
        request_duration_seconds=request_duration_seconds,
    )
    return record


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_messages(messages: Sequence[Mapping[str, Any]]) -> None:
    if not messages:
        raise ValueError("messages must be non-empty")
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{index}].role must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError(f"messages[{index}].content must be a string")


def _add_collection_metadata(
    record: dict[str, Any],
    *,
    record_id: str,
    task: Mapping[str, Any] | None,
    provider: Mapping[str, Any] | None,
    request_duration_seconds: float | None,
) -> None:
    if task is not None:
        if not isinstance(task, Mapping):
            raise ValueError("task must be an object or null")
        if task.get("id") != record_id:
            raise ValueError("task.id must match record_id")
        record["task"] = copy.deepcopy(dict(task))
    if provider is not None:
        if not isinstance(provider, Mapping):
            raise ValueError("provider must be an object or null")
        record["provider"] = copy.deepcopy(dict(provider))
    if request_duration_seconds is not None:
        if (
            isinstance(request_duration_seconds, bool)
            or not isinstance(request_duration_seconds, (int, float))
            or not math.isfinite(float(request_duration_seconds))
            or request_duration_seconds < 0
        ):
            raise ValueError("request_duration_seconds must be finite and non-negative")
        record["metrics"] = {
            "request_duration_seconds": float(request_duration_seconds),
        }


def _build_request_metadata(
    messages: Sequence[Mapping[str, Any]],
    config: GenerationConfig,
    *,
    prompt_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    request = {
        "model": config.model,
        "messages": copy.deepcopy(list(messages)),
        "generation_config": config.as_metadata(),
    }
    if prompt_contract is None:
        return request
    if not isinstance(prompt_contract, Mapping):
        raise ValueError("prompt_contract must be an object or null")
    for key in ("id", "interface_type"):
        value = prompt_contract.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"prompt_contract.{key} must be a non-empty string")
    request["prompt_contract"] = copy.deepcopy(dict(prompt_contract))
    return request


def _safe_error_details(error: Exception) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for attribute in ("status_code", "code", "param", "request_id", "type"):
        value = getattr(error, attribute, None)
        if value is None or isinstance(value, (str, int, float, bool)):
            if value is not None:
                details[attribute] = value
    return details
