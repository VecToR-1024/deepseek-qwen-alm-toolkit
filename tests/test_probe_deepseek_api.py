from __future__ import annotations

import json
from pathlib import Path

from scripts import probe_deepseek_api


class FakeClient:
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        assert kwargs["api_key"] == "test-key"

    def create_completion(self, messages, config):
        self.calls.append(config.as_api_kwargs(messages))
        return {
            "id": "probe-response",
            "model": config.model,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "x",
                        "reasoning_content": None,
                    },
                    "logprobs": {
                        "content": [
                            {
                                "token": "x",
                                "bytes": [120],
                                "logprob": -0.1,
                            }
                        ],
                        "reasoning_content": None,
                    },
                }
            ],
            "usage": {"completion_tokens": 1},
        }


def test_probe_actual_only_omits_top_logprobs_and_persists_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "raw.jsonl"
    FakeClient.calls = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(probe_deepseek_api, "DeepSeekClient", FakeClient)

    probe_deepseek_api.main(
        [
            "--prompt",
            "Return x",
            "--output",
            str(output),
            "--trace-profile",
            "actual_only",
            "--temperature",
            "0.2",
            "--max-tokens",
            "32",
        ]
    )

    assert len(FakeClient.calls) == 1
    request = FakeClient.calls[0]
    assert request["logprobs"] is True
    assert "top_logprobs" not in request
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 32
    record = json.loads(output.read_text(encoding="utf-8"))
    generation = record["request"]["generation_config"]
    assert generation["trace_profile"] == "actual_only"
    assert "top_logprobs" not in generation
