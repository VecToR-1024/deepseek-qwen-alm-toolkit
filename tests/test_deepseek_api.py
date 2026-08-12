from deepseek_distill.api import DeepSeekClient, GenerationConfig, build_success_record


class FakeResponse:
    def model_dump(self, *, mode: str) -> dict:
        assert mode == "json"
        return {
            "id": "chatcmpl-test",
            "model": "deepseek-v4-pro",
            "choices": [],
            "usage": {"total_tokens": 7},
        }


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()


class FakeSDKClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions()


def test_deepseek_client_sends_non_thinking_top20_request() -> None:
    sdk = FakeSDKClient()
    client = DeepSeekClient(sdk_client=sdk)
    messages = [{"role": "user", "content": "Write a Python function"}]

    response = client.create_completion(messages, GenerationConfig())

    assert response["id"] == "chatcmpl-test"
    assert sdk.chat.completions.kwargs == {
        "model": "deepseek-v4-pro",
        "messages": messages,
        "temperature": 1.0,
        "top_p": 1.0,
        "logprobs": True,
        "top_logprobs": 20,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def test_deepseek_client_actual_only_omits_top_logprobs() -> None:
    sdk = FakeSDKClient()
    client = DeepSeekClient(sdk_client=sdk)
    messages = [{"role": "user", "content": "Write a Python function"}]

    client.create_completion(
        messages,
        GenerationConfig(top_logprobs=None),
    )

    assert sdk.chat.completions.kwargs == {
        "model": "deepseek-v4-pro",
        "messages": messages,
        "temperature": 1.0,
        "top_p": 1.0,
        "logprobs": True,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def test_actual_only_metadata_persists_profile_without_fake_top_k() -> None:
    metadata = GenerationConfig(top_logprobs=None).as_metadata()

    assert metadata["trace_profile"] == "actual_only"
    assert metadata["logprobs"] is True
    assert "top_logprobs" not in metadata


def test_generation_config_rejects_top_logprobs_above_provider_limit() -> None:
    try:
        GenerationConfig(top_logprobs=21)
    except ValueError as error:
        assert "between 0 and 20" in str(error)
    else:
        raise AssertionError("GenerationConfig accepted top_logprobs=21")


def test_generation_config_persists_every_parameter_actually_sent() -> None:
    config = GenerationConfig(
        model="deepseek-v4-pro",
        temperature=0.2,
        top_p=1.0,
        top_logprobs=20,
        max_tokens=2048,
    )

    metadata = config.as_metadata()
    kwargs = config.as_api_kwargs([{"role": "user", "content": "solve"}])

    assert metadata["max_tokens"] == 2048
    assert kwargs["max_tokens"] == 2048
    assert metadata["thinking"] == kwargs["extra_body"]["thinking"]


def test_success_record_contains_request_metadata_but_no_credentials() -> None:
    config = GenerationConfig()
    messages = [{"role": "user", "content": "Say hi"}]

    record = build_success_record(
        record_id="problem_1",
        messages=messages,
        config=config,
        response={"id": "response_1", "choices": []},
        collected_at="2026-07-20T01:02:03Z",
        prompt_contract={
            "id": "deepseek.python.clean.v2",
            "interface_type": "function",
        },
    )

    assert record["request"]["generation_config"]["thinking"] == {"type": "disabled"}
    assert record["request"]["prompt_contract"] == {
        "id": "deepseek.python.clean.v2",
        "interface_type": "function",
    }
    assert record["response"]["id"] == "response_1"
    assert "api_key" not in repr(record).lower()


def test_success_record_preserves_task_provider_and_duration_additively() -> None:
    task = {"schema_version": "coding.task.mbpp.v1", "id": "mbpp_601", "tests": ["hidden"]}

    record = build_success_record(
        record_id="mbpp_601",
        messages=[{"role": "user", "content": "problem only"}],
        config=GenerationConfig(),
        response={"id": "response_1", "choices": []},
        task=task,
        provider={"name": "DeepSeek", "base_url": "https://api.deepseek.com"},
        request_duration_seconds=1.25,
    )

    assert record["task"] == task
    assert record["provider"]["name"] == "DeepSeek"
    assert record["metrics"]["request_duration_seconds"] == 1.25
