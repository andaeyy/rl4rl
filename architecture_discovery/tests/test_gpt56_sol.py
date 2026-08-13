import asyncio
from types import SimpleNamespace

import pytest
from openevolve.llm.openai import OpenAILLM

from common.gpt56_sol import (
    GPT56SolProfile,
    OFFICIAL_OPENAI_API_BASE,
    TARGET_MODEL,
    resolve_provider_endpoint,
)
from common.openevolve_runner import _build_model_config
from common.provider_attempts import (
    PROVIDER_ATTEMPT_ACTION_ENV,
    PROVIDER_ATTEMPT_HARNESS_ENV,
    PROVIDER_ATTEMPT_LEDGER_ENV,
    ProviderAttemptLedger,
    load_provider_attempt_ledger,
)


def _profile(environ=None):
    return GPT56SolProfile.resolve(
        model=TARGET_MODEL,
        seed=17,
        default_reasoning_effort="high",
        default_max_completion_tokens=16384,
        default_timeout_seconds=300,
        default_retries=2,
        default_retry_delay_seconds=3,
        environ={} if environ is None else environ,
    )


def test_gpt56_chat_request_uses_reasoning_fields_and_omits_sampling():
    messages = [{"role": "user", "content": "Propose one architecture."}]
    request = _profile().chat_completion_request(messages)

    assert request == {
        "model": "gpt-5.6-sol",
        "messages": messages,
        "reasoning_effort": "high",
        "max_completion_tokens": 16384,
        "seed": 17,
    }
    assert "temperature" not in request
    assert "top_p" not in request
    assert "max_tokens" not in request


def test_profile_rejects_an_accidentally_stale_model():
    with pytest.raises(ValueError, match="pinned to 'gpt-5.6-sol'"):
        GPT56SolProfile.resolve(
            model="gpt-4.1",
            seed=1,
            default_reasoning_effort="high",
            default_max_completion_tokens=16384,
            default_timeout_seconds=300,
            default_retries=2,
            default_retry_delay_seconds=3,
            environ={},
        )


def test_profile_rejects_legacy_max_tokens_override():
    with pytest.raises(ValueError, match="legacy GPT-4-style override"):
        _profile({"DISCOVERY_MAX_TOKENS": "4096"})


def test_profile_applies_explicit_gpt56_overrides():
    profile = _profile(
        {
            "DISCOVERY_REASONING_EFFORT": "xhigh",
            "DISCOVERY_MAX_COMPLETION_TOKENS": "32768",
            "DISCOVERY_REQUEST_TIMEOUT_SECONDS": "600",
            "DISCOVERY_REQUEST_RETRIES": "4",
            "DISCOVERY_RETRY_DELAY_SECONDS": "5",
        }
    )

    assert profile.reasoning_effort == "xhigh"
    assert profile.max_completion_tokens == 32768
    assert profile.timeout_seconds == 600
    assert profile.retries == 4
    assert profile.retry_delay_seconds == 5


def test_scientific_profile_rejects_conflicting_environment_overrides():
    with pytest.raises(ValueError, match="scientific generation settings are frozen"):
        GPT56SolProfile.resolve(
            model=TARGET_MODEL,
            seed=17,
            default_reasoning_effort="high",
            default_max_completion_tokens=16384,
            default_timeout_seconds=300,
            default_retries=2,
            default_retry_delay_seconds=3,
            environ={"DISCOVERY_REQUEST_RETRIES": "4"},
            allow_environment_overrides=False,
        )


def test_scientific_profile_accepts_matching_bound_environment_values():
    profile = GPT56SolProfile.resolve(
        model=TARGET_MODEL,
        seed=17,
        default_reasoning_effort="high",
        default_max_completion_tokens=16384,
        default_timeout_seconds=300,
        default_retries=2,
        default_retry_delay_seconds=3,
        environ={"DISCOVERY_REQUEST_RETRIES": "2"},
        allow_environment_overrides=False,
    )

    assert profile.retries == 2
    assert profile.manifest_fields()["request_settings_source"] == (
        "frozen_controller_configuration"
    )


def test_provider_endpoint_is_normalized_and_scientific_endpoint_is_frozen():
    endpoint = resolve_provider_endpoint(
        "HTTPS://API.OPENAI.COM/v1/",
        scientific=True,
    )

    assert endpoint.base_url == OFFICIAL_OPENAI_API_BASE
    assert endpoint.provider_identity == "openai_official"
    assert "key" not in endpoint.manifest_fields()

    with pytest.raises(ValueError, match="pinned to the official OpenAI API"):
        resolve_provider_endpoint("https://proxy.example/v1", scientific=True)


def test_provider_endpoint_rejects_embedded_secrets():
    with pytest.raises(ValueError, match="may not contain credentials"):
        resolve_provider_endpoint(
            "https://user:secret@api.openai.com/v1",
            scientific=False,
        )


def test_openevolve_adapter_sends_the_same_effective_gpt56_fields():
    generation = _profile()
    model_config = _build_model_config(
        generation,
        api_base="https://api.openai.com/v1",
        api_key="offline-test-key",
    )
    adapter = OpenAILLM(model_config)
    assert adapter.client.max_retries == 0

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="offline response")
                    )
                ]
            )

    adapter.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    result = asyncio.run(
        adapter.generate_with_context(
            "system prompt",
            [{"role": "user", "content": "Propose one architecture."}],
        )
    )

    assert result == "offline response"
    assert captured["model"] == "gpt-5.6-sol"
    assert captured["reasoning_effort"] == "high"
    assert captured["max_completion_tokens"] == 16384
    assert captured["seed"] == 17
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "max_tokens" not in captured


def test_controlled_openevolve_transport_records_the_actual_sdk_attempt(
    tmp_path,
    monkeypatch,
):
    generation = GPT56SolProfile.resolve(
        model=TARGET_MODEL,
        seed=1,
        default_reasoning_effort="high",
        default_max_completion_tokens=16384,
        default_timeout_seconds=300,
        default_retries=0,
        default_retry_delay_seconds=0,
        environ={},
    )
    path = tmp_path / "provider_attempts.jsonl"
    ProviderAttemptLedger.create(
        path,
        harness="openevolve_generic",
        action="one_opportunity_engineering_canary",
        controller_run_id="controller-run-1",
        api_endpoint=OFFICIAL_OPENAI_API_BASE,
        model=TARGET_MODEL,
        environ={},
    )
    monkeypatch.setenv(PROVIDER_ATTEMPT_LEDGER_ENV, str(path))
    monkeypatch.setenv(PROVIDER_ATTEMPT_HARNESS_ENV, "openevolve_generic")
    monkeypatch.setenv(
        PROVIDER_ATTEMPT_ACTION_ENV,
        "one_opportunity_engineering_canary",
    )
    monkeypatch.setenv("DISCOVERY_RUN_ID", "controller-run-1")

    adapter = OpenAILLM(
        _build_model_config(
            generation,
            api_base=OFFICIAL_OPENAI_API_BASE,
            api_key="offline-test-key",
        )
    )
    response = SimpleNamespace(
        id="chatcmpl-openevolve123",
        _request_id="req_openevolve123",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="offline response"))],
    )
    adapter.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )

    result = asyncio.run(
        adapter.generate_with_context(
            "system prompt",
            [{"role": "user", "content": "Propose one architecture."}],
        )
    )

    assert result == "offline response"
    record = load_provider_attempt_ledger(path)[0]
    assert record.harness == "openevolve_generic"
    assert record.status == "success"
    assert record.provider_response_id == "chatcmpl-openevolve123"
    assert record.provider_request_id == "req_openevolve123"
    assert record.total_tokens == 15
