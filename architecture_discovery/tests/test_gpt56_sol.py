import asyncio
from types import SimpleNamespace

import pytest
from openevolve.llm.openai import OpenAILLM

from common.gpt56_sol import GPT56SolProfile, TARGET_MODEL
from common.openevolve_runner import _build_model_config


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
