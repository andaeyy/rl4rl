from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from agents.greedy_autoresearch import run as greedy_run
from agents.semantic_autoresearch import run as semantic_run
from common.gpt56_sol import (
    OFFICIAL_OPENAI_API_BASE,
    TARGET_MODEL,
    GPT56SolProfile,
    resolve_provider_endpoint,
)
from common.provider_attempts import (
    PROVIDER_ATTEMPT_ACTION_ENV,
    PROVIDER_ATTEMPT_HARNESS_ENV,
    PROVIDER_ATTEMPT_LEDGER_ENV,
    ProviderAttemptLedger,
    generation_settings_sha256,
    load_provider_attempt_ledger,
    provider_attempt_totals,
)


def _request() -> dict[str, object]:
    return {
        "model": TARGET_MODEL,
        "messages": [{"role": "user", "content": "sensitive prompt"}],
        "reasoning_effort": "high",
        "max_completion_tokens": 16_384,
        "seed": 1,
    }


def _ledger(tmp_path):
    return ProviderAttemptLedger.create(
        tmp_path / "provider_attempts.jsonl",
        harness="greedy_autoresearch",
        action="one_opportunity_engineering_canary",
        controller_run_id="controller-run-1",
        api_endpoint=OFFICIAL_OPENAI_API_BASE,
        model=TARGET_MODEL,
        environ={},
    )


def test_success_attempt_is_exact_sanitized_and_reconciled(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    response = SimpleNamespace(
        id="chatcmpl-test123",
        _request_id="req_test123",
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
        ),
    )

    assert ledger.record_call(_request(), lambda: response) is response

    records = load_provider_attempt_ledger(ledger.path)
    assert len(records) == 1
    record = records[0]
    assert record.attempt_ordinal == 1
    assert record.status == "success"
    assert record.generation_settings_sha256 == generation_settings_sha256(
        _request()
    )
    assert record.provider_response_id == "chatcmpl-test123"
    assert record.provider_request_id == "req_test123"
    assert record.usage_known is True
    assert provider_attempt_totals(records) == {
        "attempt_count": 1,
        "success_count": 1,
        "error_count": 0,
        "usage_known_count": 1,
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    raw = ledger.path.read_text(encoding="utf-8")
    assert "sensitive prompt" not in raw
    assert set(json.loads(raw)) == record.FIELDS


def test_failed_attempt_records_only_error_class_and_exposed_request_id(
    tmp_path,
) -> None:
    ledger = _ledger(tmp_path)

    class ProviderFailure(RuntimeError):
        request_id = "req_failure123"

    def fail():
        raise ProviderFailure("secret key and raw provider body must not persist")

    with pytest.raises(ProviderFailure):
        ledger.record_call(_request(), fail)

    record = load_provider_attempt_ledger(ledger.path)[0]
    assert record.status == "error"
    assert record.error_class == "ProviderFailure"
    assert record.provider_request_id == "req_failure123"
    assert record.provider_response_id is None
    assert record.usage_known is False
    assert record.input_tokens is None
    raw = ledger.path.read_text(encoding="utf-8")
    assert "secret key" not in raw
    assert "raw provider body" not in raw


def test_unknown_or_inconsistent_usage_is_explicitly_unknown(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    response = SimpleNamespace(
        id="chatcmpl-test123",
        _request_id="req_test123",
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=999,
        ),
    )

    ledger.record_call(_request(), lambda: response)

    record = load_provider_attempt_ledger(ledger.path)[0]
    assert record.usage_known is False
    assert (record.input_tokens, record.output_tokens, record.total_tokens) == (
        None,
        None,
        None,
    )


def test_ledger_is_create_only_and_records_have_contiguous_ordinals(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    response = SimpleNamespace(
        id="chatcmpl-test123",
        _request_id="req_test123",
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        ),
    )
    ledger.record_call(_request(), lambda: response)
    ledger.record_call(_request(), lambda: response)

    assert [
        record.attempt_ordinal
        for record in load_provider_attempt_ledger(ledger.path)
    ] == [1, 2]
    with pytest.raises(FileExistsError):
        _ledger(tmp_path)


def test_ledger_rejects_symlinks_in_any_ancestor_for_create_open_and_load(
    tmp_path,
) -> None:
    real_root = tmp_path / "real-root"
    real_directory = real_root / "nested"
    real_directory.mkdir(parents=True)
    alias = tmp_path / "alias-root"
    alias.symlink_to(real_root, target_is_directory=True)
    redirected = alias / "nested" / "provider_attempts.jsonl"

    with pytest.raises(ValueError, match="components may not be symlinks"):
        ProviderAttemptLedger.create(
            redirected,
            harness="greedy_autoresearch",
            action="one_opportunity_engineering_canary",
            controller_run_id="controller-run-1",
            api_endpoint=OFFICIAL_OPENAI_API_BASE,
            model=TARGET_MODEL,
            environ={},
        )

    ledger = ProviderAttemptLedger.create(
        real_directory / "provider_attempts.jsonl",
        harness="greedy_autoresearch",
        action="one_opportunity_engineering_canary",
        controller_run_id="controller-run-1",
        api_endpoint=OFFICIAL_OPENAI_API_BASE,
        model=TARGET_MODEL,
        environ={},
    )
    with pytest.raises(ValueError, match="components may not be symlinks"):
        load_provider_attempt_ledger(redirected)
    with pytest.raises(ValueError, match="components may not be symlinks"):
        ProviderAttemptLedger.open_existing_from_environment(
            api_endpoint=OFFICIAL_OPENAI_API_BASE,
            model=TARGET_MODEL,
            environ={
                PROVIDER_ATTEMPT_LEDGER_ENV: str(redirected),
                PROVIDER_ATTEMPT_HARNESS_ENV: "greedy_autoresearch",
                PROVIDER_ATTEMPT_ACTION_ENV: "one_opportunity_engineering_canary",
                "DISCOVERY_RUN_ID": "controller-run-1",
            },
        )
    assert ledger.path == real_directory / "provider_attempts.jsonl"


def test_ledger_rejects_hard_links_and_inode_substitution_before_provider_call(
    tmp_path,
) -> None:
    ledger = _ledger(tmp_path)
    hard_link = tmp_path / "provider-attempts-hard-link.jsonl"
    os.link(ledger.path, hard_link)
    operation_started = False

    def operation():
        nonlocal operation_started
        operation_started = True
        raise AssertionError("provider operation must not start")

    with pytest.raises(ValueError, match="one regular file"):
        ledger.record_call(_request(), operation)
    assert operation_started is False
    with pytest.raises(ValueError, match="one regular file"):
        load_provider_attempt_ledger(ledger.path)

    hard_link.unlink()
    ledger.path.unlink()
    ledger.path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="inode changed"):
        ledger.record_call(_request(), operation)
    assert operation_started is False


def test_ledger_rechecks_hard_link_count_before_terminal_append(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    hard_link = tmp_path / "provider-attempts-raced-hard-link.jsonl"
    response = SimpleNamespace(
        id="chatcmpl-test123",
        _request_id="req_test123",
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        ),
    )

    def operation():
        os.link(ledger.path, hard_link)
        return response

    with pytest.raises(ValueError, match="one regular file"):
        ledger.record_call(_request(), operation)
    hard_link.unlink()
    assert load_provider_attempt_ledger(ledger.path) == ()


def test_loader_rejects_unknown_fields_even_when_json_is_valid(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    response = SimpleNamespace(
        id="chatcmpl-test123",
        _request_id="req_test123",
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        ),
    )
    ledger.record_call(_request(), lambda: response)
    payload = json.loads(ledger.path.read_text(encoding="utf-8"))
    payload["error_message"] = "forbidden"
    ledger.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected or missing fields"):
        load_provider_attempt_ledger(ledger.path)


@pytest.mark.parametrize(
    ("module", "harness"),
    [
        (greedy_run, "greedy_autoresearch"),
        (semantic_run, "semantic_autoresearch"),
    ],
)
def test_native_provider_records_the_exact_sdk_transport_attempt(
    tmp_path,
    monkeypatch,
    module,
    harness,
) -> None:
    response = SimpleNamespace(
        id=f"chatcmpl-{harness}",
        _request_id=f"req_{harness}",
        usage=SimpleNamespace(
            prompt_tokens=4,
            completion_tokens=6,
            total_tokens=10,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="offline IR"))],
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    monkeypatch.setattr(module, "OpenAI", lambda **_kwargs: fake_client)
    generation = GPT56SolProfile.resolve(
        model=TARGET_MODEL,
        seed=1,
        default_reasoning_effort="high",
        default_max_completion_tokens=16_384,
        default_timeout_seconds=300,
        default_retries=0,
        default_retry_delay_seconds=0,
        environ={},
    )
    provider = module.OpenAIProposalProvider(
        api_key="offline-test-key",
        endpoint=resolve_provider_endpoint(
            OFFICIAL_OPENAI_API_BASE,
            scientific=False,
        ),
        generation=generation,
    )
    provider.bind_attempt_ledger(
        tmp_path,
        run_id=f"{harness}-run-1",
        action="one_opportunity_engineering_canary",
    )

    result = provider.generate([{"role": "user", "content": "sensitive"}])

    assert result.text == "offline IR"
    record = load_provider_attempt_ledger(tmp_path / "provider_attempts.jsonl")[0]
    assert record.harness == harness
    assert record.status == "success"
    assert record.provider_response_id == f"chatcmpl-{harness}"
    assert record.provider_request_id == f"req_{harness}"
    assert record.total_tokens == 10
