from __future__ import annotations

import json
import socket

import openai
import pytest
from modal_boundary import (
    CANARY_ORDER,
    CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS,
    FUNCTION_TIMEOUT_SECONDS,
    PROVIDER_ATTEMPT_FINALIZATION_RESERVE_SECONDS,
    PROVIDER_REQUEST_TIMEOUT_SECONDS,
    ModalLiveCohortIdentity,
    build_image_source_manifest,
    canonical_sha256,
    modal_live_cohort_root,
)
from scripts.record_local_engineering_evidence import source_tree_sha256
from scripts.provider_canary_plan import (
    ROOT,
    _config,
    _resolved_profile,
    build_provider_canary_approval_plan,
    create_provider_canary_approval_plan,
    verify_provider_canary_approval_plan,
)


def _plan_kwargs() -> dict[str, str]:
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=source_tree_sha256(ROOT),
        image_source_sha256=build_image_source_manifest(ROOT).manifest_sha256,
        cohort_id="provider-plan-test",
    )
    preflight = (
        modal_live_cohort_root(identity)
        / "components"
        / "candidate_resume_preflight_receipts"
        / "v2.0"
        / (("a" * 64) + ".json")
    )
    return {
        "source_tree_sha256": identity.source_tree_sha256,
        "cohort_id": identity.cohort_id,
        "candidate_resume_preflight_receipt_path": preflight.as_posix(),
        "candidate_resume_preflight_receipt_sha256": "b" * 64,
    }


def test_provider_canary_plan_is_exactly_four_cost_free_attempts(monkeypatch):
    def forbidden_client(*_args, **_kwargs):
        raise AssertionError("approval planning initialized an OpenAI client")

    def forbidden_network(*_args, **_kwargs):
        raise AssertionError("approval planning attempted network access")

    monkeypatch.setattr(openai, "OpenAI", forbidden_client)
    monkeypatch.setattr(socket, "create_connection", forbidden_network)

    plan = build_provider_canary_approval_plan(**_plan_kwargs())

    assert plan["schema_name"] == "ProviderCanaryApprovalPlan"
    assert plan["schema_version"] == "1.2"
    assert plan["source_tree_sha256"] == _plan_kwargs()["source_tree_sha256"]
    assert plan["cohort_id"] == "provider-plan-test"
    assert plan["candidate_resume_preflight_receipt"] == {
        "path": _plan_kwargs()["candidate_resume_preflight_receipt_path"],
        "sha256": "b" * 64,
    }
    assert verify_provider_canary_approval_plan(plan) == plan[
        "approval_plan_sha256"
    ]
    unsigned = dict(plan)
    digest = unsigned.pop("approval_plan_sha256")
    assert digest == canonical_sha256(unsigned)
    assert plan["approval_plan_sha256_scope"] == (
        "canonical_json_sha256_of_complete_payload_excluding_approval_plan_sha256"
    )
    assert plan["harness_order"] == list(CANARY_ORDER)
    assert [item["harness"] for item in plan["harnesses"]] == list(CANARY_ORDER)
    assert plan["totals"] == {
        "harness_count": 4,
        "maximum_requests": 4,
        "conservative_input_token_ceiling": 81_920,
        "requested_completion_token_ceiling": 65_536,
    }
    assert plan["provider_calls_started"] == 0
    assert plan["modal_calls_started"] == 0
    assert plan["openai_clients_initialized"] == 0
    assert all(item["maximum_attempts"] == 1 for item in plan["harnesses"])
    assert all(item["request_settings"]["retries"] == 0 for item in plan["harnesses"])
    assert plan["execution_deadlines"] == {
        "function_timeout_seconds": 300,
        "controller_subprocess_timeout_seconds": 240,
        "provider_request_timeout_seconds": 180,
        "provider_attempt_finalization_reserve_seconds": 60,
    }


def test_provider_request_timeout_preserves_terminal_ledger_reserve():
    plan = build_provider_canary_approval_plan(**_plan_kwargs())

    assert all(
        item["request_settings"]["timeout_seconds"]
        == PROVIDER_REQUEST_TIMEOUT_SECONDS
        for item in plan["harnesses"]
    )
    assert (
        PROVIDER_REQUEST_TIMEOUT_SECONDS
        + PROVIDER_ATTEMPT_FINALIZATION_RESERVE_SECONDS
        == CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS
    )
    assert CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS < FUNCTION_TIMEOUT_SECONDS


def test_provider_canary_plan_uses_actual_prompt_constructors_and_safe_bounds():
    plan = build_provider_canary_approval_plan(**_plan_kwargs())
    by_harness = {item["harness"]: item for item in plan["harnesses"]}

    assert [
        by_harness[harness]["first_opportunity"][
            "approval_template_message_content_bytes"
        ]
        for harness in CANARY_ORDER
    ] == [12_375, 13_201, 20_517, 21_414]
    assert [
        by_harness[harness]["first_opportunity"][
            "live_request_payload_bytes_upper_bound"
        ]
        for harness in CANARY_ORDER
    ] == [13_493, 14_327, 22_813, 23_718]
    assert [
        by_harness[harness]["first_opportunity"][
            "conservative_input_token_ceiling"
        ]
        for harness in CANARY_ORDER
    ] == [16_384, 16_384, 24_576, 24_576]
    for item in plan["harnesses"]:
        first = item["first_opportunity"]
        assert len(first["request_payload_sha256"]) == 64
        assert first["live_request_payload_sha256"] is None
        assert first["live_request_payload_bytes_upper_bound"] <= first[
            "conservative_input_token_ceiling"
        ]
        assert item["request_settings"]["max_completion_tokens"] == 16_384
        assert item["api_endpoint"] == "https://api.openai.com/v1"
        assert item["model"] == "gpt-5.6-sol"

    for harness in CANARY_ORDER[:2]:
        first = by_harness[harness]["first_opportunity"]
        assert first["live_request_payload_bytes_exact"] is not None
        assert first["size_invariance_check"]["same_canonical_request_bytes"] is True
    for harness in CANARY_ORDER[2:]:
        first = by_harness[harness]["first_opportunity"]
        assert first["live_request_payload_bytes_exact"] is None
        assert first["size_bound_derivation"].startswith("maximum_of_actual")


def test_provider_canary_plan_rejects_retry_drift():
    config = _config(ROOT / "agents" / "greedy_autoresearch" / "config.yaml")
    config["retries"] = 1

    with pytest.raises(ValueError, match="not frozen"):
        _resolved_profile(config, openevolve=False)


def test_provider_canary_plan_output_is_create_only_and_reopened(tmp_path):
    output = tmp_path / "provider-canary-approval-plan.json"

    plan = create_provider_canary_approval_plan(output, **_plan_kwargs())
    reopened = json.loads(output.read_text(encoding="utf-8"))

    assert reopened == plan
    assert verify_provider_canary_approval_plan(reopened) == plan[
        "approval_plan_sha256"
    ]
    with pytest.raises(FileExistsError):
        create_provider_canary_approval_plan(output, **_plan_kwargs())
