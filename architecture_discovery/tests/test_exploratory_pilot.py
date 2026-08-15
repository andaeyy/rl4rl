from __future__ import annotations

import json
from pathlib import Path

import pytest

from exploratory_pilot import (
    MODE,
    approval_text,
    build_provider_approval_plan,
    load_config,
    run_pilot,
    verify_pilot_artifacts,
)
from modal_boundary import ArtifactIntegrityError, FUNCTION_SPECS, function_spec
from scripts import launch_modal


def test_exploratory_preset_is_explicitly_non_scientific() -> None:
    config = load_config()
    assert config.mode == MODE
    assert config.scientific is False
    assert config.device == "cuda"
    assert config.provider_attempts_per_opportunity == 1
    assert config.provider_retries == 0
    assert config.modal_retries == 0
    assert function_spec("exploratory_c0c3_pilot").provider_secret is True
    assert all(
        spec.provider_secret is False
        for name, spec in FUNCTION_SPECS.items()
        if name in {"offline_smoke", "cuda_environment", "candidate_smoke", "checkpoint_resume"}
    )


def test_provider_approval_plan_is_hash_bound_and_secret_free() -> None:
    payload = build_provider_approval_plan(
        source_tree_sha256="a" * 64,
        image_source_sha256="b" * 64,
        cohort_id="exploratory-test-1",
    )
    assert payload["schema_name"] == "ExploratoryModalProviderApprovalPlan"
    assert payload["approval_plan_sha256"]
    assert "DISCOVERY_API_KEY" not in json.dumps(payload)
    assert "provider-backed" in approval_text(run_id="exploratory-test-1")


def test_fake_exploratory_run_round_trips_and_rejects_tampering(tmp_path: Path) -> None:
    run_id = "exploratory-local-test-1"
    result = run_pilot(tmp_path, run_id=run_id, provider=False)
    assert result["mode"] == MODE
    assert result["scientific"] is False
    study_dir = tmp_path / f"exploratory-{run_id}"
    verified = verify_pilot_artifacts(study_dir, run_id=study_dir.name)
    assert verified["verified"] is True

    summary = study_dir / "run_summary.json"
    summary.write_text(summary.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        verify_pilot_artifacts(study_dir, run_id=study_dir.name)


def test_exploratory_launcher_requires_separate_provider_approval() -> None:
    parser = launch_modal._parser()
    arguments = parser.parse_args(
        [
            "--action",
            "exploratory_c0c3_pilot",
            "--run-id",
            "exploratory-test-1",
            "--cohort-id",
            "exploratory-test-1",
            "--expected-image-source-sha256",
            "a" * 64,
            "--outer-cli-timeout-seconds",
            "1200",
            "--modal-cost-cap-usd",
            "0.25",
            "--modal-price-basis-path",
            "outputs/modal-price.json",
            "--modal-price-basis-sha256",
            "b" * 64,
            "--approved",
            "--candidate-resume-preflight-receipt-path",
            "outputs/preflight.json",
            "--candidate-resume-preflight-receipt-sha256",
            "c" * 64,
        ]
    )
    with pytest.raises(ValueError, match="separate provider cost approval"):
        launch_modal._validate_arguments(arguments)
